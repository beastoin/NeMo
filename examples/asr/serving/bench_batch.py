#!/usr/bin/env python3
"""
Deterministic batch ASR benchmark with WER.

Downloads LibriSpeech test-clean (200 samples), runs warmup, concurrency sweep,
sustained load, computes WER using wer_utils (Whisper normalization), and outputs
a structured JSON report.

Usage:
    python3 bench_batch.py --server http://localhost:8000
    python3 bench_batch.py --server http://localhost:8000 --sustained-rounds 8
    python3 bench_batch.py --server http://localhost:8000 --concurrency 1,8,16,32,64
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bench_batch")

LIBRISPEECH_URL = (
    "https://www.openslr.org/resources/12/test-clean.tar.gz"
)
DATA_DIR = Path("/tmp/librispeech-test-clean")
WAV_DIR = DATA_DIR / "wav"
REF_FILE = DATA_DIR / "references.tsv"
MAX_SAMPLES = 200


def ensure_librispeech():
    """Download and extract LibriSpeech test-clean if not cached."""
    if WAV_DIR.exists() and len(list(WAV_DIR.glob("*.wav"))) >= MAX_SAMPLES and REF_FILE.exists():
        n = len(list(WAV_DIR.glob("*.wav")))
        log.info(f"LibriSpeech test-clean cached: {n} WAV files")
        return

    log.info("Downloading LibriSpeech test-clean...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WAV_DIR.mkdir(exist_ok=True)
    tar_path = DATA_DIR / "test-clean.tar.gz"

    if not tar_path.exists():
        urllib.request.urlretrieve(LIBRISPEECH_URL, tar_path)
        log.info(f"Downloaded {tar_path.stat().st_size / 1e6:.0f}MB")

    log.info("Extracting...")
    refs = {}
    count = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if member.name.endswith(".trans.txt"):
                f = tar.extractfile(member)
                if f:
                    for line in f.read().decode().strip().split("\n"):
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            refs[parts[0]] = parts[1]

            if member.name.endswith(".flac"):
                utt_id = Path(member.name).stem
                if count >= MAX_SAMPLES:
                    continue
                f = tar.extractfile(member)
                if f:
                    import soundfile as sf
                    import io
                    import struct

                    audio, sr = sf.read(io.BytesIO(f.read()), dtype="int16")
                    wav_path = WAV_DIR / f"{utt_id}.wav"
                    num_samples = len(audio)
                    data_size = num_samples * 2
                    with open(wav_path, "wb") as wf:
                        wf.write(b"RIFF")
                        wf.write(struct.pack("<I", 36 + data_size))
                        wf.write(b"WAVE")
                        wf.write(b"fmt ")
                        wf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
                        wf.write(b"data")
                        wf.write(struct.pack("<I", data_size))
                        wf.write(audio.tobytes())
                    count += 1

    with open(REF_FILE, "w") as f:
        for utt_id in sorted(refs):
            f.write(f"{utt_id}\t{refs[utt_id]}\n")

    n = len(list(WAV_DIR.glob("*.wav")))
    log.info(f"Extracted {n} WAV files, {len(refs)} references")


def load_references():
    """Load reference transcripts as {utt_id: text}."""
    refs = {}
    with open(REF_FILE) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                refs[parts[0]] = parts[1]
    return refs


def compute_wer(references, hypotheses):
    """Compute WER using wer_utils (Whisper normalization). Falls back to basic normalization."""
    try:
        from wer_utils import corpus_wer, pair_wer

        wer_val = corpus_wer(references, hypotheses)
        per_utt = [pair_wer(r, h) for r, h in zip(references, hypotheses)]
        return wer_val, per_utt
    except ImportError:
        log.warning("wer_utils not importable (missing jiwer/whisper-normalizer), using basic normalization")
        return _basic_wer(references, hypotheses)


def _basic_wer(references, hypotheses):
    """Fallback WER with basic normalization (lowercase + strip punctuation)."""
    import re

    def normalize(text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def edit_distance(ref_words, hyp_words):
        n, m = len(ref_words), len(hyp_words)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[n][m]

    total_errors = 0
    total_words = 0
    per_utt = []
    for ref, hyp in zip(references, hypotheses):
        ref_n = normalize(ref).split()
        hyp_n = normalize(hyp).split()
        errs = edit_distance(ref_n, hyp_n)
        total_errors += errs
        total_words += len(ref_n)
        per_utt.append(errs / max(len(ref_n), 1))

    return total_errors / max(total_words, 1), per_utt


def get_wav_duration(wav_path):
    """Get audio duration in seconds from WAV file header."""
    with open(wav_path, "rb") as f:
        f.read(24)  # skip to byte rate offset
        import struct

        f.seek(28)
        byte_rate = struct.unpack("<I", f.read(4))[0]
        f.seek(40)
        data_size = struct.unpack("<I", f.read(4))[0]
        return data_size / byte_rate if byte_rate > 0 else 0


async def transcribe_file(session, url, wav_path, semaphore):
    """Send one file to /v1/transcribe, return result dict."""
    async with semaphore:
        t0 = time.monotonic()
        audio_dur = get_wav_duration(wav_path)
        try:
            data = aiohttp.FormData()
            data.add_field(
                "file",
                open(wav_path, "rb"),
                filename=os.path.basename(wav_path),
                content_type="audio/wav",
            )
            async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                elapsed = time.monotonic() - t0
                if resp.status == 200:
                    result = await resp.json()
                    return {
                        "utt_id": Path(wav_path).stem,
                        "text": result.get("text", ""),
                        "elapsed": elapsed,
                        "audio_dur": audio_dur,
                        "status": "ok",
                    }
                else:
                    body = await resp.text()
                    return {
                        "utt_id": Path(wav_path).stem,
                        "error": f"HTTP {resp.status}: {body[:200]}",
                        "elapsed": elapsed,
                        "status": "error",
                    }
        except Exception as e:
            return {
                "utt_id": Path(wav_path).stem,
                "error": str(e)[:200],
                "elapsed": time.monotonic() - t0,
                "status": "error",
            }


async def run_sweep(url, wav_files, concurrency, repeat=1):
    """Run one concurrency level, return results list."""
    files = wav_files * repeat
    sem = asyncio.Semaphore(concurrency)
    t0 = time.monotonic()
    async with aiohttp.ClientSession() as session:
        tasks = [transcribe_file(session, url, f, sem) for f in files]
        results = await asyncio.gather(*tasks)
    wall = time.monotonic() - t0
    return list(results), wall


def summarize_sweep(results, wall_time, concurrency):
    """Compute throughput/latency/RTFx summary for one concurrency level."""
    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "error"]
    latencies = sorted(r["elapsed"] for r in ok)
    total_audio = sum(r.get("audio_dur", 0) for r in ok)

    summary = {
        "concurrency": concurrency,
        "total": len(results),
        "ok": len(ok),
        "failures": len(failed),
        "wall_s": round(wall_time, 2),
        "rps": round(len(ok) / wall_time, 2) if wall_time > 0 else 0,
        "rtfx": round(total_audio / wall_time, 2) if wall_time > 0 else 0,
        "rtf": round(wall_time / total_audio, 4) if total_audio > 0 else 0,
        "total_audio_s": round(total_audio, 1),
        "sess_per_min": round(len(ok) / (wall_time / 60), 1) if wall_time > 0 else 0,
    }
    if latencies:
        summary["p50_s"] = round(latencies[len(latencies) // 2], 3)
        summary["p99_s"] = round(latencies[int(len(latencies) * 0.99)], 3)
        summary["min_s"] = round(latencies[0], 3)
        summary["max_s"] = round(latencies[-1], 3)

    return summary


async def main():
    parser = argparse.ArgumentParser(description="Deterministic batch ASR benchmark with WER")
    parser.add_argument("--server", default="http://localhost:8000", help="Server base URL")
    parser.add_argument(
        "--concurrency",
        default="1,8,16,32,64",
        help="Comma-separated concurrency levels (default: 1,8,16,32,64)",
    )
    parser.add_argument("--sustained-rounds", type=int, default=8, help="Sustained load rounds (default: 8)")
    parser.add_argument("--sustained-concurrency", type=int, default=64, help="Sustained load concurrency (default: 64)")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup requests (default: 50)")
    parser.add_argument("--output", default="/tmp/bench_batch_report.json", help="Output JSON path")
    parser.add_argument("--skip-wer", action="store_true", help="Skip WER computation")
    args = parser.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]
    url = f"{args.server}/v1/transcribe"

    log.info("=== Deterministic Batch ASR Benchmark ===")
    log.info(f"Server: {args.server}")
    log.info(f"Concurrency levels: {levels}")

    # Step 1: Ensure LibriSpeech
    ensure_librispeech()
    refs = load_references()

    wav_files = sorted(WAV_DIR.glob("*.wav"))[:MAX_SAMPLES]
    log.info(f"Using {len(wav_files)} WAV files, {len(refs)} references")

    report = {
        "benchmark": "NeMo ASR Batch Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": args.server,
        "samples": len(wav_files),
        "dataset": "LibriSpeech test-clean",
    }

    # Step 2: Warmup
    log.info(f"Warmup: {args.warmup} requests at c=16...")
    await run_sweep(url, wav_files, concurrency=16, repeat=1)
    log.info("Warmup complete")

    # Step 3: WER evaluation (c=1 for deterministic ordering)
    if not args.skip_wer:
        log.info("WER evaluation: c=1, 200 samples...")
        wer_results, _ = await run_sweep(url, wav_files, concurrency=1)

        ok_results = [r for r in wer_results if r["status"] == "ok"]
        ref_texts = []
        hyp_texts = []
        for r in ok_results:
            utt_id = r["utt_id"]
            if utt_id in refs:
                ref_texts.append(refs[utt_id])
                hyp_texts.append(r["text"])

        if ref_texts:
            wer_val, per_utt = compute_wer(ref_texts, hyp_texts)
            report["wer"] = {
                "corpus_wer_pct": round(wer_val * 100, 2),
                "samples_evaluated": len(ref_texts),
                "normalization": "whisper_english",
            }
            high_wer = [(ref_texts[i], hyp_texts[i], per_utt[i]) for i in range(len(per_utt)) if per_utt[i] > 0.1]
            report["wer"]["high_wer_count"] = len(high_wer)
            log.info(f"WER: {wer_val*100:.2f}% ({len(ref_texts)} samples, {len(high_wer)} with >10% WER)")
        else:
            log.warning("No matching references found for WER")
    else:
        log.info("WER evaluation skipped (--skip-wer)")

    # Step 4: Concurrency sweep (warm model)
    log.info("Concurrency sweep (warm model)...")
    sweep_results = []
    for c in levels:
        log.info(f"  c={c}...")
        results, wall = await run_sweep(url, wav_files, concurrency=c)
        summary = summarize_sweep(results, wall, c)
        sweep_results.append(summary)
        log.info(f"    {summary['rps']} RPS, RTFx={summary['rtfx']}x, p50={summary.get('p50_s', '?')}s, failures={summary['failures']}")

    report["concurrency_sweep"] = sweep_results

    # Step 5: Sustained load
    sustained_c = args.sustained_concurrency
    rounds = args.sustained_rounds
    log.info(f"Sustained load: c={sustained_c}, {rounds} rounds x {len(wav_files)} files...")
    sustained_results, sustained_wall = await run_sweep(
        url, wav_files, concurrency=sustained_c, repeat=rounds
    )
    sustained_summary = summarize_sweep(sustained_results, sustained_wall, sustained_c)
    sustained_summary["rounds"] = rounds
    sustained_summary["total_files"] = len(wav_files) * rounds
    report["sustained_load"] = sustained_summary
    log.info(
        f"  Sustained: {sustained_summary['rps']} RPS, "
        f"{sustained_summary['total_files']} files, "
        f"{sustained_summary['failures']} failures"
    )

    # Step 6: Summary
    peak = max(sweep_results, key=lambda x: x["rps"])
    max_conc_zero_fail = max(
        (s for s in sweep_results if s["failures"] == 0),
        key=lambda x: x["concurrency"],
        default=peak,
    )
    report["summary"] = {
        "peak_rps": peak["rps"],
        "peak_rtfx": peak["rtfx"],
        "peak_concurrency": peak["concurrency"],
        "max_concurrency_zero_fail": max_conc_zero_fail["concurrency"],
        "sustained_rps": sustained_summary["rps"],
        "sustained_rtfx": sustained_summary["rtfx"],
        "total_failures": sum(s["failures"] for s in sweep_results) + sustained_summary["failures"],
        "wer_pct": report.get("wer", {}).get("corpus_wer_pct"),
    }

    # Print markdown report
    print()
    print("## Batch Benchmark Results")
    print()
    if "wer" in report:
        print(f"**WER:** {report['wer']['corpus_wer_pct']}% "
              f"({report['wer']['samples_evaluated']} samples, "
              f"{report['wer']['normalization']} normalization)")
        print()
    print("### Concurrency Sweep")
    print("| c | RPS | RTFx | RTF | sess/min | p50 | p99 | Failures |")
    print("|---|-----|------|-----|----------|-----|-----|----------|")
    for s in sweep_results:
        print(f"| {s['concurrency']} | {s['rps']} | {s['rtfx']}x | {s['rtf']} "
              f"| {s['sess_per_min']} | {s.get('p50_s', '?')}s | {s.get('p99_s', '?')}s | {s['failures']} |")
    print()
    print(f"**Peak:** {peak['rps']} RPS / {peak['rtfx']}x RTFx at c={peak['concurrency']} "
          f"| **Max concurrency (0 failures):** c={max_conc_zero_fail['concurrency']}")
    print()
    print("### Sustained Load")
    print(f"| Metric | Value |")
    print(f"|--------|-------|")
    print(f"| Concurrency | {sustained_summary['concurrency']} |")
    print(f"| Total files | {sustained_summary['total_files']} |")
    print(f"| RPS | {sustained_summary['rps']} |")
    print(f"| RTFx | {sustained_summary['rtfx']}x |")
    print(f"| sess/min | {sustained_summary['sess_per_min']} |")
    print(f"| p50 / p99 | {sustained_summary.get('p50_s', '?')}s / {sustained_summary.get('p99_s', '?')}s |")
    print(f"| Failures | {sustained_summary['failures']} |")

    # Save report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
