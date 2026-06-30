#!/usr/bin/env python3
"""
Deterministic streaming ASR benchmark with WER.

Downloads LibriSpeech test-clean (200 samples), streams audio chunks via WebSocket,
computes WER using wer_utils (Whisper normalization), runs concurrency sweep and
sustained load, and outputs a structured JSON report.

Usage:
    python3 bench_stream.py --server ws://localhost:8000
    python3 bench_stream.py --server ws://localhost:8000 --chunk-ms 480
    python3 bench_stream.py --server ws://localhost:8000 --concurrency 1,4,8,16,32
"""

import argparse
import asyncio
import json
import logging
import os
import struct
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bench_stream")

SR = 16000


def ensure_librispeech():
    """Reuse bench_batch's download logic."""
    from bench_batch import ensure_librispeech as _ensure

    _ensure()


def load_references():
    """Load reference transcripts."""
    from bench_batch import load_references as _load

    return _load()


def compute_wer(references, hypotheses):
    """Compute WER using wer_utils."""
    from bench_batch import compute_wer as _compute

    return _compute(references, hypotheses)


async def stream_file(ws_url, wav_path, chunk_ms, semaphore):
    """Stream one file via WebSocket, return transcript and timing."""
    import websockets

    chunk_samples = int(SR * chunk_ms / 1000)
    chunk_bytes = chunk_samples * 2
    utt_id = Path(wav_path).stem

    async with semaphore:
        t0 = time.monotonic()
        try:
            with open(wav_path, "rb") as f:
                f.read(44)  # skip WAV header
                raw = f.read()

            async with websockets.connect(ws_url, max_size=10 * 1024 * 1024) as ws:
                config = json.dumps({"format": "pcm_s16le", "sample_rate": SR, "language": "en"})
                await ws.send(config)
                await asyncio.wait_for(ws.recv(), timeout=5)

                offset = 0
                final_parts = []
                while offset < len(raw):
                    chunk = raw[offset : offset + chunk_bytes]
                    await ws.send(chunk)
                    offset += chunk_bytes
                    await asyncio.sleep(chunk_ms / 1000.0)

                    try:
                        while True:
                            msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                            resp = json.loads(msg)
                            if resp.get("final_transcript"):
                                final_parts.append(resp["final_transcript"])
                    except asyncio.TimeoutError:
                        pass

                await ws.send(json.dumps({"action": "close"}))

                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=5)
                        resp = json.loads(msg)
                        if resp.get("final_transcript"):
                            final_parts.append(resp["final_transcript"])
                        if resp.get("final_text"):
                            final_parts = [resp["final_text"]]
                        if resp.get("done") or resp.get("status") == "closed":
                            break
                except (asyncio.TimeoutError, Exception):
                    pass

                elapsed = time.monotonic() - t0
                text = " ".join(final_parts).strip()
                audio_dur = len(raw) / (SR * 2)
                return {
                    "utt_id": utt_id,
                    "text": text,
                    "elapsed": elapsed,
                    "audio_dur": audio_dur,
                    "rtfx": audio_dur / elapsed if elapsed > 0 else 0,
                    "status": "ok",
                }
        except Exception as e:
            return {
                "utt_id": utt_id,
                "error": str(e)[:200],
                "elapsed": time.monotonic() - t0,
                "status": "error",
            }


async def run_sweep(ws_url, wav_files, concurrency, chunk_ms, repeat=1):
    """Run one concurrency level for streaming."""
    files = wav_files * repeat
    sem = asyncio.Semaphore(concurrency)
    t0 = time.monotonic()
    tasks = [stream_file(ws_url, f, chunk_ms, sem) for f in files]
    results = await asyncio.gather(*tasks)
    wall = time.monotonic() - t0
    return list(results), wall


def summarize_sweep(results, wall_time, concurrency):
    """Compute summary for one streaming concurrency level."""
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
        "sess_per_min": round(len(ok) / (wall_time / 60), 1) if wall_time > 0 else 0,
        "rtfx": round(total_audio / wall_time, 2) if wall_time > 0 else 0,
    }
    if latencies:
        summary["p50_s"] = round(latencies[len(latencies) // 2], 3)
        summary["p99_s"] = round(latencies[int(len(latencies) * 0.99)], 3)

    return summary


async def main():
    parser = argparse.ArgumentParser(description="Deterministic streaming ASR benchmark with WER")
    parser.add_argument("--server", default="ws://localhost:8000", help="Server WebSocket base URL")
    parser.add_argument("--chunk-ms", type=int, default=160, help="Chunk duration in ms (default: 160)")
    parser.add_argument(
        "--concurrency",
        default="1,4,8,16,32",
        help="Comma-separated concurrency levels (default: 1,4,8,16,32)",
    )
    parser.add_argument("--sustained-rounds", type=int, default=4, help="Sustained load rounds (default: 4)")
    parser.add_argument("--sustained-concurrency", type=int, default=32, help="Sustained load concurrency (default: 32)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup streams (default: 10)")
    parser.add_argument("--output", default="/tmp/bench_stream_report.json", help="Output JSON path")
    parser.add_argument("--skip-wer", action="store_true", help="Skip WER computation")
    args = parser.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]
    ws_url = f"{args.server}/v1/stream"

    log.info("=== Deterministic Streaming ASR Benchmark ===")
    log.info(f"Server: {args.server}")
    log.info(f"Chunk: {args.chunk_ms}ms, Concurrency levels: {levels}")

    ensure_librispeech()
    refs = load_references()

    wav_dir = Path("/tmp/librispeech-test-clean/wav")
    wav_files = sorted(wav_dir.glob("*.wav"))[:200]
    log.info(f"Using {len(wav_files)} WAV files")

    report = {
        "benchmark": "NeMo ASR Streaming Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": args.server,
        "chunk_ms": args.chunk_ms,
        "samples": len(wav_files),
        "dataset": "LibriSpeech test-clean",
    }

    # Warmup
    log.info(f"Warmup: {args.warmup} streams...")
    await run_sweep(ws_url, wav_files[: args.warmup], concurrency=4, chunk_ms=args.chunk_ms)

    # WER evaluation (c=1)
    if not args.skip_wer:
        log.info("WER evaluation: c=1...")
        wer_results, _ = await run_sweep(ws_url, wav_files, concurrency=1, chunk_ms=args.chunk_ms)
        ok_results = [r for r in wer_results if r["status"] == "ok"]

        ref_texts, hyp_texts = [], []
        for r in ok_results:
            if r["utt_id"] in refs:
                ref_texts.append(refs[r["utt_id"]])
                hyp_texts.append(r["text"])

        if ref_texts:
            wer_val, per_utt = compute_wer(ref_texts, hyp_texts)
            report["wer"] = {
                "corpus_wer_pct": round(wer_val * 100, 2),
                "samples_evaluated": len(ref_texts),
                "normalization": "whisper_english",
            }
            log.info(f"WER: {wer_val*100:.2f}%")

    # Concurrency sweep
    log.info("Concurrency sweep...")
    sweep_results = []
    for c in levels:
        log.info(f"  c={c}...")
        results, wall = await run_sweep(ws_url, wav_files, concurrency=c, chunk_ms=args.chunk_ms)
        summary = summarize_sweep(results, wall, c)
        sweep_results.append(summary)
        log.info(f"    RTFx={summary['rtfx']}, sess/min={summary['sess_per_min']}, failures={summary['failures']}")

    report["concurrency_sweep"] = sweep_results

    # Sustained load
    sc = args.sustained_concurrency
    rounds = args.sustained_rounds
    log.info(f"Sustained load: c={sc}, {rounds} rounds...")
    sustained_results, sustained_wall = await run_sweep(
        ws_url, wav_files, concurrency=sc, chunk_ms=args.chunk_ms, repeat=rounds
    )
    sustained_summary = summarize_sweep(sustained_results, sustained_wall, sc)
    sustained_summary["rounds"] = rounds
    report["sustained_load"] = sustained_summary

    # Print markdown
    print()
    print("## Streaming Benchmark Results")
    print()
    if "wer" in report:
        print(f"**WER:** {report['wer']['corpus_wer_pct']}% "
              f"({report['wer']['samples_evaluated']} samples, "
              f"{report['wer']['normalization']} normalization)")
        print()
    print("### Concurrency Sweep")
    print("| c | RTFx | sess/min | p50 | p99 | Failures |")
    print("|---|------|----------|-----|-----|----------|")
    for s in sweep_results:
        print(f"| {s['concurrency']} | {s['rtfx']}x | {s['sess_per_min']} | "
              f"{s.get('p50_s', '?')}s | {s.get('p99_s', '?')}s | {s['failures']} |")
    print()
    print("### Sustained Load")
    print(f"| Metric | Value |")
    print(f"|--------|-------|")
    print(f"| Concurrency | {sustained_summary['concurrency']} |")
    print(f"| RTFx | {sustained_summary['rtfx']}x |")
    print(f"| sess/min | {sustained_summary['sess_per_min']} |")
    print(f"| Failures | {sustained_summary['failures']} |")

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
