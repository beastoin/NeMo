#!/usr/bin/env python3
"""
Combined batch + streaming ASR benchmark.

Measures performance when both batch (REST) and streaming (WebSocket) inference
run simultaneously on the same GPU, sharing the single GPU worker thread.  This
is the production scenario: real workloads send a mix of file uploads and live
streams concurrently.

Phases:
  1. Isolated baselines  — batch-only and stream-only (for comparison)
  2. Combined load       — batch and streaming concurrently at several mix ratios
  3. Sustained combined  — extended combined run to check stability

Usage:
    python3 bench_combined.py --server http://localhost:8000
    python3 bench_combined.py --server http://localhost:8000 --batch-concurrency 16,32 --stream-concurrency 4,8
    python3 bench_combined.py --server http://localhost:8000 --skip-baselines
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bench_combined")

sys.path.insert(0, os.path.dirname(__file__))

from bench_batch import (
    compute_wer,
    ensure_librispeech,
    get_wav_duration,
    load_references,
)
from bench_batch import summarize_sweep as summarize_batch
from bench_batch import (
    transcribe_file,
)
from bench_stream import (
    stream_file,
)
from bench_stream import summarize_sweep as summarize_stream

SR = 16000


async def run_batch(url, wav_files, concurrency):
    """Run batch requests, return (results, wall_time)."""
    import aiohttp

    sem = asyncio.Semaphore(concurrency)
    t0 = time.monotonic()
    async with aiohttp.ClientSession() as session:
        tasks = [transcribe_file(session, url, f, sem) for f in wav_files]
        results = await asyncio.gather(*tasks)
    wall = time.monotonic() - t0
    return list(results), wall


async def run_stream(ws_url, wav_files, concurrency, chunk_ms):
    """Run streaming sessions, return (results, wall_time)."""
    sem = asyncio.Semaphore(concurrency)
    t0 = time.monotonic()
    tasks = [stream_file(ws_url, f, chunk_ms, sem) for f in wav_files]
    results = await asyncio.gather(*tasks)
    wall = time.monotonic() - t0
    return list(results), wall


async def run_combined(batch_url, ws_url, wav_files, batch_c, stream_c, chunk_ms):
    """Run batch and streaming concurrently, return (batch_results, stream_results, wall_time)."""
    half = len(wav_files) // 2
    batch_files = wav_files[:half]
    stream_files = wav_files[half:]

    t0 = time.monotonic()
    batch_task = asyncio.create_task(run_batch(batch_url, batch_files, batch_c))
    stream_task = asyncio.create_task(run_stream(ws_url, stream_files, stream_c, chunk_ms))

    batch_results, batch_wall = await batch_task
    stream_results, stream_wall = await stream_task
    wall = time.monotonic() - t0

    return batch_results, stream_results, wall


def format_pct(current, baseline):
    """Format percentage change from baseline."""
    if baseline == 0:
        return "N/A"
    change = ((current - baseline) / baseline) * 100
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:.0f}%"


async def main():
    parser = argparse.ArgumentParser(description="Combined batch + streaming ASR benchmark")
    parser.add_argument("--server", default="http://localhost:8000", help="Server base URL (http)")
    parser.add_argument("--batch-concurrency", default="8,16,32", help="Batch concurrency levels")
    parser.add_argument("--stream-concurrency", default="4,8,16", help="Stream concurrency levels")
    parser.add_argument("--chunk-ms", type=int, default=160, help="Streaming chunk duration in ms")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup requests per mode")
    parser.add_argument("--sustained-minutes", type=float, default=3.0, help="Sustained load duration")
    parser.add_argument("--sustained-batch-c", type=int, default=16, help="Sustained batch concurrency")
    parser.add_argument("--sustained-stream-c", type=int, default=8, help="Sustained stream concurrency")
    parser.add_argument("--skip-baselines", action="store_true", help="Skip isolated baseline runs")
    parser.add_argument("--skip-wer", action="store_true", help="Skip WER computation")
    parser.add_argument("--output", default="/tmp/bench_combined_report.json", help="Output JSON path")
    args = parser.parse_args()

    batch_url = f"{args.server}/v1/transcribe"
    ws_server = args.server.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_server}/v1/stream"
    batch_levels = [int(x) for x in args.batch_concurrency.split(",")]
    stream_levels = [int(x) for x in args.stream_concurrency.split(",")]

    log.info("=== Combined Batch + Streaming ASR Benchmark ===")
    log.info(f"Server: {args.server}")
    log.info(f"Batch concurrency: {batch_levels}, Stream concurrency: {stream_levels}")
    log.info(f"Chunk: {args.chunk_ms}ms")

    ensure_librispeech()
    refs = load_references()
    wav_dir = Path("/tmp/librispeech-test-clean/wav")
    wav_files = sorted(wav_dir.glob("*.wav"))[:200]
    log.info(f"Using {len(wav_files)} WAV files")

    report = {
        "benchmark": "NeMo ASR Combined Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": args.server,
        "chunk_ms": args.chunk_ms,
        "samples": len(wav_files),
        "dataset": "LibriSpeech test-clean",
    }

    # ── Warmup ──
    log.info(f"Warmup: {args.warmup} batch + {args.warmup} stream requests...")
    await run_batch(batch_url, wav_files[: args.warmup], concurrency=8)
    await run_stream(ws_url, wav_files[: args.warmup], concurrency=4, chunk_ms=args.chunk_ms)
    log.info("Warmup complete")

    # ── Phase 1: Isolated baselines ──
    baselines = {"batch": {}, "stream": {}}
    if not args.skip_baselines:
        log.info("Phase 1: Isolated baselines")

        for c in batch_levels:
            log.info(f"  Batch-only c={c}...")
            results, wall = await run_batch(batch_url, wav_files, c)
            baselines["batch"][c] = summarize_batch(results, wall, c)
            s = baselines["batch"][c]
            log.info(f"    RPS={s['rps']}, RTFx={s['rtfx']}, failures={s['failures']}")

        for c in stream_levels:
            log.info(f"  Stream-only c={c}...")
            results, wall = await run_stream(ws_url, wav_files, c, args.chunk_ms)
            baselines["stream"][c] = summarize_stream(results, wall, c)
            s = baselines["stream"][c]
            log.info(f"    RTFx={s['rtfx']}, sess/min={s['sess_per_min']}, failures={s['failures']}")

        report["baselines"] = {
            "batch": list(baselines["batch"].values()),
            "stream": list(baselines["stream"].values()),
        }
    else:
        log.info("Phase 1: Skipped (--skip-baselines)")

    # ── Phase 2: Combined load — all mix combinations ──
    log.info("Phase 2: Combined load (batch + stream simultaneously)")
    combined_results = []
    for bc in batch_levels:
        for sc in stream_levels:
            log.info(f"  Combined: batch c={bc} + stream c={sc}...")
            batch_res, stream_res, wall = await run_combined(batch_url, ws_url, wav_files, bc, sc, args.chunk_ms)
            batch_summary = summarize_batch(batch_res, wall, bc)
            stream_summary = summarize_stream(stream_res, wall, sc)

            entry = {
                "batch_concurrency": bc,
                "stream_concurrency": sc,
                "wall_s": round(wall, 2),
                "batch": batch_summary,
                "stream": stream_summary,
            }

            if not args.skip_baselines and bc in baselines["batch"] and sc in baselines["stream"]:
                bb = baselines["batch"][bc]
                sb = baselines["stream"][sc]
                entry["batch_rps_vs_baseline"] = format_pct(batch_summary["rps"], bb["rps"])
                entry["batch_rtfx_vs_baseline"] = format_pct(batch_summary["rtfx"], bb["rtfx"])
                entry["stream_rtfx_vs_baseline"] = format_pct(stream_summary["rtfx"], sb["rtfx"])
                entry["stream_sessmin_vs_baseline"] = format_pct(stream_summary["sess_per_min"], sb["sess_per_min"])

            combined_results.append(entry)
            log.info(
                f"    Batch: RPS={batch_summary['rps']}, RTFx={batch_summary['rtfx']}, fail={batch_summary['failures']} | "
                f"Stream: RTFx={stream_summary['rtfx']}, sess/min={stream_summary['sess_per_min']}, fail={stream_summary['failures']}"
            )

    report["combined_sweep"] = combined_results

    # ── Phase 3: Sustained combined load ──
    bc = args.sustained_batch_c
    sc = args.sustained_stream_c
    target_seconds = args.sustained_minutes * 60
    files_per_round = len(wav_files)
    avg_dur = sum(get_wav_duration(f) for f in wav_files[:20]) / 20
    estimated_round_time = (files_per_round / 2 * avg_dur) / max(bc, sc)
    rounds = max(2, int(target_seconds / max(estimated_round_time, 1)))

    log.info(
        f"Phase 3: Sustained combined — batch c={bc} + stream c={sc}, ~{rounds} rounds ({args.sustained_minutes} min target)"
    )

    all_batch_results = []
    all_stream_results = []
    t0 = time.monotonic()
    for r in range(rounds):
        batch_res, stream_res, _ = await run_combined(batch_url, ws_url, wav_files, bc, sc, args.chunk_ms)
        all_batch_results.extend(batch_res)
        all_stream_results.extend(stream_res)
        elapsed = time.monotonic() - t0
        log.info(f"  Round {r+1}/{rounds} done ({elapsed:.0f}s elapsed)")
        if elapsed >= target_seconds:
            log.info(f"  Target duration reached ({args.sustained_minutes} min)")
            break

    sustained_wall = time.monotonic() - t0
    sustained_batch = summarize_batch(all_batch_results, sustained_wall, bc)
    sustained_stream = summarize_stream(all_stream_results, sustained_wall, sc)

    report["sustained_combined"] = {
        "batch_concurrency": bc,
        "stream_concurrency": sc,
        "rounds": r + 1,
        "wall_s": round(sustained_wall, 2),
        "wall_min": round(sustained_wall / 60, 1),
        "batch": sustained_batch,
        "stream": sustained_stream,
        "total_requests": len(all_batch_results) + len(all_stream_results),
        "total_failures": sustained_batch["failures"] + sustained_stream["failures"],
    }
    log.info(
        f"  Sustained: {sustained_batch['rps']} batch RPS + {sustained_stream['sess_per_min']} stream sess/min, "
        f"{sustained_batch['failures']+sustained_stream['failures']} total failures in {sustained_wall/60:.1f} min"
    )

    # ── Phase 4: WER (combined mode, c=1+1) ──
    if not args.skip_wer:
        log.info("Phase 4: WER evaluation (combined c=1+1)...")
        batch_res, stream_res, _ = await run_combined(batch_url, ws_url, wav_files, 1, 1, args.chunk_ms)

        wer_data = {}
        for label, results in [("batch", batch_res), ("stream", stream_res)]:
            ok = [r for r in results if r["status"] == "ok"]
            ref_texts, hyp_texts = [], []
            for r in ok:
                if r["utt_id"] in refs:
                    ref_texts.append(refs[r["utt_id"]])
                    hyp_texts.append(r["text"])
            if ref_texts:
                wer_val, _ = compute_wer(ref_texts, hyp_texts)
                wer_data[label] = {
                    "corpus_wer_pct": round(wer_val * 100, 2),
                    "samples_evaluated": len(ref_texts),
                    "normalization": "whisper_english",
                }
                log.info(f"  {label} WER: {wer_val*100:.2f}% ({len(ref_texts)} samples)")

        report["wer"] = wer_data

    # ── Summary ──
    best_combined = max(combined_results, key=lambda x: x["batch"]["rps"] + x["stream"]["sess_per_min"])
    zero_fail = [c for c in combined_results if c["batch"]["failures"] == 0 and c["stream"]["failures"] == 0]
    max_zero_fail = max(
        zero_fail,
        key=lambda x: x["batch_concurrency"] + x["stream_concurrency"],
        default=best_combined,
    )

    report["summary"] = {
        "best_combined_batch_c": best_combined["batch_concurrency"],
        "best_combined_stream_c": best_combined["stream_concurrency"],
        "best_combined_batch_rps": best_combined["batch"]["rps"],
        "best_combined_stream_sessmin": best_combined["stream"]["sess_per_min"],
        "max_zero_fail_batch_c": max_zero_fail["batch_concurrency"],
        "max_zero_fail_stream_c": max_zero_fail["stream_concurrency"],
        "sustained_total_requests": report["sustained_combined"]["total_requests"],
        "sustained_total_failures": report["sustained_combined"]["total_failures"],
        "sustained_minutes": report["sustained_combined"]["wall_min"],
    }
    if "wer" in report:
        report["summary"]["batch_wer_pct"] = report["wer"].get("batch", {}).get("corpus_wer_pct")
        report["summary"]["stream_wer_pct"] = report["wer"].get("stream", {}).get("corpus_wer_pct")

    # ── Print markdown ──
    print()
    print("## Combined Batch + Streaming Benchmark Results")
    print()

    if "wer" in report:
        for label in ["batch", "stream"]:
            w = report["wer"].get(label)
            if w:
                print(
                    f"**{label.title()} WER:** {w['corpus_wer_pct']}% "
                    f"({w['samples_evaluated']} samples, {w['normalization']} normalization)"
                )
        print()

    if not args.skip_baselines:
        print("### Isolated Baselines")
        print()
        print("**Batch (isolated)**")
        print("| c | RPS | RTFx | p50 | p99 | Failures |")
        print("|---|-----|------|-----|-----|----------|")
        for s in report["baselines"]["batch"]:
            print(
                f"| {s['concurrency']} | {s['rps']} | {s['rtfx']}x | "
                f"{s.get('p50_s', '?')}s | {s.get('p99_s', '?')}s | {s['failures']} |"
            )
        print()
        print("**Stream (isolated)**")
        print("| c | RTFx | sess/min | p50 | p99 | Failures |")
        print("|---|------|----------|-----|-----|----------|")
        for s in report["baselines"]["stream"]:
            print(
                f"| {s['concurrency']} | {s['rtfx']}x | {s['sess_per_min']} | "
                f"{s.get('p50_s', '?')}s | {s.get('p99_s', '?')}s | {s['failures']} |"
            )
        print()

    print("### Combined Load (batch + stream simultaneously)")
    if not args.skip_baselines:
        print(
            "| Batch c | Stream c | Batch RPS | vs baseline | Stream sess/min | vs baseline | Batch fail | Stream fail |"
        )
        print(
            "|---------|----------|-----------|-------------|-----------------|-------------|------------|-------------|"
        )
        for c in combined_results:
            print(
                f"| {c['batch_concurrency']} | {c['stream_concurrency']} | "
                f"{c['batch']['rps']} | {c.get('batch_rps_vs_baseline', 'N/A')} | "
                f"{c['stream']['sess_per_min']} | {c.get('stream_sessmin_vs_baseline', 'N/A')} | "
                f"{c['batch']['failures']} | {c['stream']['failures']} |"
            )
    else:
        print(
            "| Batch c | Stream c | Batch RPS | Batch RTFx | Stream RTFx | Stream sess/min | Batch fail | Stream fail |"
        )
        print(
            "|---------|----------|-----------|------------|-------------|-----------------|------------|-------------|"
        )
        for c in combined_results:
            print(
                f"| {c['batch_concurrency']} | {c['stream_concurrency']} | "
                f"{c['batch']['rps']} | {c['batch']['rtfx']}x | "
                f"{c['stream']['rtfx']}x | {c['stream']['sess_per_min']} | "
                f"{c['batch']['failures']} | {c['stream']['failures']} |"
            )

    print()
    print("### Sustained Combined Load")
    sc = report["sustained_combined"]
    print(f"| Metric | Batch | Stream |")
    print(f"|--------|-------|--------|")
    print(f"| Concurrency | {sc['batch_concurrency']} | {sc['stream_concurrency']} |")
    print(f"| Requests | {sc['batch']['ok']}/{sc['batch']['total']} | {sc['stream']['ok']}/{sc['stream']['total']} |")
    print(f"| RPS / sess/min | {sc['batch']['rps']} | {sc['stream']['sess_per_min']} |")
    print(f"| RTFx | {sc['batch']['rtfx']}x | {sc['stream']['rtfx']}x |")
    print(
        f"| p50 / p99 | {sc['batch'].get('p50_s', '?')}s / {sc['batch'].get('p99_s', '?')}s | "
        f"{sc['stream'].get('p50_s', '?')}s / {sc['stream'].get('p99_s', '?')}s |"
    )
    print(f"| Failures | {sc['batch']['failures']} | {sc['stream']['failures']} |")
    print(f"| Duration | {sc['wall_min']} min | {sc['rounds']} rounds |")

    print()
    print(
        f"**Best combined (0 failures):** batch c={max_zero_fail['batch_concurrency']} + "
        f"stream c={max_zero_fail['stream_concurrency']}"
    )
    print(
        f"**Sustained:** {sc['total_requests']} total requests, "
        f"{sc['total_failures']} failures in {sc['wall_min']} min"
    )

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
