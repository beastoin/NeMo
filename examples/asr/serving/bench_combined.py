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
  4. WER evaluation      — quality check under combined load
  5. Duration stress     — test with long audio files (30s–300s) to find OOM threshold
  6. Soak test           — 20-min sustained load with VRAM leak detection via linear regression
  7. Chaos test          — mixed traffic ratio (default 70/30 batch/stream) with random durations

Usage:
    python3 bench_combined.py --server http://localhost:8000
    python3 bench_combined.py --server http://localhost:8000 --soak
    python3 bench_combined.py --server http://localhost:8000 --chaos --chaos-minutes 10
    python3 bench_combined.py --server http://localhost:8000 --skip-baselines
"""

import argparse
import asyncio
import json
import logging
import os
import struct
import subprocess
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


def get_vram_mb():
    """Get current GPU VRAM usage in MB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            timeout=5,
        )
        used, total = out.decode().strip().split(", ")
        return int(used), int(total)
    except Exception:
        return None, None


def create_long_wav(source_files, target_duration_s, output_path):
    """Create a WAV file of approximately target_duration_s by concatenating source files."""
    import soundfile as sf

    chunks = []
    total_samples = 0
    target_samples = int(target_duration_s * SR)

    while total_samples < target_samples:
        for src in source_files:
            audio, sr = sf.read(str(src), dtype="int16")
            if sr != SR:
                continue
            chunks.append(audio)
            total_samples += len(audio)
            if total_samples >= target_samples:
                break

    import numpy as np

    combined = np.concatenate(chunks)[:target_samples]
    data_size = len(combined) * 2
    with open(output_path, "wb") as wf:
        wf.write(b"RIFF")
        wf.write(struct.pack("<I", 36 + data_size))
        wf.write(b"WAVE")
        wf.write(b"fmt ")
        wf.write(struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16))
        wf.write(b"data")
        wf.write(struct.pack("<I", data_size))
        wf.write(combined.tobytes())

    return get_wav_duration(output_path)


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


async def run_combined(batch_url, ws_url, batch_files, stream_files, batch_c, stream_c, chunk_ms):
    """Run batch and streaming concurrently, return (batch_results, stream_results, wall_time)."""

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
    parser.add_argument("--n-stream-files", type=int, default=50, help="Max WAV files for streaming tests")
    parser.add_argument("--duration-sweep", action="store_true", default=False, help="Run duration stress test")
    parser.add_argument(
        "--duration-targets", default="30,60,120,300", help="Target durations in seconds for stress test"
    )
    parser.add_argument("--duration-batch-c", default="1,4,8", help="Batch concurrency levels for duration sweep")
    parser.add_argument(
        "--soak", action="store_true", default=False, help="Run 20-min soak test with VRAM leak detection"
    )
    parser.add_argument("--soak-minutes", type=float, default=20.0, help="Soak test duration in minutes")
    parser.add_argument("--soak-batch-c", type=int, default=16, help="Soak test batch concurrency")
    parser.add_argument("--soak-stream-c", type=int, default=8, help="Soak test stream concurrency")
    parser.add_argument(
        "--soak-slope-gate", type=float, default=50.0, help="VRAM slope gate in MiB/min (above = leak)"
    )
    parser.add_argument("--chaos", action="store_true", default=False, help="Run chaos test with mixed traffic ratio")
    parser.add_argument("--chaos-minutes", type=float, default=10.0, help="Chaos test duration in minutes")
    parser.add_argument(
        "--chaos-ratio", type=float, default=0.7, help="Batch fraction (0.7 = 70%% batch, 30%% stream)"
    )
    parser.add_argument("--chaos-max-dur", type=float, default=60.0, help="Max random audio duration for chaos test")
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
    stream_wav_files = wav_files[: args.n_stream_files]
    log.info(f"Using {len(wav_files)} WAV files (batch), {len(stream_wav_files)} (stream)")

    # Build duration map for existing files
    file_durations = {}
    for f in wav_files:
        file_durations[str(f)] = get_wav_duration(f)

    dur_stats = sorted(file_durations.values())
    log.info(
        f"File durations: min={dur_stats[0]:.1f}s, max={dur_stats[-1]:.1f}s, "
        f"mean={sum(dur_stats)/len(dur_stats):.1f}s, median={dur_stats[len(dur_stats)//2]:.1f}s"
    )

    report = {
        "benchmark": "NeMo ASR Combined Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": args.server,
        "chunk_ms": args.chunk_ms,
        "samples": len(wav_files),
        "dataset": "LibriSpeech test-clean",
        "file_duration_stats": {
            "min_s": round(dur_stats[0], 1),
            "max_s": round(dur_stats[-1], 1),
            "mean_s": round(sum(dur_stats) / len(dur_stats), 1),
            "median_s": round(dur_stats[len(dur_stats) // 2], 1),
            "buckets": {
                "0-5s": sum(1 for d in dur_stats if d <= 5),
                "5-10s": sum(1 for d in dur_stats if 5 < d <= 10),
                "10-15s": sum(1 for d in dur_stats if 10 < d <= 15),
                "15-20s": sum(1 for d in dur_stats if 15 < d <= 20),
                "20s+": sum(1 for d in dur_stats if d > 20),
            },
        },
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
            results, wall = await run_stream(ws_url, stream_wav_files, c, args.chunk_ms)
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
            batch_res, stream_res, wall = await run_combined(
                batch_url, ws_url, wav_files, stream_wav_files, bc, sc, args.chunk_ms
            )
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
        batch_res, stream_res, _ = await run_combined(
            batch_url, ws_url, wav_files, stream_wav_files, bc, sc, args.chunk_ms
        )
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
        batch_res, stream_res, _ = await run_combined(
            batch_url, ws_url, wav_files, stream_wav_files, 1, 1, args.chunk_ms
        )

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

    # ── Phase 5: Duration stress test ──
    if args.duration_sweep:
        duration_targets = [int(x) for x in args.duration_targets.split(",")]
        duration_batch_levels = [int(x) for x in args.duration_batch_c.split(",")]
        log.info(f"Phase 5: Duration stress test — {duration_targets}s files, batch c={duration_batch_levels}")

        long_dir = Path("/tmp/bench-long-wavs")
        long_dir.mkdir(exist_ok=True)

        source_files = sorted(wav_dir.glob("*.wav"))
        long_files = {}
        for dur in duration_targets:
            out_path = long_dir / f"synthetic_{dur}s.wav"
            if not out_path.exists():
                actual = create_long_wav(source_files, dur, out_path)
                log.info(f"  Created {dur}s synthetic file ({actual:.1f}s actual)")
            else:
                actual = get_wav_duration(out_path)
                log.info(f"  Reusing {dur}s synthetic file ({actual:.1f}s actual)")
            long_files[dur] = out_path

        n_copies = 4
        duration_results = []

        for dur in duration_targets:
            copies = [long_files[dur]] * n_copies
            dur_entry = {"target_duration_s": dur, "actual_duration_s": round(get_wav_duration(long_files[dur]), 1)}

            vram_before, vram_total = get_vram_mb()
            dur_entry["vram_before_mb"] = vram_before
            dur_entry["vram_total_mb"] = vram_total

            # Batch-only with long files
            batch_results_by_c = []
            for bc in duration_batch_levels:
                log.info(f"  Duration {dur}s — batch-only c={bc} ({n_copies} files)...")
                try:
                    results, wall = await asyncio.wait_for(
                        run_batch(batch_url, copies, bc), timeout=max(dur * n_copies * 2, 120)
                    )
                    summary = summarize_batch(results, wall, bc)
                    vram_after, _ = get_vram_mb()
                    summary["vram_after_mb"] = vram_after
                    batch_results_by_c.append(summary)
                    log.info(
                        f"    RPS={summary['rps']}, RTFx={summary['rtfx']}, "
                        f"fail={summary['failures']}, VRAM={vram_after}MB"
                    )
                except (asyncio.TimeoutError, Exception) as e:
                    vram_after, _ = get_vram_mb()
                    error_entry = {
                        "concurrency": bc,
                        "total": n_copies,
                        "ok": 0,
                        "failures": n_copies,
                        "error": str(e)[:200],
                        "vram_after_mb": vram_after,
                    }
                    batch_results_by_c.append(error_entry)
                    log.info(f"    FAILED: {str(e)[:100]}, VRAM={vram_after}MB")

            dur_entry["batch_only"] = batch_results_by_c

            # Combined: long batch files + streaming
            combined_by_c = []
            for bc in duration_batch_levels:
                sc = stream_levels[-1]
                log.info(f"  Duration {dur}s — combined batch c={bc} + stream c={sc}...")
                try:
                    batch_res, stream_res, wall = await asyncio.wait_for(
                        run_combined(batch_url, ws_url, copies, stream_wav_files[:10], bc, sc, args.chunk_ms),
                        timeout=max(dur * n_copies * 2, 120),
                    )
                    b_sum = summarize_batch(batch_res, wall, bc)
                    s_sum = summarize_stream(stream_res, wall, sc)
                    vram_after, _ = get_vram_mb()
                    combo = {
                        "batch_concurrency": bc,
                        "stream_concurrency": sc,
                        "batch": b_sum,
                        "stream": s_sum,
                        "vram_after_mb": vram_after,
                    }
                    combined_by_c.append(combo)
                    log.info(
                        f"    Batch: RPS={b_sum['rps']}, fail={b_sum['failures']} | "
                        f"Stream: sess/min={s_sum['sess_per_min']}, fail={s_sum['failures']} | "
                        f"VRAM={vram_after}MB"
                    )
                except (asyncio.TimeoutError, Exception) as e:
                    vram_after, _ = get_vram_mb()
                    combo = {
                        "batch_concurrency": bc,
                        "stream_concurrency": sc,
                        "error": str(e)[:200],
                        "vram_after_mb": vram_after,
                    }
                    combined_by_c.append(combo)
                    log.info(f"    FAILED: {str(e)[:100]}, VRAM={vram_after}MB")

            dur_entry["combined"] = combined_by_c
            duration_results.append(dur_entry)

        report["duration_stress"] = duration_results

        # Find OOM threshold
        max_ok_dur = 0
        for dr in duration_results:
            all_ok = all(b.get("failures", 0) == 0 for b in dr["batch_only"]) and all(
                c.get("error") is None and c.get("batch", {}).get("failures", 0) == 0 for c in dr["combined"]
            )
            if all_ok:
                max_ok_dur = dr["target_duration_s"]
        report["duration_stress_summary"] = {
            "max_zero_fail_duration_s": max_ok_dur,
            "durations_tested": duration_targets,
        }
        log.info(f"  Duration stress: max zero-fail duration = {max_ok_dur}s")

    # ── Phase 6: Soak test ──
    if args.soak:
        log.info(
            f"Phase 6: Soak test — {args.soak_minutes} min sustained, "
            f"batch c={args.soak_batch_c}, stream c={args.soak_stream_c}"
        )
        soak_target = args.soak_minutes * 60
        vram_samples = []
        soak_batch_results = []
        soak_stream_results = []
        soak_failures = 0
        soak_t0 = time.monotonic()

        vram_used, vram_total = get_vram_mb()
        if vram_used is not None:
            vram_samples.append((0.0, vram_used))

        soak_round = 0
        while time.monotonic() - soak_t0 < soak_target:
            soak_round += 1
            batch_res, stream_res, _ = await run_combined(
                batch_url, ws_url, wav_files, stream_wav_files, args.soak_batch_c, args.soak_stream_c, args.chunk_ms
            )
            soak_batch_results.extend(batch_res)
            soak_stream_results.extend(stream_res)
            b_fail = sum(1 for r in batch_res if r.get("status") != "ok")
            s_fail = sum(1 for r in stream_res if r.get("status") != "ok")
            soak_failures += b_fail + s_fail

            elapsed_min = (time.monotonic() - soak_t0) / 60
            vram_used, _ = get_vram_mb()
            if vram_used is not None:
                vram_samples.append((elapsed_min, vram_used))

            if soak_round % 5 == 0:
                log.info(
                    f"  Soak round {soak_round}: {elapsed_min:.1f} min, "
                    f"VRAM={vram_used}MB, failures={soak_failures}"
                )

        soak_wall = time.monotonic() - soak_t0

        # Linear regression on VRAM samples
        soak_slope = 0.0
        soak_r2 = 0.0
        soak_leak_detected = False
        if len(vram_samples) >= 3:
            xs = [s[0] for s in vram_samples]
            ys = [s[1] for s in vram_samples]
            n = len(xs)
            x_mean = sum(xs) / n
            y_mean = sum(ys) / n
            ss_xy = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
            ss_xx = sum((xs[i] - x_mean) ** 2 for i in range(n))
            ss_yy = sum((ys[i] - y_mean) ** 2 for i in range(n))
            soak_slope = ss_xy / ss_xx if ss_xx > 0 else 0.0
            soak_r2 = (ss_xy**2) / (ss_xx * ss_yy) if ss_xx > 0 and ss_yy > 0 else 0.0
            soak_leak_detected = soak_slope > args.soak_slope_gate and soak_r2 > 0.3

        soak_batch_summary = summarize_batch(soak_batch_results, soak_wall, args.soak_batch_c)
        soak_stream_summary = summarize_stream(soak_stream_results, soak_wall, args.soak_stream_c)

        report["soak_test"] = {
            "duration_min": round(soak_wall / 60, 1),
            "rounds": soak_round,
            "batch_concurrency": args.soak_batch_c,
            "stream_concurrency": args.soak_stream_c,
            "total_requests": len(soak_batch_results) + len(soak_stream_results),
            "total_failures": soak_failures,
            "batch": soak_batch_summary,
            "stream": soak_stream_summary,
            "vram_samples": vram_samples,
            "vram_slope_mb_per_min": round(soak_slope, 2),
            "vram_r2": round(soak_r2, 4),
            "slope_gate_mb_per_min": args.soak_slope_gate,
            "leak_detected": soak_leak_detected,
            "result": "FAIL" if soak_leak_detected or soak_failures > 0 else "PASS",
        }
        result_str = "FAIL (leak)" if soak_leak_detected else ("FAIL (failures)" if soak_failures > 0 else "PASS")
        log.info(
            f"  Soak complete: {soak_round} rounds, {soak_wall/60:.1f} min, "
            f"slope={soak_slope:.1f} MiB/min (R²={soak_r2:.3f}), "
            f"failures={soak_failures} — {result_str}"
        )

    # ── Phase 7: Chaos test ──
    if args.chaos:
        import random

        log.info(
            f"Phase 7: Chaos test — {args.chaos_minutes} min, "
            f"ratio={args.chaos_ratio:.0%} batch / {1-args.chaos_ratio:.0%} stream"
        )

        long_dir = Path("/tmp/bench-chaos-wavs")
        long_dir.mkdir(exist_ok=True)
        source_files = sorted(wav_dir.glob("*.wav"))

        chaos_durations = [2, 5, 10, 20, 30, 60]
        chaos_files = {}
        for dur in chaos_durations:
            if dur <= 15:
                matching = [f for f in source_files if abs(get_wav_duration(f) - dur) < dur * 0.5]
                if matching:
                    chaos_files[dur] = matching[0]
                    continue
            out_path = long_dir / f"chaos_{dur}s.wav"
            if not out_path.exists():
                create_long_wav(source_files, dur, out_path)
            chaos_files[dur] = out_path

        available_durations = [d for d in chaos_durations if d in chaos_files and d <= args.chaos_max_dur]
        if not available_durations:
            log.error(f"  No chaos durations available (max_dur={args.chaos_max_dur}s, need >= 2s). Skipping chaos.")
            report["chaos_test"] = {
                "duration_min": 0,
                "rounds": 0,
                "result": "SKIP",
                "reason": f"No durations <= {args.chaos_max_dur}s available",
            }
        else:
            log.info(f"  Chaos files ready: {available_durations}s variants")

            chaos_target = args.chaos_minutes * 60
            chaos_t0 = time.monotonic()
            chaos_batch_results = []
            chaos_stream_results = []
            chaos_vram_samples = []
            chaos_round = 0
            chaos_round_errors = 0

            while time.monotonic() - chaos_t0 < chaos_target:
                chaos_round += 1
                batch_count = max(1, int(20 * args.chaos_ratio))
                stream_count = max(1, 20 - batch_count)

                batch_files = [chaos_files[random.choice(available_durations)] for _ in range(batch_count)]
                stream_files_chaos = [chaos_files[random.choice(available_durations)] for _ in range(stream_count)]

                batch_c = random.choice([4, 8, 16])
                stream_c = random.choice([2, 4, 8])

                try:
                    batch_res, stream_res, _ = await asyncio.wait_for(
                        run_combined(
                            batch_url, ws_url, batch_files, stream_files_chaos, batch_c, stream_c, args.chunk_ms
                        ),
                        timeout=300,
                    )
                    chaos_batch_results.extend(batch_res)
                    chaos_stream_results.extend(stream_res)
                except (asyncio.TimeoutError, Exception) as e:
                    chaos_round_errors += 1
                    log.warning(f"  Chaos round {chaos_round} error: {str(e)[:100]}")

                vram_used, _ = get_vram_mb()
                elapsed_min = (time.monotonic() - chaos_t0) / 60
                if vram_used is not None:
                    chaos_vram_samples.append((elapsed_min, vram_used))

                if chaos_round % 10 == 0:
                    b_fail = sum(1 for r in chaos_batch_results if r.get("status") != "ok")
                    s_fail = sum(1 for r in chaos_stream_results if r.get("status") != "ok")
                    log.info(
                        f"  Chaos round {chaos_round}: {elapsed_min:.1f} min, "
                        f"VRAM={vram_used}MB, batch_fail={b_fail}, stream_fail={s_fail}, "
                        f"round_errors={chaos_round_errors}"
                    )

            chaos_wall = time.monotonic() - chaos_t0
            chaos_b_fail = sum(1 for r in chaos_batch_results if r.get("status") != "ok")
            chaos_s_fail = sum(1 for r in chaos_stream_results if r.get("status") != "ok")
            chaos_total_fail = chaos_b_fail + chaos_s_fail + chaos_round_errors

            report["chaos_test"] = {
                "duration_min": round(chaos_wall / 60, 1),
                "rounds": chaos_round,
                "batch_ratio": args.chaos_ratio,
                "duration_variants_s": available_durations,
                "total_batch_requests": len(chaos_batch_results),
                "total_stream_requests": len(chaos_stream_results),
                "batch_failures": chaos_b_fail,
                "stream_failures": chaos_s_fail,
                "round_errors": chaos_round_errors,
                "vram_samples": chaos_vram_samples,
                "result": "FAIL" if chaos_total_fail > 0 else "PASS",
            }
            log.info(
                f"  Chaos complete: {chaos_round} rounds, {chaos_wall/60:.1f} min, "
                f"batch={len(chaos_batch_results)} ({chaos_b_fail} fail), "
                f"stream={len(chaos_stream_results)} ({chaos_s_fail} fail), "
                f"round_errors={chaos_round_errors} — "
                f"{'FAIL' if chaos_total_fail > 0 else 'PASS'}"
            )

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

    if "duration_stress" in report:
        print()
        print("### Duration Stress Test (long audio files)")
        print()
        print("**Batch-only with long files**")
        print("| Duration | Concurrency | RPS | RTFx | Failures | VRAM (MB) | Result |")
        print("|----------|-------------|-----|------|----------|-----------|--------|")
        for dr in report["duration_stress"]:
            for b in dr["batch_only"]:
                fails = b.get("failures", b.get("total", "?"))
                result = "PASS" if b.get("failures", 1) == 0 else "FAIL"
                rps = b.get("rps", "-")
                rtfx = b.get("rtfx", "-")
                vram = b.get("vram_after_mb", "-")
                print(
                    f"| {dr['target_duration_s']}s | {b.get('concurrency', '?')} | "
                    f"{rps} | {rtfx} | {fails} | {vram} | {result} |"
                )
        print()
        print("**Combined (long batch + streaming)**")
        print("| Duration | Batch c | Stream c | Batch fail | Stream fail | VRAM (MB) | Result |")
        print("|----------|---------|----------|------------|-------------|-----------|--------|")
        for dr in report["duration_stress"]:
            for c in dr["combined"]:
                if "error" in c:
                    print(
                        f"| {dr['target_duration_s']}s | {c['batch_concurrency']} | "
                        f"{c['stream_concurrency']} | ERROR | ERROR | {c.get('vram_after_mb', '-')} | FAIL |"
                    )
                else:
                    bf = c["batch"]["failures"]
                    sf = c["stream"]["failures"]
                    result = "PASS" if bf == 0 and sf == 0 else "FAIL"
                    print(
                        f"| {dr['target_duration_s']}s | {c['batch_concurrency']} | "
                        f"{c['stream_concurrency']} | {bf} | {sf} | {c.get('vram_after_mb', '-')} | {result} |"
                    )
        ds = report.get("duration_stress_summary", {})
        print()
        print(f"**Max zero-fail duration:** {ds.get('max_zero_fail_duration_s', '?')}s")

    if "soak_test" in report:
        print()
        print("### Soak Test (VRAM leak detection)")
        st = report["soak_test"]
        print(f"| Metric | Value |")
        print(f"|--------|-------|")
        print(f"| Duration | {st['duration_min']} min ({st['rounds']} rounds) |")
        print(f"| Concurrency | batch c={st['batch_concurrency']}, stream c={st['stream_concurrency']} |")
        print(f"| Total requests | {st['total_requests']} |")
        print(f"| Total failures | {st['total_failures']} |")
        print(f"| VRAM slope | {st['vram_slope_mb_per_min']} MiB/min (R²={st['vram_r2']}) |")
        print(f"| Slope gate | < {st['slope_gate_mb_per_min']} MiB/min |")
        print(f"| Leak detected | {st['leak_detected']} |")
        print(f"| **Result** | **{st['result']}** |")

    if "chaos_test" in report:
        print()
        print("### Chaos Test (mixed traffic ratio)")
        ct = report["chaos_test"]
        print(f"| Metric | Value |")
        print(f"|--------|-------|")
        print(f"| Duration | {ct['duration_min']} min ({ct['rounds']} rounds) |")
        print(f"| Traffic ratio | {ct['batch_ratio']:.0%} batch / {1-ct['batch_ratio']:.0%} stream |")
        print(f"| Duration variants | {ct['duration_variants_s']}s |")
        print(f"| Batch requests | {ct['total_batch_requests']} ({ct['batch_failures']} failures) |")
        print(f"| Stream requests | {ct['total_stream_requests']} ({ct['stream_failures']} failures) |")
        print(f"| Round errors | {ct.get('round_errors', 0)} |")
        print(f"| **Result** | **{ct['result']}** |")

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
