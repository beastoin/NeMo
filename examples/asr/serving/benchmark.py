#!/usr/bin/env python3
"""
Industrial ASR batch performance benchmark.

Runs a systematic sweep across concurrency levels and audio durations,
then produces a structured report suitable for capacity planning.

Usage:
    python3 benchmark.py --server http://localhost:8000
    python3 benchmark.py --server http://localhost:8000 --sustained-minutes 5
"""

import argparse
import asyncio
import io
import json
import logging
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import aiohttp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("benchmark")

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "asr-test-audio")


def _tts_to_wav(text, output_path):
    try:
        subprocess.run(
            ["espeak-ng", "-w", output_path, "--stdin", "-s", "150", "-a", "100"],
            input=text, text=True, capture_output=True, check=True, timeout=30,
        )
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def generate_wav(duration_seconds, sample_rate=16000):
    num_samples = int(duration_seconds * sample_rate)
    samples = np.zeros(num_samples, dtype=np.int16)
    buf = io.BytesIO()
    data_size = num_samples * 2
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(samples.tobytes())
    return buf.getvalue()


def prepare_real_audio():
    os.makedirs(_CACHE_DIR, exist_ok=True)
    files = []
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming how we interact with technology.",
        "Please remember to save your work before closing the application.",
        "Speech recognition accuracy has improved dramatically over the past decade.",
        "The conference will be held next Tuesday at three o'clock in the afternoon.",
    ]
    for i, s in enumerate(sentences):
        path = os.path.join(_CACHE_DIR, f"bench_tts_{i}.wav")
        if not os.path.exists(path):
            if not _tts_to_wav(s, path):
                continue
        with open(path, "rb") as f:
            files.append({"data": f.read(), "name": f"tts-{i}", "ref": s})

    import urllib.request
    url = "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav"
    path = os.path.join(_CACHE_DIR, "librispeech_2086.wav")
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url, path)
        except Exception:
            pass
    if os.path.exists(path):
        with open(path, "rb") as f:
            files.append({"data": f.read(), "name": "librispeech-2086", "ref": None})

    return files


async def run_batch(session, server, audio_data, filename, semaphore):
    async with semaphore:
        t0 = time.monotonic()
        try:
            form = aiohttp.FormData()
            form.add_field("file", audio_data, filename=filename, content_type="audio/wav")
            async with session.post(f"{server}/v1/transcribe", data=form) as resp:
                latency = time.monotonic() - t0
                if resp.status == 200:
                    body = await resp.json()
                    text = body.get("text", "")
                    return {"ok": True, "latency": latency, "words": len(text.split()), "empty": not text.strip()}
                else:
                    return {"ok": False, "latency": latency, "error": f"HTTP {resp.status}"}
        except Exception as exc:
            return {"ok": False, "latency": time.monotonic() - t0, "error": str(exc)[:200]}


async def concurrency_sweep(server, audio_files, levels, requests_per_level):
    results = []
    for conc in levels:
        log.info(f"  Concurrency={conc}, requests={requests_per_level}")
        semaphore = asyncio.Semaphore(conc)
        t_wall_start = time.monotonic()
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(requests_per_level):
                af = audio_files[i % len(audio_files)]
                tasks.append(run_batch(session, server, af["data"], f"{af['name']}_{i}.wav", semaphore))
            responses = await asyncio.gather(*tasks)
        wall_clock = time.monotonic() - t_wall_start

        ok = [r for r in responses if r["ok"]]
        fail = [r for r in responses if not r["ok"]]
        empty = [r for r in ok if r.get("empty")]
        latencies = sorted([r["latency"] for r in ok])

        entry = {
            "concurrency": conc,
            "total_requests": requests_per_level,
            "successes": len(ok),
            "failures": len(fail),
            "empty_transcriptions": len(empty),
            "wall_clock_s": round(wall_clock, 2),
            "throughput_rps": round(len(ok) / wall_clock, 2) if wall_clock > 0 else 0,
        }
        if latencies:
            entry["latency_min_ms"] = round(latencies[0] * 1000, 1)
            entry["latency_p50_ms"] = round(latencies[len(latencies) // 2] * 1000, 1)
            entry["latency_p90_ms"] = round(latencies[int(len(latencies) * 0.90)] * 1000, 1)
            entry["latency_p95_ms"] = round(latencies[int(len(latencies) * 0.95)] * 1000, 1)
            entry["latency_p99_ms"] = round(latencies[int(len(latencies) * 0.99)] * 1000, 1)
            entry["latency_max_ms"] = round(latencies[-1] * 1000, 1)
        if fail:
            entry["error_sample"] = [r["error"] for r in fail[:3]]

        results.append(entry)
        log.info(f"    -> {len(ok)}/{requests_per_level} ok, {entry.get('throughput_rps', 0)} RPS, p50={entry.get('latency_p50_ms', 0)}ms")

    return results


async def sustained_load(server, audio_files, concurrency, duration_minutes):
    log.info(f"  Sustained load: concurrency={concurrency}, duration={duration_minutes}min")
    semaphore = asyncio.Semaphore(concurrency)
    deadline = time.monotonic() + duration_minutes * 60
    results = []
    idx = 0
    active = []

    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline or active:
            while len(active) < concurrency * 2 and time.monotonic() < deadline:
                af = audio_files[idx % len(audio_files)]
                task = asyncio.create_task(
                    run_batch(session, server, af["data"], f"{af['name']}_{idx}.wav", semaphore)
                )
                active.append(task)
                idx += 1

            if active:
                done, pending = await asyncio.wait(active, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
                for t in done:
                    results.append(t.result())
                active = list(pending)

                if time.monotonic() >= deadline:
                    for t in active:
                        results.append(await t)
                    active = []

    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    latencies = sorted([r["latency"] for r in ok])
    elapsed = duration_minutes * 60

    buckets = []
    bucket_size = 10
    for r in results:
        bucket_idx = int(r["latency"] / bucket_size) if "latency" in r else 0
        while len(buckets) <= bucket_idx:
            buckets.append({"ok": 0, "fail": 0})
        if r["ok"]:
            buckets[bucket_idx]["ok"] += 1
        else:
            buckets[bucket_idx]["fail"] += 1

    entry = {
        "duration_minutes": duration_minutes,
        "concurrency": concurrency,
        "total_requests": len(results),
        "successes": len(ok),
        "failures": len(fail),
        "sustained_rps": round(len(ok) / elapsed, 2),
    }
    if latencies:
        entry["latency_min_ms"] = round(latencies[0] * 1000, 1)
        entry["latency_p50_ms"] = round(latencies[len(latencies) // 2] * 1000, 1)
        entry["latency_p90_ms"] = round(latencies[int(len(latencies) * 0.90)] * 1000, 1)
        entry["latency_p95_ms"] = round(latencies[int(len(latencies) * 0.95)] * 1000, 1)
        entry["latency_p99_ms"] = round(latencies[int(len(latencies) * 0.99)] * 1000, 1)
        entry["latency_max_ms"] = round(latencies[-1] * 1000, 1)

    return entry


async def duration_sweep(server, concurrency, requests_per):
    log.info("Phase: Audio duration sweep")
    results = []
    for dur in [1.0, 5.0, 15.0, 30.0, 60.0]:
        log.info(f"  Duration={dur}s, concurrency={concurrency}")
        audio_data = generate_wav(dur)
        semaphore = asyncio.Semaphore(concurrency)

        t_wall_start = time.monotonic()
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(requests_per):
                tasks.append(run_batch(session, server, audio_data, f"dur{dur}_{i}.wav", semaphore))
            responses = await asyncio.gather(*tasks)
        wall_clock = time.monotonic() - t_wall_start

        ok = [r for r in responses if r["ok"]]
        latencies = sorted([r["latency"] for r in ok])

        entry = {
            "audio_duration_s": dur,
            "total_requests": requests_per,
            "successes": len(ok),
            "failures": requests_per - len(ok),
            "wall_clock_s": round(wall_clock, 2),
            "throughput_rps": round(len(ok) / wall_clock, 2) if wall_clock > 0 else 0,
            "rtf": round((dur * len(ok)) / wall_clock, 2) if wall_clock > 0 else 0,
        }
        if latencies:
            entry["latency_p50_ms"] = round(latencies[len(latencies) // 2] * 1000, 1)
            entry["latency_p95_ms"] = round(latencies[int(len(latencies) * 0.95)] * 1000, 1)
            entry["latency_max_ms"] = round(latencies[-1] * 1000, 1)

        results.append(entry)
        log.info(f"    -> {len(ok)}/{requests_per} ok, RTF={entry['rtf']}x, p50={entry.get('latency_p50_ms', 0)}ms")

    return results


async def main():
    parser = argparse.ArgumentParser(description="Industrial ASR batch benchmark")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--sustained-minutes", type=int, default=3)
    parser.add_argument("--requests-per-level", type=int, default=200)
    parser.add_argument("--output", type=str, default=None, help="Save JSON report to file")
    args = parser.parse_args()

    server = args.server
    log.info(f"Industrial benchmark: server={server}")

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{server}/health") as resp:
            health = await resp.json()
            log.info(f"Server health: {health}")

    log.info("Preparing real audio samples...")
    audio_files = prepare_real_audio()
    if not audio_files:
        log.error("No audio samples available")
        return
    log.info(f"Loaded {len(audio_files)} audio samples")

    report = {
        "benchmark": "NeMo ASR Batch Inference — Industrial Performance Report",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": server,
        "audio_samples": len(audio_files),
        "model": "nvidia/parakeet-tdt-0.6b-v3",
        "gpu": "NVIDIA L4 (24GB)",
    }

    # Phase 1: Warmup
    log.info("Phase 1: Warmup (50 requests)")
    semaphore = asyncio.Semaphore(16)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(50):
            af = audio_files[i % len(audio_files)]
            tasks.append(run_batch(session, server, af["data"], f"warmup_{i}.wav", semaphore))
        await asyncio.gather(*tasks)
    log.info("  Warmup complete")

    # Phase 2: Concurrency sweep with real audio
    log.info("Phase 2: Concurrency sweep (real audio)")
    report["concurrency_sweep"] = await concurrency_sweep(
        server, audio_files,
        levels=[1, 2, 4, 8, 16, 32, 64, 128],
        requests_per_level=args.requests_per_level,
    )

    # Phase 3: Audio duration sweep (synthetic — tests processing time scaling)
    log.info("Phase 3: Audio duration sweep")
    report["duration_sweep"] = await duration_sweep(server, concurrency=16, requests_per=50)

    # Phase 4: Sustained load
    log.info(f"Phase 4: Sustained load ({args.sustained_minutes} min)")
    report["sustained_load"] = await sustained_load(
        server, audio_files, concurrency=32, duration_minutes=args.sustained_minutes
    )

    # Summary
    sweep = report["concurrency_sweep"]
    peak_rps = max(r["throughput_rps"] for r in sweep)
    peak_conc = next(r["concurrency"] for r in sweep if r["throughput_rps"] == peak_rps)
    total_failures = sum(r["failures"] for r in sweep)
    total_requests = sum(r["total_requests"] for r in sweep)

    report["summary"] = {
        "peak_throughput_rps": peak_rps,
        "peak_throughput_concurrency": peak_conc,
        "total_requests_all_phases": total_requests + 50 + report["sustained_load"]["total_requests"],
        "total_failures_all_phases": total_failures + report["sustained_load"]["failures"],
        "error_rate_pct": round(
            (total_failures + report["sustained_load"]["failures"])
            / (total_requests + 50 + report["sustained_load"]["total_requests"])
            * 100,
            3,
        ),
        "sustained_rps": report["sustained_load"]["sustained_rps"],
    }

    output = json.dumps(report, indent=2)
    print(output)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        log.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
