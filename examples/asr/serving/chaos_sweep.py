#!/usr/bin/env python3
"""
Chaos Parameter Sweep — find optimal server config on current hardware.

Tests every combination of tunable parameters via the /admin/config API,
measures throughput and latency for each, and reports the optimal config
with minimum resource footprint.

Usage:
    python3 chaos_sweep.py --server http://localhost:8000
"""

import argparse
import asyncio
import io
import json
import logging
import os
import struct
import subprocess
import tempfile
import time

import aiohttp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("chaos-sweep")

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "asr-test-audio")


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


def prepare_audio():
    os.makedirs(_CACHE_DIR, exist_ok=True)
    files = []
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming how we interact with technology.",
        "Please remember to save your work before closing the application.",
    ]
    for i, s in enumerate(sentences):
        path = os.path.join(_CACHE_DIR, f"bench_tts_{i}.wav")
        if not os.path.exists(path):
            try:
                subprocess.run(
                    ["espeak-ng", "-w", path, "--stdin", "-s", "150", "-a", "100"],
                    input=s, text=True, capture_output=True, check=True, timeout=30,
                )
            except Exception:
                continue
        if os.path.exists(path):
            with open(path, "rb") as f:
                files.append(f.read())
    return files


async def set_config(session, server, **params):
    """Set server config via admin API."""
    query = "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
    async with session.post(f"{server}/admin/config?{query}") as resp:
        if resp.status == 200:
            return await resp.json()
        return None


async def measure_throughput(session, server, audio_files, concurrency, n_requests, label=""):
    """Quick throughput measurement at given concurrency."""
    sem = asyncio.Semaphore(concurrency)
    ok = 0
    fail = 0
    latencies = []

    async def one(idx):
        nonlocal ok, fail
        data = audio_files[idx % len(audio_files)]
        async with sem:
            t0 = time.monotonic()
            try:
                form = aiohttp.FormData()
                form.add_field("file", data, filename=f"sweep_{idx}.wav", content_type="audio/wav")
                async with session.post(f"{server}/v1/transcribe", data=form) as resp:
                    await resp.json()
                    if resp.status == 200:
                        ok += 1
                        latencies.append(time.monotonic() - t0)
                    else:
                        fail += 1
            except Exception:
                fail += 1

    t_wall = time.monotonic()
    tasks = [one(i) for i in range(n_requests)]
    await asyncio.gather(*tasks)
    wall = time.monotonic() - t_wall

    latencies.sort()
    return {
        "label": label,
        "concurrency": concurrency,
        "requests": n_requests,
        "successes": ok,
        "failures": fail,
        "wall_s": round(wall, 2),
        "rps": round(ok / wall, 2) if wall > 0 else 0,
        "p50_ms": round(latencies[len(latencies) // 2] * 1000, 1) if latencies else 0,
        "p95_ms": round(latencies[int(len(latencies) * 0.95)] * 1000, 1) if latencies else 0,
        "p99_ms": round(latencies[int(len(latencies) * 0.99)] * 1000, 1) if latencies else 0,
    }


async def drain_server(session, server):
    """Wait for server queue to drain."""
    await asyncio.sleep(1.0)
    for _ in range(10):
        async with session.get(f"{server}/metrics") as resp:
            m = await resp.json()
            pending = m.get("batch", {}).get("pending_requests", 0)
            if pending == 0:
                return
        await asyncio.sleep(0.5)


async def main():
    parser = argparse.ArgumentParser(description="Chaos parameter sweep")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=200, help="Requests per test")
    parser.add_argument("--concurrency", type=int, default=32, help="Test concurrency")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    server = args.server
    n_req = args.requests
    conc = args.concurrency

    audio_files = prepare_audio()
    if not audio_files:
        log.error("No audio samples")
        return
    log.info(f"Loaded {len(audio_files)} audio samples")

    report = {
        "benchmark": "Chaos Parameter Sweep — Optimize Current Hardware",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": server,
        "test_concurrency": conc,
        "test_requests": n_req,
        "sweeps": {},
    }

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        # Verify admin API
        async with session.get(f"{server}/admin/config") as resp:
            if resp.status != 200:
                log.error("Admin API not available — deploy the latest server code first")
                return
            baseline_config = await resp.json()
            log.info(f"Baseline config: {baseline_config}")
            report["baseline_config"] = baseline_config

        # Warmup
        log.info("Warmup...")
        await measure_throughput(session, server, audio_files, 16, 50)
        await drain_server(session, server)

        # === SWEEP 1: max_wait_seconds ===
        log.info("=== Sweep 1: max_wait_seconds ===")
        sweep1 = []
        for wait in [0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002]:
            await set_config(session, server, max_wait_seconds=wait)
            await drain_server(session, server)
            r = await measure_throughput(session, server, audio_files, conc, n_req, f"wait={wait}s")
            sweep1.append({**r, "max_wait_seconds": wait})
            log.info(f"  wait={wait}s → {r['rps']} RPS, p50={r['p50_ms']}ms, fail={r['failures']}")
        report["sweeps"]["max_wait_seconds"] = sweep1

        # Reset to best wait
        best_wait = max(sweep1, key=lambda x: x["rps"])
        await set_config(session, server, max_wait_seconds=best_wait["max_wait_seconds"])
        log.info(f"  Best: wait={best_wait['max_wait_seconds']}s → {best_wait['rps']} RPS")

        # === SWEEP 2: max_batch_size ===
        log.info("=== Sweep 2: max_batch_size ===")
        sweep2 = []
        for bs in [4, 8, 16, 32, 48, 64, 96, 128]:
            await set_config(session, server, max_batch_size=bs)
            await drain_server(session, server)
            r = await measure_throughput(session, server, audio_files, conc, n_req, f"batch={bs}")
            sweep2.append({**r, "max_batch_size": bs})
            log.info(f"  batch_size={bs} → {r['rps']} RPS, p50={r['p50_ms']}ms, fail={r['failures']}")
        report["sweeps"]["max_batch_size"] = sweep2

        # Reset to best batch size
        best_bs = max(sweep2, key=lambda x: x["rps"])
        await set_config(session, server, max_batch_size=best_bs["max_batch_size"])
        log.info(f"  Best: batch_size={best_bs['max_batch_size']} → {best_bs['rps']} RPS")

        # === SWEEP 3: gpu_poll_timeout ===
        log.info("=== Sweep 3: gpu_poll_timeout ===")
        sweep3 = []
        for poll in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
            await set_config(session, server, gpu_poll_timeout=poll)
            await drain_server(session, server)
            r = await measure_throughput(session, server, audio_files, conc, n_req, f"poll={poll}s")
            sweep3.append({**r, "gpu_poll_timeout": poll})
            log.info(f"  poll={poll}s → {r['rps']} RPS, p50={r['p50_ms']}ms, fail={r['failures']}")
        report["sweeps"]["gpu_poll_timeout"] = sweep3

        best_poll = max(sweep3, key=lambda x: x["rps"])
        await set_config(session, server, gpu_poll_timeout=best_poll["gpu_poll_timeout"])
        log.info(f"  Best: poll={best_poll['gpu_poll_timeout']}s → {best_poll['rps']} RPS")

        # === SWEEP 4: max_queue_depth ===
        log.info("=== Sweep 4: max_queue_depth ===")
        sweep4 = []
        for qd in [64, 128, 256, 512, 1024, 2048]:
            await set_config(session, server, max_queue_depth=qd)
            await drain_server(session, server)
            r = await measure_throughput(session, server, audio_files, conc, n_req, f"queue={qd}")
            sweep4.append({**r, "max_queue_depth": qd})
            log.info(f"  queue_depth={qd} → {r['rps']} RPS, p50={r['p50_ms']}ms, fail={r['failures']}")
        report["sweeps"]["max_queue_depth"] = sweep4

        best_qd = max(sweep4, key=lambda x: x["rps"])
        log.info(f"  Best: queue_depth={best_qd['max_queue_depth']} → {best_qd['rps']} RPS")

        # === SWEEP 5: Combined optimal — apply all bests ===
        log.info("=== Sweep 5: Combined optimal ===")
        optimal_config = {
            "max_wait_seconds": best_wait["max_wait_seconds"],
            "max_batch_size": best_bs["max_batch_size"],
            "gpu_poll_timeout": best_poll["gpu_poll_timeout"],
            "max_queue_depth": best_qd["max_queue_depth"],
        }
        await set_config(session, server, **optimal_config)
        await drain_server(session, server)

        # Test at multiple concurrency levels with optimal config
        combined = []
        for c in [8, 16, 32, 64, 128]:
            r = await measure_throughput(session, server, audio_files, c, n_req, f"optimal@conc={c}")
            combined.append({**r, **optimal_config})
            log.info(f"  conc={c} → {r['rps']} RPS, p50={r['p50_ms']}ms, fail={r['failures']}")
            await drain_server(session, server)
        report["sweeps"]["combined_optimal"] = combined

        # === SWEEP 6: Concurrency sweet spot with optimal config ===
        log.info("=== Sweep 6: Sustained load at optimal ===")
        sustained_results = []
        deadline = time.monotonic() + 60  # 1 minute sustained
        sem = asyncio.Semaphore(conc)
        ok_count = 0
        fail_count = 0
        all_latencies = []
        idx = 0

        active = []
        while time.monotonic() < deadline or active:
            while len(active) < conc * 2 and time.monotonic() < deadline:
                data = audio_files[idx % len(audio_files)]
                t_sent = time.monotonic()

                async def do_one(d, ts, s=sem):
                    nonlocal ok_count, fail_count
                    async with s:
                        try:
                            form = aiohttp.FormData()
                            form.add_field("file", d, filename="sust.wav", content_type="audio/wav")
                            async with session.post(f"{server}/v1/transcribe", data=form) as resp:
                                await resp.json()
                                lat = time.monotonic() - ts
                                if resp.status == 200:
                                    ok_count += 1
                                    all_latencies.append(lat)
                                else:
                                    fail_count += 1
                        except Exception:
                            fail_count += 1

                task = asyncio.create_task(do_one(data, t_sent))
                active.append(task)
                idx += 1

            if active:
                done, pending = await asyncio.wait(active, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
                active = list(pending)
                if time.monotonic() >= deadline:
                    for t in active:
                        await t
                    active = []

        all_latencies.sort()
        report["sustained_optimal"] = {
            "duration_seconds": 60,
            "config": optimal_config,
            "total_requests": ok_count + fail_count,
            "successes": ok_count,
            "failures": fail_count,
            "sustained_rps": round(ok_count / 60, 2),
            "p50_ms": round(all_latencies[len(all_latencies) // 2] * 1000, 1) if all_latencies else 0,
            "p95_ms": round(all_latencies[int(len(all_latencies) * 0.95)] * 1000, 1) if all_latencies else 0,
            "p99_ms": round(all_latencies[int(len(all_latencies) * 0.99)] * 1000, 1) if all_latencies else 0,
        }
        log.info(f"  Sustained: {report['sustained_optimal']['sustained_rps']} RPS, "
                 f"p50={report['sustained_optimal']['p50_ms']}ms")

        # Restore original config
        await set_config(session, server, **baseline_config)

    # === Analysis ===
    report["optimal_config"] = optimal_config
    report["improvements"] = {}

    # Find baseline (first sweep1 entry closest to original 0.1s wait)
    baseline_rps = next((s["rps"] for s in sweep1 if s["max_wait_seconds"] == 0.1), sweep1[0]["rps"])
    optimal_rps = max(r["rps"] for r in combined)
    report["improvements"]["baseline_rps"] = baseline_rps
    report["improvements"]["optimal_rps"] = optimal_rps
    report["improvements"]["improvement_pct"] = round((optimal_rps - baseline_rps) / baseline_rps * 100, 1) if baseline_rps > 0 else 0

    output = json.dumps(report, indent=2)
    print(output)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        log.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
