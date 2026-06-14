#!/usr/bin/env python3
"""
Level 2 Industrial Benchmark — Chaos & Bottleneck Analysis.

Systematically isolates the throughput bottleneck by testing:
1. Serial baseline (pure GPU compute time)
2. Burst patterns (batch fill efficiency)
3. Request size sweep (I/O vs compute ratio)
4. Sustained ramp (find exact saturation knee)
5. Back-to-back timing (queue wait decomposition)
6. Parallel upload test (network/CPU isolation)
7. Theoretical vs actual throughput gap analysis

Usage:
    python3 chaos_benchmark.py --server http://localhost:8000
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
log = logging.getLogger("chaos")

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


def prepare_real_audio():
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


async def timed_request(session, server, audio_data, filename):
    """Single request with detailed timing breakdown."""
    t_start = time.monotonic()
    form = aiohttp.FormData()
    form.add_field("file", audio_data, filename=filename, content_type="audio/wav")
    t_form = time.monotonic()

    async with session.post(f"{server}/v1/transcribe", data=form) as resp:
        t_headers = time.monotonic()
        body = await resp.json()
        t_done = time.monotonic()

    return {
        "ok": resp.status == 200,
        "total": t_done - t_start,
        "form_build": t_form - t_start,
        "server_time": t_headers - t_form,
        "body_read": t_done - t_headers,
        "text": body.get("text", ""),
    }


async def test_serial_baseline(server, audio_files, n=50):
    """Pure serial requests — measures true single-request GPU time."""
    log.info(f"Test 1: Serial baseline ({n} sequential requests)")
    results = []
    async with aiohttp.ClientSession() as session:
        for i in range(n):
            data = audio_files[i % len(audio_files)]
            r = await timed_request(session, server, data, f"serial_{i}.wav")
            results.append(r)

    server_times = [r["server_time"] for r in results if r["ok"]]
    return {
        "test": "serial_baseline",
        "requests": n,
        "successes": sum(1 for r in results if r["ok"]),
        "server_time_min_ms": round(min(server_times) * 1000, 1),
        "server_time_p50_ms": round(sorted(server_times)[len(server_times) // 2] * 1000, 1),
        "server_time_p95_ms": round(sorted(server_times)[int(len(server_times) * 0.95)] * 1000, 1),
        "server_time_max_ms": round(max(server_times) * 1000, 1),
        "serial_rps": round(n / sum(server_times), 2),
        "avg_gpu_time_ms": round(sum(server_times) / len(server_times) * 1000, 1),
    }


async def test_burst_patterns(server, audio_files):
    """Send exact burst sizes to measure batch fill efficiency."""
    log.info("Test 2: Burst pattern analysis")
    results = []

    for burst_size in [1, 2, 4, 8, 16, 32, 48, 64]:
        log.info(f"  Burst size={burst_size}")
        latencies = []
        for trial in range(3):
            async with aiohttp.ClientSession() as session:
                t0 = time.monotonic()
                tasks = []
                for i in range(burst_size):
                    data = audio_files[i % len(audio_files)]
                    form = aiohttp.FormData()
                    form.add_field("file", data, filename=f"burst_{burst_size}_{i}.wav", content_type="audio/wav")
                    tasks.append(session.post(f"{server}/v1/transcribe", data=form))

                responses = await asyncio.gather(*tasks)
                bodies = [await r.json() for r in responses]
                wall = time.monotonic() - t0
                for r in responses:
                    r.close()
                latencies.append(wall)

            await asyncio.sleep(0.5)  # let GPU drain between trials

        avg_wall = sum(latencies) / len(latencies)
        results.append({
            "burst_size": burst_size,
            "avg_wall_ms": round(avg_wall * 1000, 1),
            "effective_rps": round(burst_size / avg_wall, 2),
            "per_request_ms": round(avg_wall / burst_size * 1000, 1),
            "trials": len(latencies),
        })

    return {"test": "burst_patterns", "results": results}


async def test_saturation_ramp(server, audio_files, duration_per_level=10):
    """Gradually increase load to find exact saturation point."""
    log.info("Test 3: Saturation ramp (find exact knee)")
    results = []

    for target_rps in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80]:
        log.info(f"  Target RPS={target_rps}")
        interval = 1.0 / target_rps
        deadline = time.monotonic() + duration_per_level
        completed = []
        pending = []

        async with aiohttp.ClientSession() as session:
            next_send = time.monotonic()
            idx = 0

            while time.monotonic() < deadline:
                now = time.monotonic()
                while next_send <= now and time.monotonic() < deadline:
                    data = audio_files[idx % len(audio_files)]
                    form = aiohttp.FormData()
                    form.add_field("file", data, filename=f"ramp_{target_rps}_{idx}.wav", content_type="audio/wav")
                    t_sent = time.monotonic()
                    task = asyncio.create_task(session.post(f"{server}/v1/transcribe", data=form))
                    pending.append((task, t_sent))
                    idx += 1
                    next_send += interval

                done_idx = []
                for pi, (task, t_sent) in enumerate(pending):
                    if task.done():
                        try:
                            resp = task.result()
                            await resp.json()
                            resp.close()
                            completed.append({
                                "latency": time.monotonic() - t_sent,
                                "ok": resp.status == 200,
                            })
                        except Exception:
                            completed.append({"latency": time.monotonic() - t_sent, "ok": False})
                        done_idx.append(pi)

                for pi in sorted(done_idx, reverse=True):
                    pending.pop(pi)

                await asyncio.sleep(0.01)

            # drain remaining
            for task, t_sent in pending:
                try:
                    resp = await asyncio.wait_for(task, timeout=10)
                    await resp.json()
                    resp.close()
                    completed.append({"latency": time.monotonic() - t_sent, "ok": resp.status == 200})
                except Exception:
                    completed.append({"latency": time.monotonic() - t_sent, "ok": False})

        ok = [c for c in completed if c["ok"]]
        fail = [c for c in completed if not c["ok"]]
        latencies = sorted([c["latency"] for c in ok])
        actual_rps = len(ok) / duration_per_level if ok else 0

        entry = {
            "target_rps": target_rps,
            "actual_rps": round(actual_rps, 2),
            "sent": len(completed),
            "successes": len(ok),
            "failures": len(fail),
            "utilization_pct": round(actual_rps / target_rps * 100, 1) if target_rps > 0 else 0,
        }
        if latencies:
            entry["latency_p50_ms"] = round(latencies[len(latencies) // 2] * 1000, 1)
            entry["latency_p95_ms"] = round(latencies[int(len(latencies) * 0.95)] * 1000, 1)
            entry["latency_p99_ms"] = round(latencies[int(len(latencies) * 0.99)] * 1000, 1)

        results.append(entry)
        log.info(f"    -> actual={actual_rps:.1f} RPS ({entry['utilization_pct']}%), p50={entry.get('latency_p50_ms', 0)}ms")

    return {"test": "saturation_ramp", "results": results}


async def test_io_isolation(server):
    """Compare different audio sizes to isolate I/O overhead from compute."""
    log.info("Test 4: I/O isolation (upload size vs latency)")
    results = []

    for dur, label in [(0.1, "100ms"), (0.5, "500ms"), (1.0, "1s"), (5.0, "5s"), (15.0, "15s"), (30.0, "30s")]:
        audio = generate_wav(dur)
        size_kb = len(audio) / 1024

        latencies = []
        async with aiohttp.ClientSession() as session:
            for i in range(20):
                r = await timed_request(session, server, audio, f"io_{label}_{i}.wav")
                if r["ok"]:
                    latencies.append(r)

        if latencies:
            server_times = [l["server_time"] for l in latencies]
            results.append({
                "duration": label,
                "size_kb": round(size_kb, 1),
                "server_p50_ms": round(sorted(server_times)[len(server_times) // 2] * 1000, 1),
                "server_p95_ms": round(sorted(server_times)[int(len(server_times) * 0.95)] * 1000, 1),
                "form_build_ms": round(sum(l["form_build"] for l in latencies) / len(latencies) * 1000, 2),
                "body_read_ms": round(sum(l["body_read"] for l in latencies) / len(latencies) * 1000, 2),
            })

    return {"test": "io_isolation", "results": results}


async def test_queue_pressure(server, audio_files):
    """Measure queue wait time by comparing serial vs pipelined latency."""
    log.info("Test 5: Queue pressure analysis")

    # Serial: no queueing
    serial_latencies = []
    async with aiohttp.ClientSession() as session:
        for i in range(30):
            data = audio_files[i % len(audio_files)]
            r = await timed_request(session, server, data, f"qserial_{i}.wav")
            if r["ok"]:
                serial_latencies.append(r["server_time"])

    # Pipelined: requests overlap → some queue
    pipeline_depths = [2, 4, 8, 16, 32]
    results = []

    for depth in pipeline_depths:
        log.info(f"  Pipeline depth={depth}")
        pipe_latencies = []
        sem = asyncio.Semaphore(depth)
        async with aiohttp.ClientSession() as session:
            async def do_one(idx):
                async with sem:
                    data = audio_files[idx % len(audio_files)]
                    return await timed_request(session, server, data, f"qpipe_{depth}_{idx}.wav")

            tasks = [do_one(i) for i in range(60)]
            responses = await asyncio.gather(*tasks)
            pipe_latencies = [r["server_time"] for r in responses if r["ok"]]

        serial_p50 = sorted(serial_latencies)[len(serial_latencies) // 2]
        pipe_p50 = sorted(pipe_latencies)[len(pipe_latencies) // 2]

        results.append({
            "pipeline_depth": depth,
            "serial_p50_ms": round(serial_p50 * 1000, 1),
            "pipelined_p50_ms": round(pipe_p50 * 1000, 1),
            "queue_overhead_ms": round((pipe_p50 - serial_p50) * 1000, 1),
            "queue_overhead_pct": round((pipe_p50 - serial_p50) / serial_p50 * 100, 1),
        })

    return {"test": "queue_pressure", "serial_p50_ms": round(sorted(serial_latencies)[len(serial_latencies) // 2] * 1000, 1), "results": results}


async def test_network_overhead(server):
    """Measure pure network round-trip by hitting /health repeatedly."""
    log.info("Test 6: Network overhead (health endpoint)")
    latencies = []
    async with aiohttp.ClientSession() as session:
        for i in range(200):
            t0 = time.monotonic()
            async with session.get(f"{server}/health") as resp:
                await resp.json()
            latencies.append(time.monotonic() - t0)

    latencies.sort()
    return {
        "test": "network_overhead",
        "requests": 200,
        "p50_ms": round(latencies[100] * 1000, 2),
        "p95_ms": round(latencies[190] * 1000, 2),
        "p99_ms": round(latencies[198] * 1000, 2),
        "max_ms": round(latencies[-1] * 1000, 2),
    }


def analyze_bottleneck(report):
    """Derive root cause and recommendations from all test results."""
    analysis = {"bottleneck_candidates": [], "root_cause": "", "recommendations": []}

    # 1. Check network overhead
    net = report["network_overhead"]
    net_p50 = net["p50_ms"]
    if net_p50 > 5:
        analysis["bottleneck_candidates"].append(f"Network: {net_p50}ms p50 round-trip (port-forward overhead)")

    # 2. Check I/O scaling
    io_test = report["io_isolation"]
    if len(io_test["results"]) >= 2:
        short = io_test["results"][0]["server_p50_ms"]
        long = io_test["results"][-1]["server_p50_ms"]
        compute_ratio = long / short if short > 0 else 0
        analysis["compute_scaling_ratio"] = round(compute_ratio, 2)
        if compute_ratio > 5:
            analysis["bottleneck_candidates"].append(
                f"GPU compute scales {compute_ratio}x from shortest to longest audio — compute-bound"
            )

    # 3. Burst efficiency
    burst = report["burst_patterns"]
    if burst["results"]:
        single = burst["results"][0]["effective_rps"]
        best_burst = max(r["effective_rps"] for r in burst["results"])
        batch_speedup = best_burst / single if single > 0 else 0
        analysis["batch_speedup"] = round(batch_speedup, 2)

        # Find where burst efficiency peaks
        peak = max(burst["results"], key=lambda r: r["effective_rps"])
        analysis["peak_burst_size"] = peak["burst_size"]
        analysis["peak_burst_rps"] = peak["effective_rps"]

    # 4. Saturation knee
    ramp = report["saturation_ramp"]
    if ramp["results"]:
        knee_rps = None
        for r in ramp["results"]:
            if r["utilization_pct"] < 90:
                knee_rps = r["target_rps"]
                break
        if knee_rps:
            analysis["saturation_knee_rps"] = knee_rps
            analysis["bottleneck_candidates"].append(
                f"Saturation knee at {knee_rps} target RPS — server can't keep up beyond this"
            )

    # 5. Queue overhead
    queue = report["queue_pressure"]
    if queue["results"]:
        max_overhead = max(r["queue_overhead_pct"] for r in queue["results"])
        analysis["max_queue_overhead_pct"] = round(max_overhead, 1)
        if max_overhead > 50:
            analysis["bottleneck_candidates"].append(
                f"Queue wait adds {max_overhead}% overhead at max pipeline depth"
            )

    # 6. Serial baseline
    serial = report["serial_baseline"]
    gpu_time = serial["avg_gpu_time_ms"]
    serial_rps = serial["serial_rps"]

    # Theoretical max: if GPU processes batch_size=32 in one go at the same time as a single request
    # then theoretical = 32 * serial_rps
    theoretical_max = 32 * serial_rps
    actual_peak = max(r["actual_rps"] for r in ramp["results"]) if ramp["results"] else 0
    efficiency = actual_peak / theoretical_max * 100 if theoretical_max > 0 else 0

    analysis["gpu_time_per_request_ms"] = gpu_time
    analysis["serial_rps"] = serial_rps
    analysis["theoretical_max_rps"] = round(theoretical_max, 1)
    analysis["actual_peak_rps"] = round(actual_peak, 1)
    analysis["gpu_efficiency_pct"] = round(efficiency, 1)

    # Root cause determination
    if efficiency < 50:
        analysis["root_cause"] = (
            f"GPU batch utilization is {efficiency:.0f}% of theoretical maximum. "
            f"Single-request GPU time is {gpu_time:.0f}ms → theoretical max with batch_size=32 is {theoretical_max:.0f} RPS, "
            f"but actual peak is {actual_peak:.0f} RPS. "
            "The GPU-dedicated thread processes requests sequentially from the queue. "
            "Even with batching, the single-threaded GPU loop serializes all inference — "
            "the batch assembler (max_wait_seconds=0.1s) adds up to 100ms idle time per batch cycle, "
            "and the Python GIL + queue overhead further reduce throughput."
        )
    else:
        analysis["root_cause"] = (
            f"GPU is well-utilized at {efficiency:.0f}% efficiency. "
            f"Throughput is near the theoretical limit of {theoretical_max:.0f} RPS."
        )

    # Recommendations
    recs = []

    if efficiency < 70:
        recs.append({
            "priority": "HIGH",
            "action": "Reduce max_wait_seconds from 0.1s to 0.02-0.05s",
            "expected_impact": "Faster batch flush → less idle GPU time between batches",
            "risk": "Smaller batches at low load, slightly less efficient",
        })

    if analysis.get("peak_burst_size", 0) < 32:
        recs.append({
            "priority": "HIGH",
            "action": f"Tune max_batch_size — peak efficiency at burst_size={analysis.get('peak_burst_size', 'N/A')}",
            "expected_impact": "Match batch size to actual throughput pattern",
            "risk": "None",
        })

    recs.append({
        "priority": "MEDIUM",
        "action": "Enable torch.compile on the batch model (compile: true in config)",
        "expected_impact": "~20-30% reduction in per-request GPU time via kernel fusion",
        "risk": "Longer warmup (first request ~60s), requires PyTorch 2.x",
    })

    if net_p50 > 2:
        recs.append({
            "priority": "LOW",
            "action": f"Reduce port-forward overhead ({net_p50}ms per round-trip)",
            "expected_impact": "Direct pod access or service mesh would save ~{0}ms/request".format(round(net_p50)),
            "risk": "Infrastructure change",
        })

    recs.append({
        "priority": "MEDIUM",
        "action": "Scale horizontally — add a second L4 GPU pod behind a load balancer",
        "expected_impact": f"~2x throughput to {actual_peak * 2:.0f} RPS",
        "risk": "Cost: +1 L4 GPU ($0.7/hr spot)",
    })

    recs.append({
        "priority": "LOW",
        "action": "Upgrade to L40S or A100 for single-node scaling",
        "expected_impact": "~2-4x GPU compute speed, higher batch throughput ceiling",
        "risk": "Cost: L40S ~3x L4 price, A100 ~5x",
    })

    analysis["recommendations"] = recs
    return analysis


async def main():
    parser = argparse.ArgumentParser(description="Level 2 chaos benchmark")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    server = args.server
    log.info(f"Chaos benchmark: server={server}")

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{server}/health") as resp:
            health = await resp.json()
            log.info(f"Server health: {health}")

    audio_files = prepare_real_audio()
    if not audio_files:
        log.error("No audio samples")
        return
    log.info(f"Loaded {len(audio_files)} audio samples")

    report = {
        "benchmark": "Level 2 Industrial — Chaos & Bottleneck Analysis",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": server,
        "model": "nvidia/parakeet-tdt-0.6b-v3",
        "gpu": "NVIDIA L4 (24GB)",
    }

    # Run all tests
    report["network_overhead"] = await test_network_overhead(server)
    report["serial_baseline"] = await test_serial_baseline(server, audio_files)
    report["burst_patterns"] = await test_burst_patterns(server, audio_files)
    report["io_isolation"] = await test_io_isolation(server)
    report["queue_pressure"] = await test_queue_pressure(server, audio_files)
    report["saturation_ramp"] = await test_saturation_ramp(server, audio_files)

    # Analyze
    report["analysis"] = analyze_bottleneck(report)

    output = json.dumps(report, indent=2)
    print(output)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        log.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
