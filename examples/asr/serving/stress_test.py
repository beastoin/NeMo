# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Stress test client for the ASR serving server.

Tests batch throughput, concurrent requests, streaming sessions,
and mixed-mode workloads.

Usage:
    python stress_test.py --server http://localhost:8000 --mode batch --concurrency 32
    python stress_test.py --server http://localhost:8000 --mode stream --streams 16
    python stress_test.py --server http://localhost:8000 --mode mixed
"""

import argparse
import asyncio
import io
import json
import logging
import struct
import time
from pathlib import Path

import aiohttp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stress-test")


def generate_test_audio(duration_seconds: float = 5.0, sample_rate: int = 16000) -> bytes:
    """Generate a silent WAV file for testing."""
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


def generate_pcm_chunk(duration_ms: int = 160, sample_rate: int = 16000) -> bytes:
    """Generate a silent PCM16 chunk for streaming tests."""
    num_samples = int(duration_ms * sample_rate / 1000)
    return np.zeros(num_samples, dtype=np.int16).tobytes()


class StressTestResult:
    def __init__(self, name: str):
        self.name = name
        self.successes = 0
        self.failures = 0
        self.latencies: list[float] = []
        self.errors: list[str] = []
        self.start_time = time.monotonic()

    def record_success(self, latency: float):
        self.successes += 1
        self.latencies.append(latency)

    def record_failure(self, error: str):
        self.failures += 1
        self.errors.append(error)

    def summary(self) -> dict:
        elapsed = time.monotonic() - self.start_time
        latencies = sorted(self.latencies)
        return {
            "test": self.name,
            "elapsed_seconds": round(elapsed, 2),
            "total_requests": self.successes + self.failures,
            "successes": self.successes,
            "failures": self.failures,
            "throughput_rps": round(self.successes / elapsed, 2) if elapsed > 0 else 0,
            "latency_p50_ms": round(latencies[len(latencies) // 2] * 1000, 1) if latencies else 0,
            "latency_p95_ms": round(latencies[int(len(latencies) * 0.95)] * 1000, 1) if latencies else 0,
            "latency_p99_ms": round(latencies[int(len(latencies) * 0.99)] * 1000, 1) if latencies else 0,
            "latency_max_ms": round(max(latencies) * 1000, 1) if latencies else 0,
            "errors_sample": self.errors[:5],
        }


async def batch_stress_test(
    server: str,
    concurrency: int = 32,
    total_requests: int = 200,
    audio_duration: float = 5.0,
) -> StressTestResult:
    """Fire concurrent batch transcription requests."""
    result = StressTestResult(f"batch (concurrency={concurrency})")
    audio_data = generate_test_audio(audio_duration)
    semaphore = asyncio.Semaphore(concurrency)

    async def send_one(session: aiohttp.ClientSession, idx: int):
        async with semaphore:
            t0 = time.monotonic()
            try:
                form = aiohttp.FormData()
                form.add_field("file", audio_data, filename=f"test_{idx}.wav", content_type="audio/wav")
                async with session.post(f"{server}/v1/transcribe", data=form) as resp:
                    if resp.status == 200:
                        await resp.json()
                        result.record_success(time.monotonic() - t0)
                    else:
                        text = await resp.text()
                        result.record_failure(f"HTTP {resp.status}: {text[:200]}")
            except Exception as exc:
                result.record_failure(str(exc)[:200])

    async with aiohttp.ClientSession() as session:
        tasks = [send_one(session, i) for i in range(total_requests)]
        await asyncio.gather(*tasks)

    return result


async def stream_stress_test(
    server: str,
    streams: int = 16,
    chunks_per_stream: int = 100,
    chunk_duration_ms: int = 160,
) -> StressTestResult:
    """Open concurrent streaming sessions."""
    result = StressTestResult(f"stream (streams={streams})")
    ws_url = server.replace("http://", "ws://").replace("https://", "wss://")

    async def stream_one(idx: int):
        t0 = time.monotonic()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"{ws_url}/v1/stream") as ws:
                    init = await ws.receive_json()
                    if "error" in init:
                        result.record_failure(init["error"])
                        return

                    chunk = generate_pcm_chunk(chunk_duration_ms)
                    for _ in range(chunks_per_stream):
                        await ws.send_bytes(chunk)
                        resp = await ws.receive_json()

                    await ws.send_json({"action": "close"})
                    final = await ws.receive_json()
                    result.record_success(time.monotonic() - t0)

        except Exception as exc:
            result.record_failure(str(exc)[:200])

    tasks = [stream_one(i) for i in range(streams)]
    await asyncio.gather(*tasks)
    return result


async def mixed_stress_test(
    server: str,
    batch_concurrency: int = 16,
    batch_requests: int = 100,
    stream_count: int = 8,
    stream_chunks: int = 50,
) -> list[StressTestResult]:
    """Run batch and streaming concurrently."""
    batch_task = asyncio.create_task(
        batch_stress_test(server, batch_concurrency, batch_requests)
    )
    stream_task = asyncio.create_task(
        stream_stress_test(server, stream_count, stream_chunks)
    )
    batch_result, stream_result = await asyncio.gather(batch_task, stream_task)
    return [batch_result, stream_result]


async def main():
    parser = argparse.ArgumentParser(description="ASR server stress test")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--mode", choices=["batch", "stream", "mixed"], default="batch")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--chunks", type=int, default=100)
    parser.add_argument("--audio-duration", type=float, default=5.0)
    args = parser.parse_args()

    log.info(f"Stress test: mode={args.mode} server={args.server}")

    # Health check
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{args.server}/health") as resp:
                health = await resp.json()
                log.info(f"Server health: {health}")
        except Exception as exc:
            log.error(f"Server not reachable: {exc}")
            return

    if args.mode == "batch":
        result = await batch_stress_test(args.server, args.concurrency, args.requests, args.audio_duration)
        print(json.dumps(result.summary(), indent=2))

    elif args.mode == "stream":
        result = await stream_stress_test(args.server, args.streams, args.chunks)
        print(json.dumps(result.summary(), indent=2))

    elif args.mode == "mixed":
        results = await mixed_stress_test(
            args.server, args.concurrency, args.requests, args.streams, args.chunks
        )
        for r in results:
            print(json.dumps(r.summary(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
