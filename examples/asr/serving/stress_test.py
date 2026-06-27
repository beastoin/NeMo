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
Stress and quality test client for the ASR serving server.

Modes:
    batch   - Throughput stress test with concurrent requests (synthetic audio)
    stream  - Concurrent WebSocket streaming sessions (synthetic audio)
    mixed   - Batch + streaming simultaneously
    quality - Transcription accuracy test with real speech audio

Usage:
    python stress_test.py --server http://localhost:8000 --mode batch --concurrency 32
    python stress_test.py --server http://localhost:8000 --mode stream --streams 16
    python stress_test.py --server http://localhost:8000 --mode mixed
    python stress_test.py --server http://localhost:8000 --mode quality
"""

import argparse
import asyncio
import hashlib
import io
import json
import logging
import os
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import aiohttp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stress-test")

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "asr-test-audio")

QUALITY_SAMPLES = [
    {
        "url": "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav",
        "expected_words": ["wish", "see", "phoebe", "portrait"],
        "name": "librispeech-2086",
        "min_words": 10,
    },
    {
        "url": "https://dldata-public.s3.us-east-2.amazonaws.com/VAD_demo.wav",
        "expected_words": [],
        "name": "vad-demo",
        "min_words": 5,
    },
]

TTS_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming how we interact with technology.",
    "Please remember to save your work before closing the application.",
]


def generate_test_audio(duration_seconds: float = 5.0, sample_rate: int = 16000) -> bytes:
    """Generate a silent WAV file for throughput testing."""
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


def _ensure_cache_dir():
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _tts_to_wav(text: str, output_path: str) -> bool:
    """Generate speech audio from text using espeak-ng (available on most Linux)."""
    try:
        subprocess.run(
            ["espeak-ng", "-w", output_path, "--stdin", "-s", "150", "-a", "100"],
            input=text,
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    try:
        subprocess.run(
            ["espeak", "-w", output_path, "--stdin", "-s", "150"],
            input=text,
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def _download_file(url: str, output_path: str) -> bool:
    """Download a file, with caching."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True
    try:
        import urllib.request

        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as exc:
        log.warning(f"Download failed for {url}: {exc}")
        return False


def prepare_quality_audio() -> list[dict]:
    """Prepare audio samples for quality testing.

    Returns list of dicts, each with:
        path: str — local audio file path
        name: str — human label
        mode: "tts" | "download"
        ref: str — exact reference text (TTS only)
        expected_words: list[str] — words that should appear in transcript (downloads)
        min_words: int — minimum transcript length (downloads)
    """
    _ensure_cache_dir()
    samples = []

    for sent_idx, sentence in enumerate(TTS_SENTENCES):
        path = os.path.join(_CACHE_DIR, f"tts_{sent_idx}.wav")
        if not os.path.exists(path):
            if _tts_to_wav(sentence, path):
                log.info(f"Generated TTS audio: {path}")
            else:
                log.warning(f"TTS unavailable, skipping sentence {sent_idx}")
                continue
        samples.append({
            "path": path,
            "ref": sentence.lower().rstrip(".!?,"),
            "name": f"tts-{sent_idx}",
            "mode": "tts",
        })

    for sample in QUALITY_SAMPLES:
        ext = Path(sample["url"]).suffix
        cache_key = hashlib.md5(sample["url"].encode()).hexdigest()[:12]
        path = os.path.join(_CACHE_DIR, f"{cache_key}{ext}")
        if _download_file(sample["url"], path):
            samples.append({
                "path": path,
                "name": sample["name"],
                "mode": "download",
                "expected_words": sample.get("expected_words", []),
                "min_words": sample.get("min_words", 3),
            })
        else:
            log.warning(f"Skipping {sample['name']} (download failed)")

    return samples


def _normalize_text(text: str) -> str:
    """Normalize text for WER: lowercase, remove all punctuation, collapse whitespace."""
    import re

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate between reference and hypothesis."""
    ref_words = _normalize_text(reference).split()
    hyp_words = _normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


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
    real_audio: bool = False,
) -> StressTestResult:
    """Fire concurrent batch transcription requests."""
    label = f"batch-real (concurrency={concurrency})" if real_audio else f"batch (concurrency={concurrency})"
    result = StressTestResult(label)
    semaphore = asyncio.Semaphore(concurrency)

    if real_audio:
        samples = prepare_quality_audio()
        if not samples:
            raise RuntimeError("No real audio available. Install espeak-ng: apt install espeak-ng")
        audio_files = []
        for s in samples:
            with open(s["path"], "rb") as f:
                suffix = Path(s["path"]).suffix or ".wav"
                audio_files.append((f.read(), s["name"], suffix))
        log.info(f"Loaded {len(audio_files)} real audio samples for stress test")
    else:
        audio_data = generate_test_audio(audio_duration)
        audio_files = [(audio_data, "synthetic", ".wav")]

    async def send_one(session: aiohttp.ClientSession, idx: int):
        async with semaphore:
            data, name, suffix = audio_files[idx % len(audio_files)]
            t0 = time.monotonic()
            try:
                form = aiohttp.FormData()
                form.add_field("file", data, filename=f"{name}_{idx}{suffix}", content_type="audio/wav")
                async with session.post(f"{server}/v1/transcribe", data=form) as resp:
                    if resp.status == 200:
                        body = await resp.json()
                        text = body.get("text", "")
                        if real_audio and not text.strip():
                            result.record_failure(f"Empty transcription for {name}")
                        else:
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
    batch_task = asyncio.create_task(batch_stress_test(server, batch_concurrency, batch_requests))
    stream_task = asyncio.create_task(stream_stress_test(server, stream_count, stream_chunks))
    batch_result, stream_result = await asyncio.gather(batch_task, stream_task)
    return [batch_result, stream_result]


async def quality_test(server: str) -> dict:
    """Test transcription quality with real speech audio.

    Uses TTS-generated speech (espeak-ng) for exact WER measurement and
    public audio samples (NeMo tutorial files) for smoke-test validation.
    """
    log.info("Preparing quality test audio...")
    samples = prepare_quality_audio()
    if not samples:
        return {"error": "No test audio available. Install espeak-ng: apt install espeak-ng"}

    log.info(f"Testing {len(samples)} audio samples")
    results = []

    async with aiohttp.ClientSession() as session:
        for sample in samples:
            t0 = time.monotonic()
            with open(sample["path"], "rb") as f:
                audio_data = f.read()

            suffix = Path(sample["path"]).suffix or ".wav"
            form = aiohttp.FormData()
            form.add_field("file", audio_data, filename=f"{sample['name']}{suffix}", content_type="audio/wav")

            try:
                async with session.post(f"{server}/v1/transcribe", data=form) as resp:
                    latency = time.monotonic() - t0
                    if resp.status != 200:
                        text = await resp.text()
                        results.append({
                            "name": sample["name"],
                            "status": "error",
                            "http_status": resp.status,
                            "detail": text[:200],
                        })
                        continue

                    data = await resp.json()
                    hypothesis = data.get("text", "").lower().strip()

                    entry = {
                        "name": sample["name"],
                        "status": "ok",
                        "latency_ms": round(latency * 1000, 1),
                        "hyp_words": len(hypothesis.split()),
                        "hypothesis": hypothesis[:200],
                    }

                    if sample["mode"] == "tts":
                        reference = sample["ref"]
                        wer = compute_wer(reference, hypothesis)
                        entry["wer"] = round(wer, 4)
                        entry["ref_words"] = len(reference.split())
                        entry["reference"] = reference[:100]
                        log.info(f"  {sample['name']}: WER={wer:.1%} latency={latency:.1f}s")
                    else:
                        expected = sample.get("expected_words", [])
                        min_words = sample.get("min_words", 3)
                        hyp_words = hypothesis.split()
                        matched = [w for w in expected if w in hyp_words]
                        entry["expected_words_matched"] = f"{len(matched)}/{len(expected)}"
                        entry["word_count_ok"] = len(hyp_words) >= min_words
                        log.info(
                            f"  {sample['name']}: {len(hyp_words)} words, "
                            f"matched {len(matched)}/{len(expected)} keywords, "
                            f"latency={latency:.1f}s"
                        )

                    results.append(entry)

            except Exception as exc:
                results.append({"name": sample["name"], "status": "error", "detail": str(exc)[:200]})

    ok_results = [r for r in results if r.get("status") == "ok"]
    tts_results = [r for r in ok_results if "wer" in r]
    avg_wer = sum(r["wer"] for r in tts_results) / len(tts_results) if tts_results else None
    avg_latency = sum(r["latency_ms"] for r in ok_results) / len(ok_results) if ok_results else None

    return {
        "test": "quality",
        "total_samples": len(samples),
        "successful": len(ok_results),
        "failed": len(results) - len(ok_results),
        "average_wer_tts": round(avg_wer, 4) if avg_wer is not None else None,
        "average_latency_ms": round(avg_latency, 1) if avg_latency is not None else None,
        "samples": results,
    }


async def main():
    parser = argparse.ArgumentParser(description="ASR server stress & quality test")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--mode", choices=["batch", "stream", "mixed", "quality"], default="batch")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--chunks", type=int, default=100)
    parser.add_argument("--audio-duration", type=float, default=5.0)
    parser.add_argument("--real-audio", action="store_true", help="Use real speech audio instead of silence")
    args = parser.parse_args()

    log.info(f"Stress test: mode={args.mode} server={args.server}")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{args.server}/health") as resp:
                health = await resp.json()
                log.info(f"Server health: {health}")
        except Exception as exc:
            log.error(f"Server not reachable: {exc}")
            return

    if args.mode == "batch":
        result = await batch_stress_test(
            args.server, args.concurrency, args.requests, args.audio_duration, args.real_audio
        )
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

    elif args.mode == "quality":
        result = await quality_test(args.server)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
