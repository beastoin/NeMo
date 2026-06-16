#!/usr/bin/env python3
"""
Streaming ASR benchmark for Nemotron Speech Streaming 0.6B.

Measures WebSocket streaming performance: per-chunk latency, end-to-end
latency, RTF, and concurrent stream capacity across latency modes.

Usage:
    python3 stream_benchmark.py --server ws://localhost:8000
    python3 stream_benchmark.py --server ws://localhost:8000 --sustained-minutes 5
    python3 stream_benchmark.py --server ws://localhost:8000 --concurrency 1,2,4,8,16,32
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
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stream-benchmark")

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "asr-stream-test-audio")


def generate_pcm16(duration_seconds, sample_rate=16000):
    """Generate silence as raw PCM16 bytes."""
    num_samples = int(duration_seconds * sample_rate)
    samples = np.zeros(num_samples, dtype=np.int16)
    return samples.tobytes()


def _tts_to_pcm16(text, sample_rate=16000):
    """Generate speech via espeak-ng, return raw PCM16 bytes."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["espeak-ng", "-w", tmp_path, "--stdin", "-s", "150", "-a", "100"],
            input=text, text=True, capture_output=True, check=True, timeout=30,
        )
        import soundfile as sf
        data, sr = sf.read(tmp_path, dtype='int16')
        if sr != sample_rate:
            data_f = data.astype(np.float32) / 32768.0
            import librosa
            data_f = librosa.resample(data_f, orig_sr=sr, target_sr=sample_rate)
            data = (data_f * 32768.0).clip(-32768, 32767).astype(np.int16)
        return data.tobytes()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def prepare_audio_samples(sample_rate=16000):
    """Prepare real speech PCM16 samples for benchmarking."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    samples = []
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming how we interact with technology.",
        "Please remember to save your work before closing the application.",
        "Speech recognition accuracy has improved dramatically over the past decade.",
        "The conference will be held next Tuesday at three o'clock in the afternoon.",
    ]
    for i, s in enumerate(sentences):
        cache_path = os.path.join(_CACHE_DIR, f"stream_tts_{i}.pcm16")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                pcm_data = f.read()
        else:
            pcm_data = _tts_to_pcm16(s, sample_rate)
            if pcm_data is None:
                continue
            with open(cache_path, "wb") as f:
                f.write(pcm_data)
        duration = len(pcm_data) / (2 * sample_rate)
        samples.append({"pcm16": pcm_data, "text": s, "duration_s": duration, "name": f"tts-{i}"})

    if not samples:
        log.warning("espeak-ng not available, falling back to synthetic speech tone")
        for dur in [3.0, 5.0, 8.0]:
            t = np.linspace(0, dur, int(dur * sample_rate), dtype=np.float32)
            tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
            samples.append({
                "pcm16": tone.tobytes(),
                "text": "(synthetic tone)",
                "duration_s": dur,
                "name": f"tone-{dur}s",
            })

    return samples


def chunk_audio(pcm16_bytes, chunk_duration_ms=160, sample_rate=16000):
    """Split PCM16 audio into chunks of chunk_duration_ms."""
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    chunk_bytes = chunk_samples * 2
    chunks = []
    for i in range(0, len(pcm16_bytes), chunk_bytes):
        chunk = pcm16_bytes[i:i + chunk_bytes]
        if len(chunk) > 0:
            chunks.append(chunk)
    return chunks


async def stream_one_session(ws_url, pcm16_bytes, chunk_duration_ms=160,
                             sample_rate=16000, realtime_pace=True,
                             ws_timeout=30):
    """Run one streaming session and collect per-chunk latencies."""
    import websockets

    chunks = chunk_audio(pcm16_bytes, chunk_duration_ms, sample_rate)
    chunk_interval = chunk_duration_ms / 1000.0

    result = {
        "ok": True,
        "chunks_sent": 0,
        "chunk_latencies_ms": [],
        "partial_transcripts": [],
        "final_text": "",
        "total_audio_s": len(pcm16_bytes) / (2 * sample_rate),
        "wall_clock_s": 0,
    }

    t_session_start = time.monotonic()
    try:
        async with websockets.connect(
            f"{ws_url}/v1/stream", max_size=2**20, open_timeout=ws_timeout, close_timeout=ws_timeout
        ) as ws:
            open_msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=ws_timeout))
            if "error" in open_msg:
                result["ok"] = False
                result["error"] = open_msg["error"]
                return result

            for i, chunk in enumerate(chunks):
                t_chunk_start = time.monotonic()
                await ws.send(chunk)
                resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=ws_timeout))
                chunk_lat = (time.monotonic() - t_chunk_start) * 1000
                result["chunk_latencies_ms"].append(chunk_lat)
                result["chunks_sent"] += 1

                partial = resp.get("partial_transcript", "")
                if partial:
                    result["partial_transcripts"].append(partial)

                if realtime_pace and i < len(chunks) - 1:
                    elapsed = time.monotonic() - t_session_start
                    expected = (i + 1) * chunk_interval
                    if expected > elapsed:
                        await asyncio.sleep(expected - elapsed)

            await ws.send(json.dumps({"action": "close"}))
            close_msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=ws_timeout))
            result["final_text"] = close_msg.get("final_text", "")

    except Exception as exc:
        result["ok"] = False
        result["error"] = str(exc)[:200]

    result["wall_clock_s"] = time.monotonic() - t_session_start
    return result


async def concurrency_sweep(ws_url, audio_samples, levels, chunk_duration_ms=160,
                            sample_rate=16000, realtime_pace=True):
    """Run concurrent streaming sessions at various levels."""
    results = []
    for conc in levels:
        log.info(f"  Concurrency={conc}")
        semaphore = asyncio.Semaphore(conc)

        async def run_one(idx):
            async with semaphore:
                sample = audio_samples[idx % len(audio_samples)]
                return await stream_one_session(
                    ws_url, sample["pcm16"], chunk_duration_ms,
                    sample_rate, realtime_pace,
                )

        t_wall_start = time.monotonic()
        tasks = [run_one(i) for i in range(conc)]
        responses = await asyncio.gather(*tasks)
        wall_clock = time.monotonic() - t_wall_start

        ok = [r for r in responses if r["ok"]]
        fail = [r for r in responses if not r["ok"]]
        all_chunk_lats = []
        for r in ok:
            all_chunk_lats.extend(r["chunk_latencies_ms"])
        all_chunk_lats.sort()

        total_audio = sum(r["total_audio_s"] for r in ok)

        entry = {
            "concurrency": conc,
            "streams": len(responses),
            "successes": len(ok),
            "failures": len(fail),
            "wall_clock_s": round(wall_clock, 2),
            "total_audio_s": round(total_audio, 2),
            "rtf": round(wall_clock / total_audio, 4) if total_audio > 0 else None,
            "rtfx": round(total_audio / wall_clock, 1) if wall_clock > 0 else 0,
        }

        if all_chunk_lats:
            entry["chunk_latency_p50_ms"] = round(all_chunk_lats[len(all_chunk_lats) // 2], 1)
            entry["chunk_latency_p90_ms"] = round(all_chunk_lats[int(len(all_chunk_lats) * 0.90)], 1)
            entry["chunk_latency_p95_ms"] = round(all_chunk_lats[int(len(all_chunk_lats) * 0.95)], 1)
            entry["chunk_latency_p99_ms"] = round(all_chunk_lats[int(len(all_chunk_lats) * 0.99)], 1)
            entry["chunk_latency_max_ms"] = round(all_chunk_lats[-1], 1)
            entry["total_chunks"] = len(all_chunk_lats)

        if fail:
            entry["error_sample"] = [r.get("error", "unknown") for r in fail[:3]]

        results.append(entry)
        log.info(
            f"    -> {len(ok)}/{len(responses)} ok, "
            f"RTFx={entry.get('rtfx', 0)}x, "
            f"chunk p50={entry.get('chunk_latency_p50_ms', 0)}ms, "
            f"p99={entry.get('chunk_latency_p99_ms', 0)}ms"
        )

    return results


async def sustained_streaming(ws_url, audio_samples, concurrency, duration_minutes,
                              chunk_duration_ms=160, sample_rate=16000):
    """Run sustained concurrent streams for a duration, cycling sessions."""
    log.info(f"  Sustained streaming: concurrency={concurrency}, duration={duration_minutes}min")
    deadline = time.monotonic() + duration_minutes * 60
    completed = []
    active = set()
    idx = 0

    async def run_session(session_idx):
        sample = audio_samples[session_idx % len(audio_samples)]
        return await stream_one_session(
            ws_url, sample["pcm16"], chunk_duration_ms,
            sample_rate, realtime_pace=True,
        )

    while time.monotonic() < deadline or active:
        while len(active) < concurrency and time.monotonic() < deadline:
            task = asyncio.create_task(run_session(idx))
            active.add(task)
            idx += 1

        if active:
            done, active = await asyncio.wait(active, timeout=0.5, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                completed.append(t.result())

    ok = [r for r in completed if r["ok"]]
    fail = [r for r in completed if not r["ok"]]
    all_chunk_lats = []
    for r in ok:
        all_chunk_lats.extend(r["chunk_latencies_ms"])
    all_chunk_lats.sort()
    total_audio = sum(r["total_audio_s"] for r in ok)
    elapsed = duration_minutes * 60

    entry = {
        "duration_minutes": duration_minutes,
        "concurrency": concurrency,
        "total_sessions": len(completed),
        "successes": len(ok),
        "failures": len(fail),
        "total_audio_s": round(total_audio, 2),
        "audio_throughput_hrs_per_hr": round(total_audio / elapsed * 3600 / 3600, 2) if elapsed > 0 else 0,
        "sessions_per_minute": round(len(ok) / duration_minutes, 1),
    }

    if all_chunk_lats:
        entry["chunk_latency_p50_ms"] = round(all_chunk_lats[len(all_chunk_lats) // 2], 1)
        entry["chunk_latency_p95_ms"] = round(all_chunk_lats[int(len(all_chunk_lats) * 0.95)], 1)
        entry["chunk_latency_p99_ms"] = round(all_chunk_lats[int(len(all_chunk_lats) * 0.99)], 1)
        entry["chunk_latency_max_ms"] = round(all_chunk_lats[-1], 1)
        entry["total_chunks"] = len(all_chunk_lats)

    if fail:
        entry["error_sample"] = [r.get("error", "unknown") for r in fail[:3]]

    return entry


async def latency_mode_comparison(ws_url, audio_samples, modes=None):
    """Compare streaming performance across latency modes via admin API."""
    import aiohttp

    if modes is None:
        modes = ["80ms", "160ms", "480ms", "1040ms"]

    http_url = ws_url.replace("ws://", "http://").replace("wss://", "https://")
    results = []

    for mode in modes:
        log.info(f"  Latency mode: {mode}")
        log.info(f"    (Note: latency mode change requires server restart with config override)")

        sample = audio_samples[0]
        r = await stream_one_session(ws_url, sample["pcm16"], realtime_pace=False)

        if r["ok"]:
            lats = sorted(r["chunk_latencies_ms"])
            entry = {
                "latency_mode": mode,
                "chunks_sent": r["chunks_sent"],
                "wall_clock_s": round(r["wall_clock_s"], 3),
                "audio_duration_s": round(r["total_audio_s"], 2),
                "rtf": round(r["wall_clock_s"] / r["total_audio_s"], 4) if r["total_audio_s"] > 0 else None,
                "final_text": r["final_text"][:100],
            }
            if lats:
                entry["chunk_latency_p50_ms"] = round(lats[len(lats) // 2], 1)
                entry["chunk_latency_p95_ms"] = round(lats[int(len(lats) * 0.95)], 1)
                entry["chunk_latency_max_ms"] = round(lats[-1], 1)
            results.append(entry)
            log.info(f"    -> ok, RTF={entry['rtf']}, p50={entry.get('chunk_latency_p50_ms', 0)}ms")
        else:
            results.append({"latency_mode": mode, "error": r.get("error", "unknown")})
            log.info(f"    -> FAILED: {r.get('error', 'unknown')}")

        # Only run current mode — mode switching requires restart
        break

    return results


async def quality_check(ws_url, audio_samples, chunk_duration_ms=160, sample_rate=16000):
    """Run a single stream per sample and check transcription quality."""
    log.info("  Quality check: single stream per sample")
    results = []
    for sample in audio_samples:
        r = await stream_one_session(
            ws_url, sample["pcm16"], chunk_duration_ms, sample_rate, realtime_pace=False,
        )
        entry = {
            "name": sample["name"],
            "reference": sample["text"],
            "hypothesis": r.get("final_text", ""),
            "ok": r["ok"],
            "audio_duration_s": round(sample["duration_s"], 2),
            "wall_clock_s": round(r["wall_clock_s"], 3),
            "chunks_sent": r.get("chunks_sent", 0),
        }
        if r["ok"] and r["chunk_latencies_ms"]:
            lats = sorted(r["chunk_latencies_ms"])
            entry["chunk_latency_p50_ms"] = round(lats[len(lats) // 2], 1)

        results.append(entry)
        status = "ok" if r["ok"] else "FAIL"
        log.info(f"    [{status}] {sample['name']}: \"{r.get('final_text', '')[:60]}\"")

    return results


async def main():
    parser = argparse.ArgumentParser(description="Streaming ASR benchmark")
    parser.add_argument("--server", default="ws://localhost:8000",
                        help="WebSocket server URL (ws:// or wss://)")
    parser.add_argument("--sustained-minutes", type=int, default=3)
    parser.add_argument("--concurrency", type=str, default="1,2,4,8,16,32",
                        help="Comma-separated concurrency levels")
    parser.add_argument("--chunk-ms", type=int, default=160,
                        help="Chunk duration in ms (must match server config)")
    parser.add_argument("--no-realtime-pace", action="store_true",
                        help="Send chunks as fast as possible (stress test)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON report to file")
    args = parser.parse_args()

    ws_url = args.server
    conc_levels = [int(x) for x in args.concurrency.split(",")]
    realtime_pace = not args.no_realtime_pace

    log.info(f"Streaming benchmark: server={ws_url}")
    log.info(f"  Concurrency levels: {conc_levels}")
    log.info(f"  Chunk duration: {args.chunk_ms}ms")
    log.info(f"  Realtime pace: {realtime_pace}")

    # Health check via HTTP
    import aiohttp
    http_url = ws_url.replace("ws://", "http://").replace("wss://", "https://")
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{http_url}/health") as resp:
            health = await resp.json()
            log.info(f"Server health: {health}")

    log.info("Preparing audio samples...")
    audio_samples = prepare_audio_samples()
    log.info(f"Loaded {len(audio_samples)} samples "
             f"(total {sum(s['duration_s'] for s in audio_samples):.1f}s audio)")

    report = {
        "benchmark": "NeMo ASR Streaming — Performance Report",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "server": ws_url,
        "model": "nvidia/nemotron-speech-streaming-en-0.6b",
        "chunk_duration_ms": args.chunk_ms,
        "realtime_pace": realtime_pace,
        "audio_samples": len(audio_samples),
    }

    # Phase 1: Quality check
    log.info("Phase 1: Quality check")
    report["quality"] = await quality_check(ws_url, audio_samples, args.chunk_ms)

    # Phase 2: Warmup
    log.info("Phase 2: Warmup (5 sequential streams)")
    for i in range(5):
        sample = audio_samples[i % len(audio_samples)]
        await stream_one_session(ws_url, sample["pcm16"], args.chunk_ms, realtime_pace=False)
    log.info("  Warmup complete")

    # Phase 3: Concurrency sweep
    log.info("Phase 3: Concurrency sweep")
    report["concurrency_sweep"] = await concurrency_sweep(
        ws_url, audio_samples, conc_levels, args.chunk_ms,
        realtime_pace=realtime_pace,
    )

    # Phase 4: Sustained load
    log.info(f"Phase 4: Sustained load ({args.sustained_minutes} min)")
    report["sustained_load"] = await sustained_streaming(
        ws_url, audio_samples, concurrency=max(4, min(conc_levels)),
        duration_minutes=args.sustained_minutes,
        chunk_duration_ms=args.chunk_ms,
    )

    # Summary
    sweep = report["concurrency_sweep"]
    peak_rtfx = max(r.get("rtfx", 0) for r in sweep)
    peak_conc = next(r["concurrency"] for r in sweep if r.get("rtfx", 0) == peak_rtfx)
    total_failures = sum(r["failures"] for r in sweep)
    total_streams = sum(r["streams"] for r in sweep)

    report["summary"] = {
        "peak_rtfx": peak_rtfx,
        "peak_rtfx_concurrency": peak_conc,
        "peak_rtf": round(1 / peak_rtfx, 4) if peak_rtfx > 0 else None,
        "total_streams_all_phases": total_streams + 5 + report["sustained_load"]["total_sessions"],
        "total_failures_all_phases": total_failures + report["sustained_load"]["failures"],
        "sustained_sessions_per_min": report["sustained_load"]["sessions_per_minute"],
    }

    output = json.dumps(report, indent=2)
    print(output)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        log.info(f"Report saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
