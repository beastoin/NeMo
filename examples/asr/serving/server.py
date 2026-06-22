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
High-performance ASR inference server.

Serves two models via a single GPU worker thread:
  - Parakeet TDT 0.6B for batch/offline transcription (REST API)
  - Nemotron ASR Streaming 0.6B for real-time streaming (WebSocket)

Key design:
  - All GPU inference runs on a dedicated thread (gpu_worker.py)
  - Dynamic batching collects REST requests for throughput (batch_engine.py)
  - Per-client streaming sessions with cache-aware inference (stream_engine.py)
  - Backpressure: 503 when queue is full or stream limit reached

Usage:
    python server.py                           # defaults from conf/serving.yaml
    python server.py --port 8080               # override port
    python server.py --config path/to/cfg.yaml # custom config
"""

import asyncio
import functools
import gc
import json
import logging
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Disable Python's automatic cyclic GC.  Under sustained load, automatic GC
# can trigger on the async event-loop thread and free CUDA pinned-memory
# tensors there, crashing in CachingHostAllocatorImpl::free().  The GPU
# worker calls gc.collect() after every batch on its own thread instead.
gc.disable()

import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(__file__))
from batch_engine import BatchEngine, QueueFullError
from gpu_worker import GPUWorker
from stream_engine import ChunkTooLargeError, StreamEngine, StreamExpiredError, TooManyStreamsError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("asr-server")

gpu_worker: Optional[GPUWorker] = None
batch_engine: Optional[BatchEngine] = None
stream_engine: Optional[StreamEngine] = None
config: dict = {}
start_time: float = 0


def load_config(config_path: Optional[str] = None) -> dict:
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "conf", "serving.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gpu_worker, batch_engine, stream_engine, start_time
    start_time = time.monotonic()

    gpu_worker = GPUWorker()
    gpu_worker.start(
        batch_cfg=config.get("batch_model", {}),
        stream_cfg=config.get("stream_model", {}),
    )
    log.info("Waiting for GPU models to load...")
    gpu_worker.wait_ready(timeout=600)

    batcher_cfg = config.get("batcher", {})
    batch_engine = BatchEngine(
        gpu_worker=gpu_worker,
        max_batch_size=batcher_cfg.get("max_batch_size", 32),
        max_wait_seconds=batcher_cfg.get("max_wait_seconds", 0.1),
        max_queue_depth=batcher_cfg.get("max_queue_depth", 256),
    )
    await batch_engine.start()

    stream_cfg = config.get("stream", {})
    stream_engine = StreamEngine(
        gpu_worker=gpu_worker,
        max_concurrent_streams=stream_cfg.get("max_concurrent_streams", 128),
        chunk_duration_ms=stream_cfg.get("chunk_duration_ms", 160),
        sample_rate=stream_cfg.get("sample_rate", 16000),
        max_stream_duration=stream_cfg.get("max_stream_duration", 0),
        idle_timeout=stream_cfg.get("idle_timeout", 300),
        max_chunk_bytes=stream_cfg.get("max_chunk_bytes", 512 * 1024),
    )
    await stream_engine.start()

    log.info("ASR server ready")
    yield

    log.info("Shutting down...")
    await batch_engine.stop()
    await stream_engine.stop()
    gpu_worker.stop()


app = FastAPI(
    title="NeMo ASR Serving",
    description="High-performance batch + streaming ASR inference",
    lifespan=lifespan,
)


# --- Health & Metrics ---


@app.get("/health")
async def health():
    ready = gpu_worker.is_ready if gpu_worker else False
    return {
        "status": "ok" if ready else "loading",
        "ready": ready,
        "uptime_seconds": round(time.monotonic() - start_time, 1),
    }


@app.get("/metrics")
async def metrics():
    return {
        "uptime_seconds": round(time.monotonic() - start_time, 1),
        "batch": batch_engine.metrics if batch_engine else {},
        "stream": stream_engine.metrics if stream_engine else {},
    }


@app.get("/admin/config")
async def get_tuning():
    """Get current tunable parameters."""
    return {
        "max_batch_size": batch_engine._max_batch_size if batch_engine else None,
        "max_wait_seconds": batch_engine._max_wait_seconds if batch_engine else None,
        "max_queue_depth": batch_engine._max_queue_depth if batch_engine else None,
        "gpu_poll_timeout": gpu_worker._batch_poll_timeout if gpu_worker else None,
    }


@app.post("/admin/config")
async def set_tuning(
    max_batch_size: Optional[int] = Query(None, ge=1, le=256),
    max_wait_seconds: Optional[float] = Query(None, ge=0.001, le=5.0),
    max_queue_depth: Optional[int] = Query(None, ge=16, le=8192),
    gpu_poll_timeout: Optional[float] = Query(None, ge=0.001, le=1.0),
):
    """Live-tune server parameters without restart."""
    changes = {}
    if max_batch_size is not None:
        batch_engine._max_batch_size = max_batch_size
        changes["max_batch_size"] = max_batch_size
    if max_wait_seconds is not None:
        batch_engine._max_wait_seconds = max_wait_seconds
        changes["max_wait_seconds"] = max_wait_seconds
    if max_queue_depth is not None:
        batch_engine._max_queue_depth = max_queue_depth
        changes["max_queue_depth"] = max_queue_depth
    if gpu_poll_timeout is not None:
        gpu_worker._batch_poll_timeout = gpu_poll_timeout
        changes["gpu_poll_timeout"] = gpu_poll_timeout
    log.info(f"Config updated: {changes}")
    return {"updated": changes}


# --- Batch Transcription (Parakeet TDT) ---


def _max_upload_bytes() -> int:
    return config.get("batcher", {}).get("max_upload_bytes", 100 * 1024 * 1024)


def _save_upload_sync(src_file, suffix: str, max_bytes: int) -> str:
    """Save an upload to a temp file with size enforcement. Runs in threadpool."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        total = 0
        while True:
            chunk = src_file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                tmp.close()
                os.unlink(tmp_path)
                raise ValueError(f"File exceeds {max_bytes} byte limit")
            tmp.write(chunk)
    return tmp_path


def _require_batch_model():
    if not config.get("batch_model", {}).get("name"):
        raise HTTPException(
            status_code=404,
            detail="Batch transcription not available — server running in streaming-only mode",
        )


@app.post("/v1/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    timestamps: bool = Query(False, description="Include word-level timestamps"),
):
    """Transcribe an audio file using Parakeet TDT with dynamic batching."""
    _require_batch_model()
    max_bytes = _max_upload_bytes()
    suffix = Path(file.filename).suffix if file.filename else ".wav"

    loop = asyncio.get_running_loop()
    try:
        tmp_path = await loop.run_in_executor(None, functools.partial(_save_upload_sync, file.file, suffix, max_bytes))
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc))

    try:
        result = await batch_engine.submit(tmp_path, timestamps=timestamps, owns_file=True)
        return JSONResponse(content=result)
    except QueueFullError:
        raise HTTPException(status_code=503, detail="Server overloaded — try again later")


_MAX_BATCH_FILES = 64


@app.post("/v1/transcribe/batch")
async def transcribe_batch(
    files: list[UploadFile] = File(...),
    timestamps: bool = Query(False),
):
    """Transcribe multiple files. All are batched together for GPU efficiency."""
    _require_batch_model()
    if len(files) > _MAX_BATCH_FILES:
        raise HTTPException(status_code=400, detail=f"Too many files (max {_MAX_BATCH_FILES})")

    max_bytes = _max_upload_bytes()
    loop = asyncio.get_running_loop()
    tmp_paths = []
    submitted_paths = set()
    try:
        for f in files:
            suffix = Path(f.filename).suffix if f.filename else ".wav"
            try:
                path = await loop.run_in_executor(
                    None, functools.partial(_save_upload_sync, f.file, suffix, max_bytes)
                )
                tmp_paths.append(path)
            except ValueError:
                tmp_paths.append(None)

        tasks = []
        for i, p in enumerate(tmp_paths):
            if p is None:
                fut = loop.create_future()
                fut.set_exception(ValueError(f"File {files[i].filename} exceeds size limit"))
                tasks.append(fut)
            else:
                tasks.append(batch_engine.submit(p, timestamps=timestamps, owns_file=True))
                submitted_paths.add(p)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                output.append({"error": str(r), "file": files[i].filename})
            else:
                output.append(r)

        return JSONResponse(content={"results": output})
    finally:
        for p in tmp_paths:
            if p is not None and p not in submitted_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass


# --- Streaming Transcription (Nemotron) ---


@app.websocket("/v1/stream")
async def stream_ws(websocket: WebSocket):
    """
    WebSocket streaming ASR endpoint.

    Protocol:
        1. Client connects to /v1/stream
        2. Server sends: {"stream_id": "...", "status": "opened"}
        3. Client sends raw PCM16 audio chunks (binary frames)
        4. Server sends partial transcripts: {"text": "...", "is_final": false}
        5. Client sends text frame: {"action": "close"}
        6. Server sends final transcript and closes
    """
    await websocket.accept()

    try:
        session = await stream_engine.open_stream()
        stream_id = session["stream_id"]
        await websocket.send_json(session)
    except TooManyStreamsError:
        await websocket.send_json({"error": "Too many active streams"})
        await websocket.close(code=1013)
        return

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message:
                result = await stream_engine.process_chunk(stream_id, message["bytes"])
                await websocket.send_json(result)

            elif "text" in message:
                data = json.loads(message["text"])
                if data.get("action") == "close":
                    break

    except WebSocketDisconnect:
        pass
    except (StreamExpiredError, ChunkTooLargeError) as exc:
        await websocket.send_json({"error": str(exc)})
    except Exception as exc:
        log.error(f"Stream {stream_id} error: {exc}")
        await websocket.send_json({"error": str(exc)})
    finally:
        final = await stream_engine.close_stream(stream_id)
        try:
            await websocket.send_json(final)
            await websocket.close()
        except Exception:
            pass


def main():
    import argparse

    parser = argparse.ArgumentParser(description="NeMo ASR serving")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    global config
    config = load_config(args.config)

    server_cfg = config.get("server", {})
    host = args.host or server_cfg.get("host", "0.0.0.0")
    port = args.port or server_cfg.get("port", 8000)

    log.info(f"Starting ASR server on {host}:{port}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=server_cfg.get("workers", 1),
        log_level="info",
    )


if __name__ == "__main__":
    main()
