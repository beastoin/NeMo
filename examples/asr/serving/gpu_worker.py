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
Dedicated GPU inference thread.

All model operations run on a single thread to avoid CUDA context contention
and GIL-related performance issues. The async server communicates with this
thread via a work queue.
"""

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import torch

log = logging.getLogger(__name__)


class WorkType(Enum):
    BATCH_TRANSCRIBE = "batch_transcribe"
    STREAM_CHUNK = "stream_chunk"
    STREAM_OPEN = "stream_open"
    STREAM_CLOSE = "stream_close"
    SHUTDOWN = "shutdown"


@dataclass
class WorkItem:
    work_type: WorkType
    payload: Any
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop
    created_at: float = field(default_factory=time.monotonic)


class GPUWorker:
    """Runs inference on a dedicated thread. Async callers submit WorkItems."""

    def __init__(self):
        self._queue: queue.Queue[WorkItem] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._batch_model = None
        self._stream_model = None
        self._stream_pipeline = None
        self._running = False

    def start(self, batch_cfg: dict, stream_cfg: dict) -> None:
        self._batch_cfg = batch_cfg
        self._stream_cfg = stream_cfg
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="gpu-worker")
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        dummy_loop = asyncio.new_event_loop()
        fut = dummy_loop.create_future()
        self._queue.put(WorkItem(WorkType.SHUTDOWN, None, fut, dummy_loop))
        self._thread.join(timeout=30)
        dummy_loop.close()

    def submit(self, work_type: WorkType, payload: Any, loop: asyncio.AbstractEventLoop) -> asyncio.Future:
        fut = loop.create_future()
        self._queue.put(WorkItem(work_type, payload, fut, loop))
        return fut

    def _run_loop(self) -> None:
        log.info("GPU worker thread started")
        self._load_models()

        while self._running:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if item.work_type == WorkType.SHUTDOWN:
                break

            try:
                result = self._dispatch(item)
                item.loop.call_soon_threadsafe(item.future.set_result, result)
            except Exception as exc:
                item.loop.call_soon_threadsafe(item.future.set_exception, exc)

        log.info("GPU worker thread stopped")

    @torch.inference_mode()
    def _dispatch(self, item: WorkItem) -> Any:
        if item.work_type == WorkType.BATCH_TRANSCRIBE:
            return self._batch_transcribe(item.payload)
        elif item.work_type == WorkType.STREAM_OPEN:
            return self._stream_open(item.payload)
        elif item.work_type == WorkType.STREAM_CHUNK:
            return self._stream_chunk(item.payload)
        elif item.work_type == WorkType.STREAM_CLOSE:
            return self._stream_close(item.payload)
        raise ValueError(f"Unknown work type: {item.work_type}")

    def _load_models(self) -> None:
        import nemo.collections.asr as nemo_asr

        device = self._batch_cfg.get("device", "cuda:0")

        log.info(f"Loading batch model: {self._batch_cfg['name']}")
        self._batch_model = nemo_asr.models.ASRModel.from_pretrained(
            self._batch_cfg["name"], map_location=device
        )
        self._batch_model.eval()
        if self._batch_cfg.get("compile", False):
            log.info("Compiling batch model with torch.compile")
            self._batch_model = torch.compile(self._batch_model)

        log.info(f"Loading stream model: {self._stream_cfg['name']}")
        self._stream_model = nemo_asr.models.ASRModel.from_pretrained(
            self._stream_cfg["name"], map_location=device
        )
        self._stream_model.eval()

        log.info("Models loaded")

    def _batch_transcribe(self, payload: dict) -> list:
        audio_paths = payload["audio_paths"]
        timestamps = payload.get("timestamps", False)
        batch_size = payload.get("batch_size", 16)

        results = self._batch_model.transcribe(
            audio_paths,
            batch_size=batch_size,
            timestamps=timestamps,
        )
        return results

    def _stream_open(self, payload: dict) -> dict:
        from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

        stream_id = payload["stream_id"]
        latency_mode = payload.get("latency_mode", self._stream_cfg.get("latency_mode", "480ms"))

        if not hasattr(self, '_stream_sessions'):
            self._stream_sessions = {}

        pipeline = PipelineBuilder.from_pretrained(
            model_name=self._stream_cfg["name"],
        )
        pipeline.open_session()
        self._stream_sessions[stream_id] = {
            "pipeline": pipeline,
            "created_at": time.monotonic(),
        }
        return {"stream_id": stream_id, "status": "opened"}

    def _stream_chunk(self, payload: dict) -> dict:
        stream_id = payload["stream_id"]
        audio_chunk = payload["audio_chunk"]

        session = self._stream_sessions.get(stream_id)
        if session is None:
            raise ValueError(f"Unknown stream: {stream_id}")

        pipeline = session["pipeline"]
        result = pipeline.transcribe_step(audio_chunk)

        return {
            "stream_id": stream_id,
            "text": result.text if hasattr(result, 'text') else str(result),
            "is_final": result.is_final if hasattr(result, 'is_final') else False,
        }

    def _stream_close(self, payload: dict) -> dict:
        stream_id = payload["stream_id"]
        session = self._stream_sessions.pop(stream_id, None)
        if session is not None:
            pipeline = session["pipeline"]
            final = pipeline.close_session()
            return {
                "stream_id": stream_id,
                "final_text": final.text if hasattr(final, 'text') else str(final),
                "status": "closed",
            }
        return {"stream_id": stream_id, "status": "not_found"}
