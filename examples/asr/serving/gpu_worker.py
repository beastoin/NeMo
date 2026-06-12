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
import gc
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import torch

log = logging.getLogger(__name__)

_MAX_GPU_QUEUE = 512


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
        self._batch_queue: queue.Queue[WorkItem] = queue.Queue(maxsize=_MAX_GPU_QUEUE)
        self._stream_queue: queue.Queue[WorkItem] = queue.Queue(maxsize=_MAX_GPU_QUEUE)
        self._thread: Optional[threading.Thread] = None
        self._batch_model = None
        self._stream_pipeline = None
        self._stream_sessions: dict[str, dict] = {}
        self._next_stream_int_id = 1
        self._source_language = "English"
        self._batch_poll_timeout = 0.05
        self._ready = threading.Event()
        self._load_error: Optional[Exception] = None
        self._running = False

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and self._load_error is None

    def start(self, batch_cfg: dict, stream_cfg: dict) -> None:
        self._batch_cfg = batch_cfg
        self._stream_cfg = stream_cfg
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="gpu-worker")
        self._thread.start()

    def wait_ready(self, timeout: float = 600) -> None:
        """Block until models are loaded. Raises if loading failed or timed out."""
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError(f"GPU models did not load within {timeout}s")
        if self._load_error is not None:
            raise self._load_error

    def stop(self) -> None:
        if not self._running:
            return
        dummy_loop = asyncio.new_event_loop()
        fut = dummy_loop.create_future()
        try:
            self._stream_queue.put(WorkItem(WorkType.SHUTDOWN, None, fut, dummy_loop), timeout=5)
        except queue.Full:
            pass
        self._running = False
        self._thread.join(timeout=30)
        dummy_loop.close()

    def submit(self, work_type: WorkType, payload: Any, loop: asyncio.AbstractEventLoop) -> asyncio.Future:
        if not self.is_ready:
            fut = loop.create_future()
            fut.set_exception(RuntimeError("GPU worker not ready"))
            return fut
        fut = loop.create_future()
        q = self._stream_queue if work_type != WorkType.BATCH_TRANSCRIBE else self._batch_queue
        try:
            q.put_nowait(WorkItem(work_type, payload, fut, loop))
        except queue.Full:
            fut.set_exception(RuntimeError("GPU queue full"))
        return fut

    def _run_loop(self) -> None:
        log.info("GPU worker thread started")
        try:
            self._load_models()
            self._ready.set()
        except Exception as exc:
            log.error(f"Model loading failed: {exc}")
            self._load_error = exc
            self._ready.set()
            return

        while self._running:
            item = None
            # Streaming gets priority — drain stream queue before batch
            try:
                item = self._stream_queue.get_nowait()
            except queue.Empty:
                try:
                    item = self._batch_queue.get(timeout=self._batch_poll_timeout)
                except queue.Empty:
                    continue

            if item.work_type == WorkType.SHUTDOWN:
                break

            try:
                result = self._dispatch(item)
                item.loop.call_soon_threadsafe(self._safe_set_result, item.future, result)
            except Exception as exc:
                item.loop.call_soon_threadsafe(self._safe_set_exception, item.future, exc)
            finally:
                # Force GC on the GPU thread so NeMo's internal generators
                # and their CUDA tensors are collected here, not on the
                # async thread where CachingHostAllocator would segfault.
                gc.collect()

        for q in (self._stream_queue, self._batch_queue):
            while not q.empty():
                try:
                    item = q.get_nowait()
                    if item.work_type != WorkType.SHUTDOWN:
                        item.loop.call_soon_threadsafe(
                            item.future.set_exception, RuntimeError("GPU worker shutting down")
                        )
                except queue.Empty:
                    break

        log.info("GPU worker thread stopped")

    @staticmethod
    def _safe_set_result(future: asyncio.Future, result: Any) -> None:
        if not future.done():
            future.set_result(result)

    @staticmethod
    def _safe_set_exception(future: asyncio.Future, exc: Exception) -> None:
        if not future.done():
            future.set_exception(exc)

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

        torch.backends.cudnn.benchmark = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        log.info("Torch optimizations: cudnn.benchmark=True, matmul_precision=high")

        log.info(f"Loading batch model: {self._batch_cfg['name']}")
        self._batch_model = nemo_asr.models.ASRModel.from_pretrained(
            self._batch_cfg["name"], map_location=device
        )
        self._batch_model.eval()

        torch.cuda.empty_cache()

        if self._batch_cfg.get("compile", False):
            log.info("Compiling batch model with torch.compile")
            self._batch_model = torch.compile(self._batch_model)

        # Stream model is loaded by the pipeline builder — no manual load needed
        self._build_stream_pipeline()
        log.info("Models loaded and ready")

    _LATENCY_MODE_TO_CONTEXT = {
        "80ms": [70, 0],
        "160ms": [70, 1],
        "480ms": [70, 6],
        "1040ms": [70, 13],
    }

    def _build_stream_pipeline(self) -> None:
        """Build the streaming pipeline using NeMo's cache_aware_rnnt config."""
        if not self._stream_cfg or not self._stream_cfg.get("name"):
            log.info("No stream model configured, streaming will be unavailable")
            return

        from omegaconf import OmegaConf

        from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

        device = self._stream_cfg.get("device", "cuda:0")
        device_parts = device.split(":")
        device_name = device_parts[0]
        device_id = int(device_parts[1]) if len(device_parts) > 1 else 0

        ref_config_path = os.path.join(
            os.path.dirname(__file__), "..", "conf", "asr_streaming_inference", "cache_aware_rnnt.yaml"
        )
        if not os.path.exists(ref_config_path):
            log.warning(f"Streaming config not found at {ref_config_path}, streaming will be unavailable")
            return

        base_cfg = OmegaConf.load(ref_config_path)
        overrides = OmegaConf.create({
            "asr": {
                "model_name": self._stream_cfg["name"],
                "device": device_name,
                "device_id": device_id,
                "compute_dtype": "float16",
                "use_amp": self._stream_cfg.get("amp", True),
            },
            "enable_itn": False,
            "enable_nmt": False,
        })

        latency_mode = self._stream_cfg.get("latency_mode", "480ms")
        context = self._LATENCY_MODE_TO_CONTEXT.get(latency_mode)
        if context is not None:
            overrides["streaming"] = {"att_context_size": context}
            log.info(f"Streaming latency mode: {latency_mode} (att_context_size={context})")

        source_lang = self._stream_cfg.get("source_language", "English")
        overrides["source_language"] = source_lang
        self._source_language = source_lang

        cfg = OmegaConf.merge(base_cfg, overrides)

        self._stream_pipeline = PipelineBuilder.build_pipeline(cfg)
        # Initialize the session once — per-stream state is managed via
        # init_state() (called by transcribe_step when is_first=True) and
        # delete_state(). Do NOT call open_session() per stream — it resets
        # ALL stream state.
        self._stream_pipeline.open_session()
        log.info("Streaming pipeline built and session opened")

    def _batch_transcribe(self, payload: dict) -> list:
        audio_paths = payload["audio_paths"]
        timestamps = payload.get("timestamps", False)
        batch_size = payload.get("batch_size", 16)

        results = self._batch_model.transcribe(
            audio_paths,
            batch_size=batch_size,
            timestamps=timestamps,
            return_hypotheses=timestamps,
        )
        # Serialize on the GPU thread to prevent CUDA tensor references from
        # crossing thread boundaries.  NeMo result objects hold internal CUDA
        # tensors; if they are GC'd on the async server thread the CUDA host
        # allocator segfaults (signal 139).
        serialized = self._extract_results(results, timestamps)
        del results
        return serialized

    @staticmethod
    def _extract_results(results, timestamps: bool) -> list:
        """Convert NeMo results to plain Python objects on the GPU thread."""
        out = []
        items = results if isinstance(results, list) else [results]
        for r in items:
            if timestamps and hasattr(r, 'text') and hasattr(r, 'timestamp'):
                ts = {}
                if isinstance(r.timestamp, dict):
                    for k, entries in r.timestamp.items():
                        if k == 'timestep':
                            continue
                        ts[k] = [
                            {ek: (round(ev, 4) if isinstance(ev, float) else str(ev) if not isinstance(ev, (int, str)) else ev)
                             for ek, ev in e.items()}
                            for e in entries
                        ]
                out.append({"text": str(r.text), "timestamp": ts})
            elif hasattr(r, 'text'):
                out.append(str(r.text))
            else:
                out.append(str(r))
        return out

    def _stream_open(self, payload: dict) -> dict:
        if self._stream_pipeline is None:
            raise RuntimeError("Streaming pipeline not available")

        stream_id = payload["stream_id"]
        stream_int_id = self._next_stream_int_id
        self._next_stream_int_id += 1

        self._stream_sessions[stream_id] = {
            "int_id": stream_int_id,
            "chunk_index": 0,
            "created_at": time.monotonic(),
        }
        return {"stream_id": stream_id, "status": "opened"}

    def _stream_chunk(self, payload: dict) -> dict:
        from nemo.collections.asr.inference.streaming.framing.request import Frame
        from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions

        stream_id = payload["stream_id"]
        audio_chunk = payload["audio_chunk"]

        session = self._stream_sessions.get(stream_id)
        if session is None:
            raise ValueError(f"Unknown stream: {stream_id}")

        chunk_index = session["chunk_index"]
        is_first = chunk_index == 0
        session["chunk_index"] = chunk_index + 1

        samples = torch.tensor(audio_chunk, dtype=torch.float32)
        options = (
            ASRRequestOptions(
                enable_itn=False,
                enable_nmt=False,
                source_language=self._source_language,
            )
            if is_first
            else None
        )

        frame = Frame(
            samples=samples,
            stream_id=session["int_id"],
            is_first=is_first,
            is_last=False,
            options=options,
        )

        outputs = self._stream_pipeline.transcribe_step([frame])
        output = outputs[0] if outputs else None

        partial = ""
        final = ""
        if output is not None:
            partial = getattr(output, 'partial_transcript', '') or ''
            final = getattr(output, 'final_transcript', '') or ''

        return {
            "stream_id": stream_id,
            "partial_transcript": partial,
            "final_transcript": final,
            "is_final": bool(final),
        }

    def _stream_close(self, payload: dict) -> dict:
        stream_id = payload["stream_id"]
        session = self._stream_sessions.pop(stream_id, None)

        if session is None:
            return {"stream_id": stream_id, "status": "not_found"}

        final_text = ""

        # Only send final frame if chunks were actually processed
        if session["chunk_index"] > 0:
            from nemo.collections.asr.inference.streaming.framing.request import Frame
            from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions

            frame = Frame(
                samples=torch.zeros(1, dtype=torch.float32),
                stream_id=session["int_id"],
                is_first=False,
                is_last=True,
                options=ASRRequestOptions(enable_itn=False, enable_nmt=False),
            )
            outputs = self._stream_pipeline.transcribe_step([frame])
            output = outputs[0] if outputs else None
            if output is not None:
                final_text = getattr(output, 'final_transcript', '') or ''

            self._stream_pipeline.delete_state(session["int_id"])

        return {
            "stream_id": stream_id,
            "final_text": final_text,
            "status": "closed",
        }
