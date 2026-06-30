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
        self._batch_models: list = []
        self._pool_size = 1
        self._pool_threads: list[threading.Thread] = []
        self._pool_queues: list[queue.Queue] = []
        self._next_pool_idx = 0
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_queue: Optional[queue.Queue] = None
        self._stream_pipeline = None
        self._stream_sessions: dict[str, dict] = {}
        self._next_stream_int_id = 1
        self._source_language = "English"
        self._stream_chunk_samples = 5120
        self._batch_poll_timeout = 0.05
        self._ready = threading.Event()
        self._load_error: Optional[Exception] = None
        self._running = False
        self._submit_lock = threading.Lock()
        self._attn_mode = "full"
        self._attn_auto_threshold_sec = 600
        self._attn_local_context = [128, 128]
        self._attn_is_local = False
        self._max_file_duration_sec = 0
        self._raw_batch_model = None
        self._vram_total_mb = 0.0
        self._vram_baseline_mb = 0.0

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and self._load_error is None

    @property
    def vram_info(self) -> dict:
        return {
            "total_mb": self._vram_total_mb,
            "baseline_mb": self._vram_baseline_mb,
            "attention_mode": self._attn_mode,
            "auto_threshold_sec": self._attn_auto_threshold_sec,
        }

    def start(self, batch_cfg: dict, stream_cfg: dict) -> None:
        self._batch_cfg = batch_cfg
        self._stream_cfg = stream_cfg
        self._running = True
        self._gc_interval = batch_cfg.get("gc_interval", 50)
        self._gc_counter = 0
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="gpu-worker")
        self._thread.start()

    def _maybe_gc(self) -> None:
        gc.collect(0)
        self._gc_counter += 1
        if self._gc_counter >= self._gc_interval:
            gc.collect()
            self._gc_counter = 0

    def wait_ready(self, timeout: float = 600) -> None:
        """Block until models are loaded. Raises if loading failed or timed out."""
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError(f"GPU models did not load within {timeout}s")
        if self._load_error is not None:
            raise self._load_error

    def stop(self) -> None:
        with self._submit_lock:
            if not self._running:
                return
            self._running = False
        dummy_loop = asyncio.new_event_loop()
        fut = dummy_loop.create_future()
        try:
            self._stream_queue.put(WorkItem(WorkType.SHUTDOWN, None, fut, dummy_loop), timeout=5)
        except queue.Full:
            pass
        self._thread.join(timeout=30)
        dummy_loop.close()

    def submit(self, work_type: WorkType, payload: Any, loop: asyncio.AbstractEventLoop) -> asyncio.Future:
        with self._submit_lock:
            if not self._running:
                fut = loop.create_future()
                fut.set_exception(RuntimeError("GPU worker shutting down"))
                return fut
            if not self.is_ready:
                fut = loop.create_future()
                fut.set_exception(RuntimeError("GPU worker not ready"))
                return fut
            fut = loop.create_future()
            if work_type == WorkType.BATCH_TRANSCRIBE and self._pool_size > 1:
                idx = self._next_pool_idx % self._pool_size
                self._next_pool_idx += 1
                q = self._pool_queues[idx]
            elif work_type != WorkType.BATCH_TRANSCRIBE:
                q = self._stream_queue
            else:
                q = self._batch_queue
            try:
                q.put_nowait(WorkItem(work_type, payload, fut, loop))
            except queue.Full:
                fut.set_exception(RuntimeError("GPU queue full"))
        return fut

    def _run_loop(self) -> None:
        log.info("GPU worker thread started")
        try:
            self._load_models()
            if self._pool_size > 1:
                self._start_pool_workers()
            self._ready.set()
        except Exception as exc:
            log.error(f"Model loading failed: {exc}")
            self._load_error = exc
            self._ready.set()
            return

        if self._pool_size > 1:
            self._run_pool_mode()
        elif self._batch_cfg.get("prefetch", False):
            self._run_prefetch_mode()
        else:
            self._run_single_mode()

        log.info("GPU worker thread stopped")

    def _run_single_mode(self) -> None:
        log.info("Running in single-model mode (batched streaming)")
        while self._running:
            # Drain all pending stream chunks and batch them together
            stream_items = []
            non_chunk_item = None
            try:
                item = self._stream_queue.get_nowait()
                if item.work_type == WorkType.STREAM_CHUNK:
                    stream_items.append(item)
                else:
                    non_chunk_item = item
            except queue.Empty:
                pass

            max_batch = self._stream_cfg.get("max_batch_size", 64)
            while len(stream_items) < max_batch and non_chunk_item is None:
                try:
                    item = self._stream_queue.get_nowait()
                    if item.work_type == WorkType.STREAM_CHUNK:
                        stream_items.append(item)
                    else:
                        non_chunk_item = item
                        break
                except queue.Empty:
                    break

            # Process batched stream chunks
            if stream_items:
                self._dispatch_stream_batch(stream_items)
                self._maybe_gc()

            # Process non-chunk stream items (open/close/shutdown)
            if non_chunk_item is not None:
                if non_chunk_item.work_type == WorkType.SHUTDOWN:
                    break
                try:
                    result = self._dispatch(non_chunk_item)
                    non_chunk_item.loop.call_soon_threadsafe(self._safe_set_result, non_chunk_item.future, result)
                except Exception as exc:
                    non_chunk_item.loop.call_soon_threadsafe(self._safe_set_exception, non_chunk_item.future, exc)

            # If no stream work, check batch queue
            if not stream_items and non_chunk_item is None:
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
                    self._maybe_gc()

        self._drain_queues([self._stream_queue, self._batch_queue])

    @torch.inference_mode()
    def _dispatch_stream_batch(self, items: list) -> None:
        """Process multiple stream chunks in a single batched GPU call.

        Small WebSocket chunks (e.g. 40ms / 640 samples) are accumulated per
        stream until the pipeline's native chunk size is reached (typically
        320ms / 5120 samples).  This matches the chunk size used by
        ``pipeline.run()`` and is required for prompt-conditioned multilingual
        models to produce non-blank output.
        """
        from nemo.collections.asr.inference.streaming.framing.request import Frame
        from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions

        frames = []
        frame_stream_ids: set[str] = set()
        valid_items = []

        import numpy as np

        chunk_bytes = self._stream_chunk_samples * 4

        for item in items:
            payload = item.payload
            stream_id = payload["stream_id"]
            audio_chunk = payload["audio_chunk"]
            session = self._stream_sessions.get(stream_id)
            if session is None:
                item.loop.call_soon_threadsafe(
                    self._safe_set_exception, item.future, ValueError(f"Unknown stream: {stream_id}")
                )
                continue

            session["chunk_index"] += 1
            session["audio_buffer"].extend(audio_chunk.astype(np.float32).tobytes())
            session["buffer_samples"] += len(audio_chunk)

            if session["buffer_samples"] > self._MAX_BUFFER_SAMPLES:
                excess = session["buffer_samples"] - self._MAX_BUFFER_SAMPLES
                session["audio_buffer"] = session["audio_buffer"][excess * 4 :]
                session["buffer_samples"] = self._MAX_BUFFER_SAMPLES

            valid_items.append(item)

            while session["buffer_samples"] >= self._stream_chunk_samples:
                raw = bytes(session["audio_buffer"][:chunk_bytes])
                session["audio_buffer"] = session["audio_buffer"][chunk_bytes:]
                session["buffer_samples"] -= self._stream_chunk_samples

                is_first = session["frames_sent"] == 0
                session["frames_sent"] += 1

                samples = torch.frombuffer(raw, dtype=torch.float32).clone()
                options = (
                    ASRRequestOptions(
                        enable_itn=False,
                        enable_nmt=False,
                        source_language=self._source_language,
                    )
                    if is_first
                    else None
                )
                frames.append(
                    Frame(
                        samples=samples,
                        stream_id=session["int_id"],
                        is_first=is_first,
                        is_last=False,
                        options=options,
                    )
                )
                frame_stream_ids.add(stream_id)

        if not valid_items:
            return

        output_by_int_id = {}
        if frames:
            if len(frames) > 1:
                log.debug(f"Batched {len(frames)} stream frames")
            remaining = list(frames)
            while remaining:
                batch = []
                seen = set()
                deferred = []
                for f in remaining:
                    if f.stream_id not in seen:
                        batch.append(f)
                        seen.add(f.stream_id)
                    else:
                        deferred.append(f)
                outputs = self._stream_pipeline.transcribe_step(batch)
                for out in outputs:
                    output_by_int_id.setdefault(out.stream_id, []).append(out)
                remaining = deferred

        for item in valid_items:
            stream_id = item.payload["stream_id"]
            session = self._stream_sessions.get(stream_id)

            if stream_id in frame_stream_ids and session is not None:
                step_outputs = output_by_int_id.pop(session["int_id"], [])
                final = ""
                partial = ""
                for out in step_outputs:
                    if out.final_transcript:
                        final = (final + out.final_transcript).strip()
                    if out.partial_transcript:
                        partial = out.partial_transcript
                if final:
                    session["committed_text"] += " " + final
                if partial:
                    session["last_partial"] = partial
                result = {
                    "stream_id": stream_id,
                    "partial_transcript": partial,
                    "final_transcript": final,
                    "is_final": bool(final),
                }
            else:
                result = {
                    "stream_id": stream_id,
                    "partial_transcript": "",
                    "final_transcript": "",
                    "is_final": False,
                }
            item.loop.call_soon_threadsafe(self._safe_set_result, item.future, result)

    def _start_pool_workers(self) -> None:
        log.info(f"Starting pool mode: {self._pool_size} model workers")
        for i in range(self._pool_size):
            q = queue.Queue(maxsize=_MAX_GPU_QUEUE)
            self._pool_queues.append(q)
            t = threading.Thread(
                target=self._pool_worker_loop,
                args=(i, self._batch_models[i], q),
                daemon=True,
                name=f"gpu-pool-{i}",
            )
            self._pool_threads.append(t)
            t.start()

    def _run_pool_mode(self) -> None:
        while self._running:
            try:
                item = self._stream_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item.work_type == WorkType.SHUTDOWN:
                break
            try:
                result = self._dispatch(item)
                item.loop.call_soon_threadsafe(self._safe_set_result, item.future, result)
            except Exception as exc:
                item.loop.call_soon_threadsafe(self._safe_set_exception, item.future, exc)

        for q in self._pool_queues:
            dummy_loop = asyncio.new_event_loop()
            try:
                q.put(WorkItem(WorkType.SHUTDOWN, None, dummy_loop.create_future(), dummy_loop), timeout=5)
            except queue.Full:
                pass
            dummy_loop.close()
        for t in self._pool_threads:
            t.join(timeout=30)

        self._drain_queues(self._pool_queues + [self._stream_queue, self._batch_queue])

    @torch.inference_mode()
    def _pool_worker_loop(self, idx: int, model, q: queue.Queue) -> None:
        stream = torch.cuda.Stream()
        gc_counter = 0
        log.info(f"Pool worker {idx} started (stream={stream})")
        while self._running:
            try:
                item = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item.work_type == WorkType.SHUTDOWN:
                break
            try:
                with torch.cuda.stream(stream):
                    result = self._batch_transcribe_with_model(model, item.payload)
                item.loop.call_soon_threadsafe(self._safe_set_result, item.future, result)
            except Exception as exc:
                item.loop.call_soon_threadsafe(self._safe_set_exception, item.future, exc)
            finally:
                gc.collect(0)
                gc_counter += 1
                if gc_counter >= self._gc_interval:
                    gc.collect()
                    gc_counter = 0
        log.info(f"Pool worker {idx} stopped")

    def _run_prefetch_mode(self) -> None:
        log.info("Running in prefetch mode (tensor bypass)")
        self._prefetch_queue = queue.Queue(maxsize=4)
        self._prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True, name="prefetch")
        self._prefetch_thread.start()

        while self._running:
            item = None
            try:
                item = self._stream_queue.get_nowait()
            except queue.Empty:
                try:
                    item = self._prefetch_queue.get(timeout=self._batch_poll_timeout)
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
                self._maybe_gc()

        drain_list = [self._stream_queue, self._batch_queue]
        if self._prefetch_queue is not None:
            drain_list.append(self._prefetch_queue)
        self._drain_queues(drain_list)
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=10)
            self._drain_queues([self._prefetch_queue])

    def _prefetch_loop(self) -> None:
        import numpy as np
        import soundfile as sf

        log.info("Prefetch thread started")
        owned_item = None
        try:
            while self._running:
                try:
                    owned_item = self._batch_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if owned_item.work_type == WorkType.SHUTDOWN:
                    self._prefetch_queue.put(owned_item)
                    owned_item = None
                    break

                if owned_item.work_type == WorkType.BATCH_TRANSCRIBE:
                    try:
                        audio_arrays = []
                        for path in owned_item.payload["audio_paths"]:
                            data, sr = sf.read(path, dtype='float32')
                            if sr != 16000:
                                import librosa

                                data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                            audio_arrays.append(np.array(data, dtype=np.float32))
                        owned_item.payload["audio_tensors"] = audio_arrays
                    except Exception as exc:
                        log.warning(f"Prefetch failed, falling back to paths: {exc}")

                while self._running:
                    try:
                        self._prefetch_queue.put(owned_item, timeout=0.5)
                        owned_item = None
                        break
                    except queue.Full:
                        continue
        finally:
            if owned_item is not None and owned_item.work_type != WorkType.SHUTDOWN:
                owned_item.loop.call_soon_threadsafe(
                    self._safe_set_exception, owned_item.future, RuntimeError("GPU worker shutting down")
                )
        log.info("Prefetch thread stopped")

    def _drain_queues(self, queues) -> None:
        for q in queues:
            while not q.empty():
                try:
                    item = q.get_nowait()
                    if item.work_type != WorkType.SHUTDOWN:
                        item.loop.call_soon_threadsafe(
                            self._safe_set_exception, item.future, RuntimeError("GPU worker shutting down")
                        )
                except queue.Empty:
                    break

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

    def _load_one_model(self, nemo_asr, device, idx=0):
        tag = f" (pool #{idx})" if self._pool_size > 1 else ""
        log.info(f"Loading batch model{tag}: {self._batch_cfg['name']}")
        model = nemo_asr.models.ASRModel.from_pretrained(self._batch_cfg["name"], map_location=device)
        model.eval()

        self._attn_mode = self._batch_cfg.get("attention_mode", "full")
        self._attn_local_context = self._batch_cfg.get("local_attn_context", [128, 128])
        self._attn_auto_threshold_sec = self._batch_cfg.get("auto_local_attn_threshold_sec", 600)
        self._max_file_duration_sec = self._batch_cfg.get("max_file_duration_sec", 0)

        if self._attn_mode == "auto" and self._pool_size > 1:
            log.warning(
                f"Auto attention mode is unsafe with model_pool_size={self._pool_size} "
                f"(shared mutable state). Falling back to full attention mode."
            )
            self._attn_mode = "full"

        if self._attn_mode == "local":
            model.change_attention_model("rel_pos_local_attn", self._attn_local_context)
            model.change_subsampling_conv_chunking_factor(1)
            self._attn_is_local = True
            log.info(f"Attention mode: local{tag} (context={self._attn_local_context}) — linear VRAM scaling")
        elif self._attn_mode == "auto":
            log.info(
                f"Attention mode: auto{tag} — full for <{self._attn_auto_threshold_sec}s, "
                f"local for >={self._attn_auto_threshold_sec}s (torch.compile disabled for auto mode)"
            )
            self._raw_batch_model = model
        else:
            log.info(f"Attention mode: full{tag} (default)")

        if self._max_file_duration_sec > 0:
            log.info(f"Max file duration: {self._max_file_duration_sec}s")

        if not self._batch_cfg.get("cuda_graphs", True):
            if hasattr(model, 'decoding') and hasattr(model.decoding, 'decoding'):
                disabled = model.decoding.decoding.disable_cuda_graphs()
                log.info(f"CUDA graph decoding disabled{tag} (was active: {disabled})")
        if self._batch_cfg.get("compile", False) and self._attn_mode != "auto":
            log.info(f"Compiling batch model{tag} with torch.compile")
            model = torch.compile(model)
        elif self._attn_mode == "auto" and self._batch_cfg.get("compile", False):
            log.info(f"Skipping torch.compile{tag} — incompatible with auto attention switching")
        return model

    def _load_models(self) -> None:
        import nemo.collections.asr as nemo_asr

        torch.backends.cudnn.benchmark = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        log.info("Torch optimizations: cudnn.benchmark=True, matmul_precision=high")

        if self._batch_cfg.get("name"):
            device = self._batch_cfg.get("device", "cuda:0")
            self._pool_size = self._batch_cfg.get("model_pool_size", 1)

            if self._pool_size > 1:
                log.info(f"Loading model pool: {self._pool_size} instances")
                for i in range(self._pool_size):
                    model = self._load_one_model(nemo_asr, device, i)
                    self._batch_models.append(model)
                    torch.cuda.empty_cache()
                self._batch_model = self._batch_models[0]
            else:
                self._batch_model = self._load_one_model(nemo_asr, device)

            torch.cuda.empty_cache()
        else:
            log.info("No batch model configured, batch transcription will be unavailable")

        self._build_stream_pipeline()

        device = self._batch_cfg.get("device", "cuda:0") if self._batch_cfg.get("name") else "cuda:0"
        dev_idx = int(device.split(":")[-1]) if ":" in device else 0
        free_bytes, total_bytes = torch.cuda.mem_get_info(dev_idx)
        self._vram_total_mb = total_bytes / (1024 * 1024)
        self._vram_baseline_mb = (total_bytes - free_bytes) / (1024 * 1024)
        log.info(
            f"VRAM after all models: {self._vram_baseline_mb:.0f}MiB used / "
            f"{self._vram_total_mb:.0f}MiB total ({free_bytes / (1024 * 1024):.0f}MiB free)"
        )

        if self._batch_model is None and self._stream_pipeline is None:
            raise RuntimeError("No models loaded — configure batch_model and/or stream_model")

        log.info("Models loaded and ready")

    _LATENCY_MODE_TO_RIGHT_CONTEXT = {
        "80ms": 0,
        "160ms": 1,
        "480ms": 6,
        "1040ms": 13,
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
            raise FileNotFoundError(
                f"Streaming config not found at {ref_config_path}. "
                "Copy examples/asr/conf/asr_streaming_inference/cache_aware_rnnt.yaml "
                "to the expected path relative to the serving directory."
            )

        base_cfg = OmegaConf.load(ref_config_path)
        overrides = OmegaConf.create(
            {
                "asr": {
                    "model_name": self._stream_cfg["name"],
                    "device": device_name,
                    "device_id": device_id,
                    "compute_dtype": "float16",
                    "use_amp": self._stream_cfg.get("amp", True),
                },
                "enable_itn": False,
                "enable_nmt": False,
            }
        )

        source_lang = self._stream_cfg.get("source_language", "English")
        overrides["source_language"] = source_lang
        self._source_language = source_lang

        overrides["streaming"] = {"att_context_size": None}
        cfg = OmegaConf.merge(base_cfg, overrides)

        self._stream_pipeline = PipelineBuilder.build_pipeline(cfg)

        try:
            self._stream_chunk_samples = int(
                self._stream_pipeline.chunk_size_in_secs * self._stream_pipeline.sample_rate
            )
        except AttributeError:
            self._stream_chunk_samples = int(self._stream_cfg.get("chunk_samples", 5120))
        log.info(
            f"Stream chunk target: {self._stream_chunk_samples} samples "
            f"({self._stream_chunk_samples / 16000 * 1000:.0f}ms)"
        )

        latency_mode = self._stream_cfg.get("latency_mode", "480ms")
        right_ctx = self._LATENCY_MODE_TO_RIGHT_CONTEXT.get(latency_mode)
        if right_ctx is not None:
            left_ctx = self._stream_pipeline.asr_model.get_att_context_size()[0]
            att_context = [left_ctx, right_ctx]
            self._stream_pipeline.asr_model.set_default_att_context_size(att_context)
            log.info(f"Streaming latency mode: {latency_mode} (att_context_size={att_context})")

        # Initialize the session once — per-stream state is managed via
        # init_state() (called by transcribe_step when is_first=True) and
        # delete_state(). Do NOT call open_session() per stream — it resets
        # ALL stream state.
        self._stream_pipeline.open_session()
        log.info("Streaming pipeline built and session opened")

    def _batch_transcribe(self, payload: dict) -> list:
        if self._batch_model is None:
            raise RuntimeError("Batch model not loaded — server started in streaming-only mode")
        return self._batch_transcribe_with_model(self._batch_model, payload)

    def _get_audio_duration_sec(self, path: str) -> float:
        try:
            import torchaudio

            info = torchaudio.info(path)
            return info.num_frames / info.sample_rate
        except Exception as exc:
            try:
                import wave

                with wave.open(path) as wf:
                    return wf.getnframes() / wf.getframerate()
            except Exception:
                log.warning(f"Cannot determine audio duration for {path}: {exc}")
                return 0.0

    def _switch_attention(self, to_local: bool) -> None:
        if to_local == self._attn_is_local:
            return
        model = self._raw_batch_model
        if model is None:
            return
        if to_local:
            model.change_attention_model("rel_pos_local_attn", self._attn_local_context)
            model.change_subsampling_conv_chunking_factor(1)
            self._attn_is_local = True
        else:
            model.change_attention_model("rel_pos")
            self._attn_is_local = False

    def _batch_transcribe_with_model(self, model, payload: dict) -> list:
        timestamps = payload.get("timestamps", False)
        batch_size = payload.get("batch_size", 16)

        audio_input = payload.get("audio_tensors", payload["audio_paths"])

        if self._max_file_duration_sec > 0 and isinstance(audio_input, list):
            for path in audio_input:
                if isinstance(path, str):
                    dur = self._get_audio_duration_sec(path)
                    if dur > self._max_file_duration_sec:
                        raise RuntimeError(
                            f"Audio file {dur:.0f}s exceeds max_file_duration_sec "
                            f"({self._max_file_duration_sec}s). Use shorter files or "
                            f"set attention_mode: local/auto for longer audio."
                        )

        if self._attn_mode == "auto":
            durations_from_batcher = payload.get("durations")
            if durations_from_batcher:
                max_dur = max(durations_from_batcher)
            elif isinstance(audio_input, list):
                max_dur = max(
                    (self._get_audio_duration_sec(p) for p in audio_input if isinstance(p, str)),
                    default=0.0,
                )
            else:
                max_dur = 0.0
            need_local = max_dur >= self._attn_auto_threshold_sec
            if need_local != self._attn_is_local:
                mode_name = "local" if need_local else "full"
                log.info(f"Auto-switching attention to {mode_name} (longest file: {max_dur:.0f}s)")
                self._switch_attention(need_local)

        results = model.transcribe(
            audio_input,
            batch_size=batch_size,
            timestamps=timestamps,
            return_hypotheses=timestamps,
            num_workers=0,
            verbose=False,
        )
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
                            {
                                ek: (
                                    round(ev, 4)
                                    if isinstance(ev, float)
                                    else str(ev) if not isinstance(ev, (int, str)) else ev
                                )
                                for ek, ev in e.items()
                            }
                            for e in entries
                        ]
                out.append({"text": str(r.text), "timestamp": ts})
            elif hasattr(r, 'text'):
                out.append(str(r.text))
            else:
                out.append(str(r))
        return out

    _MAX_BUFFER_SAMPLES = 5120 * 10

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
            "committed_text": "",
            "last_partial": "",
            "audio_buffer": bytearray(),
            "buffer_samples": 0,
            "frames_sent": 0,
        }
        return {"stream_id": stream_id, "status": "opened"}

    def _stream_chunk(self, payload: dict) -> dict:
        import numpy as np

        from nemo.collections.asr.inference.streaming.framing.request import Frame
        from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions

        stream_id = payload["stream_id"]
        audio_chunk = payload["audio_chunk"]

        session = self._stream_sessions.get(stream_id)
        if session is None:
            raise ValueError(f"Unknown stream: {stream_id}")

        session["chunk_index"] += 1
        session["audio_buffer"].extend(audio_chunk.astype(np.float32).tobytes())
        session["buffer_samples"] += len(audio_chunk)

        if session["buffer_samples"] > self._MAX_BUFFER_SAMPLES:
            excess = session["buffer_samples"] - self._MAX_BUFFER_SAMPLES
            session["audio_buffer"] = session["audio_buffer"][excess * 4 :]
            session["buffer_samples"] = self._MAX_BUFFER_SAMPLES

        partial = ""
        final = ""
        chunk_bytes = self._stream_chunk_samples * 4

        while session["buffer_samples"] >= self._stream_chunk_samples:
            raw = bytes(session["audio_buffer"][:chunk_bytes])
            session["audio_buffer"] = session["audio_buffer"][chunk_bytes:]
            session["buffer_samples"] -= self._stream_chunk_samples

            is_first = session["frames_sent"] == 0
            session["frames_sent"] += 1

            samples = torch.frombuffer(raw, dtype=torch.float32).clone()
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
            if outputs:
                out = outputs[0]
                partial = out.partial_transcript or ''
                step_final = out.final_transcript or ''
                if step_final:
                    final = (final + " " + step_final).strip() if final else step_final
                    session["committed_text"] += " " + step_final
                if partial:
                    session["last_partial"] = partial

        return {
            "stream_id": stream_id,
            "partial_transcript": partial,
            "final_transcript": final,
            "is_final": bool(final),
        }

    def _stream_close(self, payload: dict) -> dict:
        from nemo.collections.asr.inference.streaming.framing.request import Frame
        from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions

        stream_id = payload["stream_id"]
        session = self._stream_sessions.pop(stream_id, None)

        if session is None:
            return {"stream_id": stream_id, "status": "not_found"}

        final_text = session.get("committed_text", "").strip()
        last_partial = session.get("last_partial", "").strip()

        # Flush any remaining buffered audio
        if session["buffer_samples"] > 0:
            raw = bytes(session["audio_buffer"][: session["buffer_samples"] * 4])
            is_first = session["frames_sent"] == 0
            samples = torch.frombuffer(raw, dtype=torch.float32).clone()
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
            session["frames_sent"] += 1
            if outputs:
                flushed = outputs[0].final_transcript or ''
                if flushed:
                    final_text = (final_text + " " + flushed).strip()

        if session["frames_sent"] > 0:
            # is_last=True triggers forced finalization with keep_all_outputs=True;
            # pipeline creates TranscribeStepOutput before cleanup and state deletion
            frame = Frame(
                samples=torch.zeros(1, dtype=torch.float32),
                stream_id=session["int_id"],
                is_first=False,
                is_last=True,
                options=ASRRequestOptions(enable_itn=False, enable_nmt=False),
            )
            outputs = self._stream_pipeline.transcribe_step([frame])
            if outputs:
                remaining_final = outputs[0].final_transcript or ''
                remaining_partial = outputs[0].partial_transcript or ''
                if remaining_final:
                    final_text = (final_text + " " + remaining_final).strip()
                elif remaining_partial and not final_text:
                    final_text = remaining_partial.strip()

        if not final_text and last_partial:
            final_text = last_partial

        return {
            "stream_id": stream_id,
            "final_text": final_text,
            "status": "closed",
        }
