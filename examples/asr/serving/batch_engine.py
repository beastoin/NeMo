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
Dynamic batching engine for offline ASR (Parakeet TDT).

Collects incoming transcription requests and groups them into GPU-efficient
batches. This is the key to high throughput: instead of processing one file
per request, we wait a short time (max_wait_seconds) or until the batch is
full (max_batch_size), then send the whole batch to the GPU worker.

Architecture:
    Client -> REST API -> BatchEngine.submit() -> accumulator queue
                                                      |
    GPU Worker <- flush_batch() (triggered by timer or full batch)
                                                      |
    Client <- asyncio.Future resolved with per-request result
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from gpu_worker import GPUWorker, WorkType

log = logging.getLogger(__name__)


@dataclass
class PendingRequest:
    audio_path: str
    timestamps: bool
    future: asyncio.Future
    owns_file: bool = False
    submitted_at: float = field(default_factory=time.monotonic)
    duration_sec: Optional[float] = None


class BatchEngine:
    """Dynamic batcher that accumulates requests and flushes to GPU."""

    def __init__(
        self,
        gpu_worker: GPUWorker,
        max_batch_size: int = 32,
        max_wait_seconds: float = 0.002,
        max_queue_depth: int = 4096,
        vram_safety_factor: float = 0.8,
        vram_bytes_per_t2: float = 136.6,
        starvation_timeout_sec: float = 5.0,
        max_inflight: int = 2,
        **kwargs,
    ):
        self._gpu_worker = gpu_worker
        self._max_batch_size = max_batch_size
        self._max_wait_seconds = max_wait_seconds
        self._max_queue_depth = max_queue_depth
        self._vram_safety_factor = vram_safety_factor
        self._vram_bytes_per_t2 = vram_bytes_per_t2
        self._starvation_timeout = starvation_timeout_sec
        self._max_inflight = max_inflight
        self._vram_available_mb = 0.0
        self._vram_enabled = False
        self._attention_mode = "full"
        self._auto_threshold_sec = 600.0
        self._pool_size = 1
        self._pending: list[PendingRequest] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._flush_pending = False
        self._inflight_sem: Optional[asyncio.Semaphore] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutting_down = False
        self._metrics = {
            "total_requests": 0,
            "total_batches": 0,
            "total_files": 0,
            "rejected_requests": 0,
            "vram_limited_batches": 0,
        }

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._inflight_sem = asyncio.Semaphore(self._max_inflight)
        vram = self._gpu_worker.vram_info
        self._attention_mode = vram.get("attention_mode", "full")
        self._auto_threshold_sec = vram.get("auto_threshold_sec", 600.0)
        self._pool_size = max(1, self._gpu_worker._pool_size) if hasattr(self._gpu_worker, '_pool_size') else 1
        if self._attention_mode == "auto" and self._pool_size > 1:
            log.warning(
                "Auto attention mode is unsafe with model_pool_size > 1 (shared mutable state). "
                "Falling back to full attention mode for VRAM estimation."
            )
            self._attention_mode = "full"
        if self._vram_safety_factor > 0 and vram["total_mb"] > 0:
            budget = vram["total_mb"] * self._vram_safety_factor - vram["baseline_mb"]
            self._vram_available_mb = max(budget, 0) / self._pool_size
            self._vram_enabled = True
            if budget <= 0:
                log.warning(
                    f"VRAM budget is non-positive ({budget:.0f} MB) — baseline exceeds safety cap. "
                    f"All batches will be capped to 1."
                )
            log.info(
                f"VRAM-aware batching enabled: {self._vram_available_mb:.0f} MB budget/worker "
                f"(total={vram['total_mb']:.0f}, baseline={vram['baseline_mb']:.0f}, "
                f"safety={self._vram_safety_factor}, pool={self._pool_size}, coeff={self._vram_bytes_per_t2})"
            )
        else:
            log.info("VRAM-aware batching disabled (safety_factor=0 or no VRAM info)")
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        self._shutting_down = True
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Drain remaining requests
        while self._pending:
            await self._flush_batch()
        # Wait for all inflight batches to complete
        if self._inflight_sem:
            for _ in range(self._max_inflight):
                await self._inflight_sem.acquire()

    @staticmethod
    def _get_audio_duration(path: str) -> Optional[float]:
        try:
            import wave

            with wave.open(path) as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            pass
        try:
            import soundfile as sf

            info = sf.info(path)
            return info.duration
        except Exception:
            pass
        return None

    def _estimate_max_batch(self, max_duration_sec: float, duration_known: bool = True) -> int:
        if not self._vram_enabled or max_duration_sec <= 0:
            return self._max_batch_size
        if self._vram_available_mb <= 0:
            return 1
        if self._attention_mode == "local":
            return self._max_batch_size
        if self._attention_mode == "auto" and duration_known and max_duration_sec >= self._auto_threshold_sec:
            return self._max_batch_size
        T = max_duration_sec / 0.08
        per_file_mb = self._vram_bytes_per_t2 * T * T / (1024 * 1024)
        if per_file_mb <= 0:
            return self._max_batch_size
        return max(1, min(self._max_batch_size, int(self._vram_available_mb / per_file_mb)))

    def _effective_duration(self, req: PendingRequest) -> float:
        if req.duration_sec is not None:
            return req.duration_sec
        return self._auto_threshold_sec

    async def submit(self, audio_path: str, timestamps: bool = False, owns_file: bool = False) -> dict:
        """Submit a file for transcription. Returns when the batch completes.

        If owns_file=True, the engine takes ownership and deletes the file
        after GPU processing completes (even on error or cancellation).
        """
        enqueued = False
        duration = self._get_audio_duration(audio_path)
        try:
            async with self._lock:
                if len(self._pending) >= self._max_queue_depth:
                    self._metrics["rejected_requests"] += 1
                    raise QueueFullError(f"Queue depth {len(self._pending)} exceeds limit {self._max_queue_depth}")

                future = self._loop.create_future()
                self._pending.append(
                    PendingRequest(
                        audio_path=audio_path,
                        timestamps=timestamps,
                        future=future,
                        owns_file=owns_file,
                        duration_sec=duration,
                    )
                )
                enqueued = True
                self._metrics["total_requests"] += 1

                pending_count = len(self._pending)
                vram_limit = self._estimate_max_batch(
                    max(self._effective_duration(r) for r in self._pending),
                    duration_known=all(r.duration_sec is not None for r in self._pending),
                ) if self._vram_enabled and self._pending else self._max_batch_size
                if pending_count >= min(self._max_batch_size, vram_limit) and not self._flush_pending:
                    self._flush_pending = True
                    asyncio.create_task(self._guarded_flush())
        except BaseException:
            if owns_file and not enqueued:
                self._unlink_safe(audio_path)
            raise

        return await future

    async def _flush_loop(self) -> None:
        """Periodically flush partial batches.

        Batches are dispatched as fire-and-forget tasks capped by _inflight_sem
        so the next batch can form while the GPU processes the current one.
        Only one flush task is scheduled at a time to prevent unbounded task
        accumulation behind a saturated semaphore.
        """
        while not self._shutting_down:
            await asyncio.sleep(self._max_wait_seconds)
            if self._pending and not self._flush_pending:
                self._flush_pending = True
                asyncio.create_task(self._guarded_flush())

    def _batch_limit_for(self, req: PendingRequest) -> int:
        dur = self._effective_duration(req)
        return self._estimate_max_batch(dur, duration_known=req.duration_sec is not None)

    def _form_vram_safe_batch(self, candidates: list[PendingRequest]) -> list[PendingRequest]:
        if not candidates or not self._vram_enabled:
            return candidates[: self._max_batch_size]

        now = time.monotonic()
        starved = [r for r in candidates if now - r.submitted_at > self._starvation_timeout]

        if starved:
            anchor = min(starved, key=lambda r: r.submitted_at)
            anchor_dur = self._effective_duration(anchor)
            any_unknown = anchor.duration_sec is None
            limit = self._estimate_max_batch(anchor_dur, duration_known=not any_unknown)
            others = sorted(
                [r for r in candidates if r is not anchor],
                key=lambda r: self._effective_duration(r),
            )
            batch = [anchor]
            for req in others:
                if len(batch) >= limit:
                    break
                candidate_max_dur = max(anchor_dur, self._effective_duration(req))
                has_unknown = any_unknown or req.duration_sec is None
                new_limit = self._estimate_max_batch(candidate_max_dur, duration_known=not has_unknown)
                if len(batch) + 1 <= new_limit:
                    batch.append(req)
                    any_unknown = has_unknown
            return batch

        sorted_candidates = sorted(candidates, key=lambda r: self._effective_duration(r))
        n = min(self._max_batch_size, len(sorted_candidates))
        while n > 1:
            longest = sorted_candidates[n - 1]
            longest_dur = self._effective_duration(longest)
            has_unknown = any(r.duration_sec is None for r in sorted_candidates[:n])
            limit = self._estimate_max_batch(longest_dur, duration_known=not has_unknown)
            if n <= limit:
                break
            n -= 1
        return sorted_candidates[:n]

    async def _guarded_flush(self) -> None:
        try:
            await self._flush_batch()
        finally:
            self._flush_pending = False

    async def _flush_batch(self) -> None:
        await self._inflight_sem.acquire()
        try:
            async with self._lock:
                if not self._pending:
                    return

                batch = self._form_vram_safe_batch(self._pending)
                taken = set(id(r) for r in batch)
                self._pending = [r for r in self._pending if id(r) not in taken]

            # Allow new flush to be scheduled now that we've dequeued
            self._flush_pending = False

            if not batch:
                return

            actual = len(batch)
            if actual < self._max_batch_size and self._vram_enabled:
                self._metrics["vram_limited_batches"] += 1

            self._metrics["total_batches"] += 1
            self._metrics["total_files"] += actual

            any_ts = any(r.timestamps for r in batch)
            durations = [self._effective_duration(r) for r in batch]
            max_dur = max(durations) if durations else 0
            log.info(
                f"Flushing batch: {actual} files "
                f"(max_dur={max_dur:.1f}s, limit={self._estimate_max_batch(max_dur)})"
            )

            await self._run_sub_batch(batch, any_ts)
        finally:
            self._inflight_sem.release()

    @staticmethod
    def _serialize_result(result: Any, audio_path: str, timestamps: bool) -> dict:
        """Convert a NeMo transcription result to a JSON-serializable dict."""
        output = {"audio_path": audio_path}
        if isinstance(result, dict) and "text" in result:
            output.update(result)
            if not timestamps:
                output.pop("timestamp", None)
            return output
        elif hasattr(result, 'text'):
            output["text"] = result.text
        else:
            output["text"] = str(result)

        if timestamps and hasattr(result, 'timestamp') and isinstance(result.timestamp, dict):
            ts = dict(result.timestamp)
            ts.pop('timestep', None)
            for key, entries in ts.items():
                serialized = []
                for entry in entries:
                    item = {}
                    for k, v in entry.items():
                        if isinstance(v, float):
                            item[k] = round(v, 4)
                        elif isinstance(v, int):
                            item[k] = v
                        else:
                            item[k] = str(v)
                    serialized.append(item)
                output[key] = serialized
        return output

    @staticmethod
    def _unlink_safe(path: str) -> None:
        try:
            os.unlink(path)
        except OSError:
            pass

    def _cleanup_owned_files(self, batch: list[PendingRequest]) -> None:
        for req in batch:
            if req.owns_file:
                self._unlink_safe(req.audio_path)

    async def _run_sub_batch(self, batch: list[PendingRequest], timestamps: bool) -> None:
        audio_paths = [r.audio_path for r in batch]
        durations = [self._effective_duration(r) for r in batch]
        try:
            gpu_future = self._gpu_worker.submit(
                WorkType.BATCH_TRANSCRIBE,
                {
                    "audio_paths": audio_paths,
                    "timestamps": timestamps,
                    "batch_size": len(batch),
                    "durations": durations,
                },
                self._loop,
            )
            results = await gpu_future

            if isinstance(results, list) and len(results) == len(batch):
                for req, result in zip(batch, results):
                    if not req.future.done():
                        req.future.set_result(self._serialize_result(result, req.audio_path, req.timestamps))
            else:
                text_results = results if isinstance(results, list) else [results]
                for i, req in enumerate(batch):
                    if not req.future.done():
                        result = text_results[i] if i < len(text_results) else ""
                        req.future.set_result(self._serialize_result(result, req.audio_path, req.timestamps))

        except RuntimeError as exc:
            if "GPU queue full" in str(exc):
                err = QueueFullError(str(exc))
            else:
                err = exc
            log.error(f"Batch transcription failed: {exc}")
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(err)
        except Exception as exc:
            log.error(f"Batch transcription failed: {exc}")
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)
        finally:
            self._cleanup_owned_files(batch)

    @property
    def metrics(self) -> dict:
        return {
            **self._metrics,
            "pending_requests": len(self._pending),
        }


class QueueFullError(Exception):
    pass
