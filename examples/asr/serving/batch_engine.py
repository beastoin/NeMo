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


class BatchEngine:
    """Dynamic batcher that accumulates requests and flushes to GPU."""

    def __init__(
        self,
        gpu_worker: GPUWorker,
        max_batch_size: int = 32,
        max_wait_seconds: float = 0.1,
        max_queue_depth: int = 256,
    ):
        self._gpu_worker = gpu_worker
        self._max_batch_size = max_batch_size
        self._max_wait_seconds = max_wait_seconds
        self._max_queue_depth = max_queue_depth
        self._pending: list[PendingRequest] = []
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutting_down = False
        self._metrics = {
            "total_requests": 0,
            "total_batches": 0,
            "total_files": 0,
            "rejected_requests": 0,
        }

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        self._shutting_down = True
        if self._flush_task:
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        while self._pending:
            await self._flush_batch()

    async def submit(self, audio_path: str, timestamps: bool = False, owns_file: bool = False) -> dict:
        """Submit a file for transcription. Returns when the batch completes.

        If owns_file=True, the engine takes ownership and deletes the file
        after GPU processing completes (even on error or cancellation).
        """
        enqueued = False
        try:
            async with self._lock:
                if len(self._pending) >= self._max_queue_depth:
                    self._metrics["rejected_requests"] += 1
                    raise QueueFullError(
                        f"Queue depth {len(self._pending)} exceeds limit {self._max_queue_depth}"
                    )

                future = self._loop.create_future()
                self._pending.append(PendingRequest(
                    audio_path=audio_path,
                    timestamps=timestamps,
                    future=future,
                    owns_file=owns_file,
                ))
                enqueued = True
                self._metrics["total_requests"] += 1

                if len(self._pending) >= self._max_batch_size:
                    asyncio.create_task(self._flush_batch())
        except BaseException:
            if owns_file and not enqueued:
                self._unlink_safe(audio_path)
            raise

        return await future

    async def _flush_loop(self) -> None:
        """Periodically flush partial batches."""
        while not self._shutting_down:
            await asyncio.sleep(self._max_wait_seconds)
            if self._pending:
                await self._flush_batch()

    async def _flush_batch(self) -> None:
        async with self._lock:
            if not self._pending:
                return
            batch = self._pending[: self._max_batch_size]
            self._pending = self._pending[self._max_batch_size:]

        # Partition by timestamps option so we don't mix decode settings
        ts_batch = [r for r in batch if r.timestamps]
        no_ts_batch = [r for r in batch if not r.timestamps]

        self._metrics["total_batches"] += 1
        self._metrics["total_files"] += len(batch)
        log.info(f"Flushing batch: {len(batch)} files (ts={len(ts_batch)}, no_ts={len(no_ts_batch)})")

        for sub_batch, timestamps in [(no_ts_batch, False), (ts_batch, True)]:
            if not sub_batch:
                continue
            await self._run_sub_batch(sub_batch, timestamps)

    @staticmethod
    def _serialize_result(result: Any, audio_path: str, timestamps: bool) -> dict:
        """Convert a NeMo transcription result to a JSON-serializable dict."""
        output = {"audio_path": audio_path}
        if isinstance(result, dict) and "text" in result:
            output.update(result)
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
        try:
            gpu_future = self._gpu_worker.submit(
                WorkType.BATCH_TRANSCRIBE,
                {
                    "audio_paths": audio_paths,
                    "timestamps": timestamps,
                    "batch_size": len(batch),
                },
                self._loop,
            )
            results = await gpu_future

            if isinstance(results, list) and len(results) == len(batch):
                for req, result in zip(batch, results):
                    if not req.future.done():
                        req.future.set_result(self._serialize_result(result, req.audio_path, timestamps))
            else:
                text_results = results if isinstance(results, list) else [results]
                for i, req in enumerate(batch):
                    if not req.future.done():
                        result = text_results[i] if i < len(text_results) else ""
                        req.future.set_result(self._serialize_result(result, req.audio_path, timestamps))

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
