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
Streaming ASR engine for Nemotron cache-aware models.

Manages per-client streaming sessions. Each client connects via WebSocket,
sends raw 16kHz mono PCM audio chunks, and receives partial/final transcripts.

Architecture:
    Client -> WebSocket -> StreamEngine.open_stream()
                              |
    Client -> audio chunks -> StreamEngine.process_chunk()
                              |
    GPU Worker <- stream_chunk work item -> pipeline.transcribe_step()
                              |
    Client <- partial/final transcript JSON
                              |
    Client -> close -> StreamEngine.close_stream()

The GPU worker thread handles all model inference. The stream engine only
manages session lifecycle and routes chunks to/from the GPU worker.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from gpu_worker import GPUWorker, WorkType

log = logging.getLogger(__name__)


@dataclass
class StreamSession:
    stream_id: str
    created_at: float = field(default_factory=time.monotonic)
    chunks_processed: int = 0
    total_audio_seconds: float = 0.0


class StreamEngine:
    """Manages streaming ASR sessions backed by the GPU worker."""

    def __init__(
        self,
        gpu_worker: GPUWorker,
        max_concurrent_streams: int = 128,
        chunk_duration_ms: int = 160,
        sample_rate: int = 16000,
        max_stream_duration: int = 1800,
        max_chunk_bytes: int = 512 * 1024,
    ):
        self._gpu_worker = gpu_worker
        self._max_concurrent = max_concurrent_streams
        self._chunk_duration_ms = chunk_duration_ms
        self._sample_rate = sample_rate
        self._max_stream_duration = max_stream_duration
        self._max_chunk_bytes = max_chunk_bytes
        self._sessions: dict[str, StreamSession] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._metrics = {
            "total_streams_opened": 0,
            "total_streams_closed": 0,
            "total_chunks_processed": 0,
        }

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()

    async def stop(self) -> None:
        stream_ids = list(self._sessions.keys())
        for sid in stream_ids:
            try:
                await self.close_stream(sid)
            except Exception as exc:
                log.warning(f"Error closing stream {sid} during shutdown: {exc}")

    async def open_stream(self) -> dict:
        """Open a new streaming session. Returns stream_id."""
        stream_id = str(uuid.uuid4())

        if len(self._sessions) >= self._max_concurrent:
            raise TooManyStreamsError(
                f"Active streams {len(self._sessions)} at limit {self._max_concurrent}"
            )
        self._sessions[stream_id] = StreamSession(stream_id=stream_id)

        try:
            result = await self._gpu_worker.submit(
                WorkType.STREAM_OPEN,
                {"stream_id": stream_id},
                self._loop,
            )
        except BaseException:
            self._sessions.pop(stream_id, None)
            raise

        self._metrics["total_streams_opened"] += 1
        log.info(f"Opened stream {stream_id} (active: {len(self._sessions)})")
        return result

    async def process_chunk(self, stream_id: str, audio_bytes: bytes) -> dict:
        """Process an audio chunk for an active stream."""
        if len(audio_bytes) > self._max_chunk_bytes:
            raise ChunkTooLargeError(f"Chunk {len(audio_bytes)} bytes exceeds limit {self._max_chunk_bytes}")

        session = self._sessions.get(stream_id)
        if session is None:
            raise ValueError(f"Unknown stream: {stream_id}")

        elapsed = time.monotonic() - session.created_at
        if elapsed > self._max_stream_duration:
            await self.close_stream(stream_id)
            raise StreamExpiredError(f"Stream {stream_id} exceeded max duration {self._max_stream_duration}s")

        import numpy as np

        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        result = await self._gpu_worker.submit(
            WorkType.STREAM_CHUNK,
            {"stream_id": stream_id, "audio_chunk": audio_np},
            self._loop,
        )

        session.chunks_processed += 1
        session.total_audio_seconds += len(audio_np) / self._sample_rate
        self._metrics["total_chunks_processed"] += 1

        return result

    async def close_stream(self, stream_id: str) -> dict:
        """Close a streaming session and return final transcript."""
        session = self._sessions.pop(stream_id, None)
        if session is None:
            return {"stream_id": stream_id, "status": "not_found"}

        try:
            result = await self._gpu_worker.submit(
                WorkType.STREAM_CLOSE,
                {"stream_id": stream_id},
                self._loop,
            )
        except Exception as exc:
            log.error(f"GPU close failed for stream {stream_id}, GPU-side state may leak: {exc}")
            return {"stream_id": stream_id, "status": "close_failed", "error": str(exc)}

        self._metrics["total_streams_closed"] += 1
        log.info(
            f"Closed stream {stream_id} "
            f"(chunks={session.chunks_processed}, audio={session.total_audio_seconds:.1f}s)"
        )
        return result

    @property
    def metrics(self) -> dict:
        return {
            **self._metrics,
            "active_streams": len(self._sessions),
        }


class TooManyStreamsError(Exception):
    pass


class StreamExpiredError(Exception):
    pass


class ChunkTooLargeError(Exception):
    pass
