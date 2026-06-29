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

"""Unit tests for VRAM-aware batch sizing in BatchEngine."""

import asyncio
import sys
import time
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, __import__("os").path.dirname(__file__))

from batch_engine import BatchEngine, PendingRequest


def _make_mock_gpu_worker(total_mb=23034.0, baseline_mb=5262.0, attention_mode="full", auto_threshold_sec=600.0):
    w = MagicMock()
    w.vram_info = {
        "total_mb": total_mb,
        "baseline_mb": baseline_mb,
        "attention_mode": attention_mode,
        "auto_threshold_sec": auto_threshold_sec,
    }
    w.is_ready = True
    return w


def _make_pending(duration_sec, timestamps=False, age=0.0):
    loop = asyncio.new_event_loop()
    fut = loop.create_future()
    req = PendingRequest(
        audio_path=f"/tmp/audio_{duration_sec}s.wav",
        timestamps=timestamps,
        future=fut,
        owns_file=False,
        submitted_at=time.monotonic() - age,
        duration_sec=duration_sec,
    )
    loop.close()
    return req


class TestEstimateMaxBatch(unittest.TestCase):
    def setUp(self):
        self.gpu = _make_mock_gpu_worker()
        self.engine = BatchEngine(self.gpu, max_batch_size=32)
        self.engine._vram_available_mb = 23034.0 * 0.8 - 5262.0  # ~13165 MB
        self.engine._vram_enabled = True
        self.engine._attention_mode = "full"
        self.engine._auto_threshold_sec = 600.0

    def test_short_audio_allows_full_batch(self):
        limit = self.engine._estimate_max_batch(9.0)
        self.assertEqual(limit, 32)

    def test_5min_audio_limits_batch(self):
        limit = self.engine._estimate_max_batch(300.0)
        self.assertGreater(limit, 1)
        self.assertLess(limit, 32)

    def test_10min_audio_very_small_batch(self):
        limit = self.engine._estimate_max_batch(600.0)
        self.assertGreaterEqual(limit, 1)
        self.assertLessEqual(limit, 3)

    def test_local_attention_skips_limit(self):
        self.engine._attention_mode = "local"
        limit = self.engine._estimate_max_batch(600.0)
        self.assertEqual(limit, 32)

    def test_auto_mode_long_file_skips_limit(self):
        self.engine._attention_mode = "auto"
        limit = self.engine._estimate_max_batch(700.0)
        self.assertEqual(limit, 32)

    def test_auto_mode_short_file_applies_limit(self):
        self.engine._attention_mode = "auto"
        limit = self.engine._estimate_max_batch(300.0)
        self.assertLess(limit, 32)

    def test_zero_duration_returns_max(self):
        limit = self.engine._estimate_max_batch(0.0)
        self.assertEqual(limit, 32)

    def test_disabled_returns_max(self):
        self.engine._vram_enabled = False
        limit = self.engine._estimate_max_batch(300.0)
        self.assertEqual(limit, 32)

    def test_negative_budget_caps_to_one(self):
        self.engine._vram_available_mb = 0.0
        self.engine._vram_enabled = True
        limit = self.engine._estimate_max_batch(300.0)
        self.assertEqual(limit, 1)

    def test_auto_unknown_duration_not_bypassed(self):
        self.engine._attention_mode = "auto"
        limit = self.engine._estimate_max_batch(600.0, duration_known=False)
        self.assertLessEqual(limit, 2)


class TestFormVramSafeBatch(unittest.TestCase):
    def setUp(self):
        self.gpu = _make_mock_gpu_worker()
        self.engine = BatchEngine(self.gpu, max_batch_size=32)
        self.engine._vram_available_mb = 23034.0 * 0.8 - 5262.0
        self.engine._vram_enabled = True
        self.engine._attention_mode = "full"
        self.engine._auto_threshold_sec = 600.0

    def test_all_short_files_full_batch(self):
        reqs = [_make_pending(9.0) for _ in range(32)]
        batch = self.engine._form_vram_safe_batch(reqs)
        self.assertEqual(len(batch), 32)

    def test_mixed_durations_separates(self):
        short = [_make_pending(9.0) for _ in range(20)]
        long = [_make_pending(300.0) for _ in range(5)]
        reqs = short + long
        batch = self.engine._form_vram_safe_batch(reqs)
        self.assertGreater(len(batch), 0)
        self.assertLessEqual(len(batch), 32)
        max_dur = max(self.engine._effective_duration(r) for r in batch)
        limit = self.engine._estimate_max_batch(max_dur)
        self.assertLessEqual(len(batch), limit)

    def test_all_long_files_small_batch(self):
        reqs = [_make_pending(300.0) for _ in range(10)]
        batch = self.engine._form_vram_safe_batch(reqs)
        limit_300s = self.engine._estimate_max_batch(300.0)
        self.assertLessEqual(len(batch), limit_300s)
        self.assertGreater(len(batch), 0)

    def test_starvation_guard(self):
        young_short = [_make_pending(9.0) for _ in range(20)]
        old_long = [_make_pending(300.0, age=10.0)]
        reqs = young_short + old_long
        batch = self.engine._form_vram_safe_batch(reqs)
        old_paths = {old_long[0].audio_path}
        batch_paths = {r.audio_path for r in batch}
        self.assertTrue(old_paths & batch_paths, "Starved request must be included in batch")

    def test_unknown_duration_treated_conservatively(self):
        known = [_make_pending(9.0) for _ in range(10)]
        unknown = [_make_pending(None) for _ in range(5)]
        for r in unknown:
            r.duration_sec = None
        reqs = known + unknown
        batch = self.engine._form_vram_safe_batch(reqs)
        self.assertGreater(len(batch), 0)

    def test_auto_mode_unknown_files_limited(self):
        self.engine._attention_mode = "auto"
        reqs = [_make_pending(None) for _ in range(10)]
        for r in reqs:
            r.duration_sec = None
        batch = self.engine._form_vram_safe_batch(reqs)
        self.assertLess(len(batch), 10, "Unknown-duration files in auto mode must not batch at full size")

    def test_disabled_vram_uses_static_limit(self):
        self.engine._vram_enabled = False
        reqs = [_make_pending(300.0) for _ in range(50)]
        batch = self.engine._form_vram_safe_batch(reqs)
        self.assertEqual(len(batch), 32)

    def test_always_at_least_one(self):
        reqs = [_make_pending(1200.0)]
        batch = self.engine._form_vram_safe_batch(reqs)
        self.assertEqual(len(batch), 1)

    def test_mixed_short_plus_unknown_batches_short(self):
        short = [_make_pending(9.0) for _ in range(20)]
        unknown = [_make_pending(None)]
        unknown[0].duration_sec = None
        reqs = short + unknown
        batch = self.engine._form_vram_safe_batch(reqs)
        self.assertGreater(len(batch), 10, "Short files should not be starved by one unknown file")
        unknown_in_batch = [r for r in batch if r.duration_sec is None]
        self.assertEqual(len(unknown_in_batch), 0, "Unknown file should be excluded, not included")

    def test_negative_budget_caps_all_modes(self):
        self.engine._vram_available_mb = 0.0
        self.engine._attention_mode = "local"
        limit = self.engine._estimate_max_batch(300.0)
        self.assertEqual(limit, 1, "Negative budget must cap to 1 even in local mode")


class TestConcurrentFlush(unittest.TestCase):
    """Tests for concurrent batch flushing (inflight semaphore, fire-and-forget)."""

    def _make_engine_with_mock(self, max_inflight=2, batch_wait=0.01):
        gpu = _make_mock_gpu_worker()
        gpu._pool_size = 1
        engine = BatchEngine(
            gpu,
            max_batch_size=4,
            max_wait_seconds=batch_wait,
            max_queue_depth=4096,
            vram_safety_factor=0.0,
            max_inflight=max_inflight,
        )
        return gpu, engine

    def test_inflight_semaphore_bounds_concurrent_gpu_calls(self):
        """Verify at most max_inflight batches run on GPU simultaneously."""
        gpu, engine = self._make_engine_with_mock(max_inflight=1)
        concurrent_count = {"current": 0, "peak": 0}

        def mock_submit(work_type, work, loop):
            future = loop.create_future()
            concurrent_count["current"] += 1
            concurrent_count["peak"] = max(concurrent_count["peak"], concurrent_count["current"])

            def resolve():
                concurrent_count["current"] -= 1
                results = [{"text": f"ok_{i}"} for i in range(work["batch_size"])]
                if not future.done():
                    future.set_result(results)

            loop.call_soon(resolve)
            return future

        gpu.submit = mock_submit
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(engine.start())

            async def run():
                from unittest.mock import patch

                with patch.object(BatchEngine, '_get_audio_duration', return_value=5.0):
                    tasks = [asyncio.create_task(engine.submit(f"/tmp/a{i}.wav")) for i in range(8)]
                    return await asyncio.gather(*tasks, return_exceptions=True)

            results = loop.run_until_complete(run())
        finally:
            loop.run_until_complete(engine.stop())
            loop.close()

        successes = [r for r in results if not isinstance(r, Exception)]
        self.assertEqual(len(successes), 8)
        self.assertLessEqual(concurrent_count["peak"], 1, "max_inflight=1 should limit to 1 concurrent GPU call")

    def test_stop_drains_inflight_batches(self):
        """stop() should wait for all inflight batches to complete."""
        gpu, engine = self._make_engine_with_mock(max_inflight=2)
        completed = {"count": 0}

        def mock_submit(work_type, work, loop):
            future = loop.create_future()
            results = [{"text": "ok"} for _ in range(work["batch_size"])]

            def resolve():
                completed["count"] += 1
                if not future.done():
                    future.set_result(results)

            loop.call_soon(resolve)
            return future

        gpu.submit = mock_submit
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(engine.start())

            async def run():
                from unittest.mock import patch

                with patch.object(BatchEngine, '_get_audio_duration', return_value=5.0):
                    tasks = [asyncio.create_task(engine.submit(f"/tmp/a{i}.wav")) for i in range(4)]
                    return await asyncio.gather(*tasks, return_exceptions=True)

            loop.run_until_complete(run())
            loop.run_until_complete(engine.stop())
        finally:
            loop.close()

        self.assertGreater(completed["count"], 0, "GPU worker should have been called")

    def test_flush_loop_fires_without_blocking(self):
        """flush_loop creates tasks without awaiting them (fire-and-forget)."""
        gpu, engine = self._make_engine_with_mock(max_inflight=2, batch_wait=0.005)
        flush_calls = {"count": 0}

        def mock_submit(work_type, work, loop):
            future = loop.create_future()
            flush_calls["count"] += 1
            results = [{"text": "ok"} for _ in range(work["batch_size"])]
            loop.call_soon(future.set_result, results)
            return future

        gpu.submit = mock_submit
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(engine.start())

            async def run():
                from unittest.mock import patch

                with patch.object(BatchEngine, '_get_audio_duration', return_value=5.0):
                    tasks = [asyncio.create_task(engine.submit(f"/tmp/a{i}.wav")) for i in range(4)]
                    await asyncio.sleep(0.05)
                    return await asyncio.gather(*tasks, return_exceptions=True)

            results = loop.run_until_complete(run())
        finally:
            loop.run_until_complete(engine.stop())
            loop.close()

        successes = [r for r in results if not isinstance(r, Exception)]
        self.assertEqual(len(successes), 4)

    def test_queue_full_rejects_at_limit(self):
        """Requests beyond max_queue_depth are rejected with QueueFullError."""
        from batch_engine import QueueFullError

        gpu, engine = self._make_engine_with_mock(max_inflight=1)
        engine._max_queue_depth = 2

        never_resolve_futures = []

        def mock_submit(work_type, work, loop):
            future = loop.create_future()
            never_resolve_futures.append(future)
            return future

        gpu.submit = mock_submit
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(engine.start())

            async def run():
                from unittest.mock import patch

                with patch.object(BatchEngine, '_get_audio_duration', return_value=5.0):
                    engine._max_wait_seconds = 999
                    t1 = asyncio.create_task(engine.submit("/tmp/a0.wav"))
                    t2 = asyncio.create_task(engine.submit("/tmp/a1.wav"))
                    await asyncio.sleep(0.01)
                    try:
                        await engine.submit("/tmp/a2.wav")
                        return None
                    except QueueFullError as e:
                        return e

            result = loop.run_until_complete(run())
        finally:
            for f in never_resolve_futures:
                if not f.done():
                    f.set_result([])
            loop.run_until_complete(engine.stop())
            loop.close()

        self.assertIsNotNone(result, "Third submit should raise QueueFullError")
        self.assertIn("Queue depth", str(result))

    def test_mixed_timestamps_no_leak(self):
        """Non-timestamp request must not receive timestamp data when batched with timestamp request."""
        gpu, engine = self._make_engine_with_mock(max_inflight=2, batch_wait=0.01)
        engine._max_batch_size = 2

        def mock_submit(work_type, work, loop):
            future = loop.create_future()
            results = []
            for i in range(work["batch_size"]):
                r = MagicMock()
                r.text = f"text_{i}"
                r.timestamp = {"word": [{"word": "hello", "start": 0.0, "end": 0.5}]}
                results.append(r)
            loop.call_soon(future.set_result, results)
            return future

        gpu.submit = mock_submit
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(engine.start())

            async def run():
                from unittest.mock import patch

                with patch.object(BatchEngine, '_get_audio_duration', return_value=5.0):
                    t_no_ts = asyncio.create_task(engine.submit("/tmp/no_ts.wav", timestamps=False))
                    t_ts = asyncio.create_task(engine.submit("/tmp/ts.wav", timestamps=True))
                    return await asyncio.gather(t_no_ts, t_ts, return_exceptions=True)

            results = loop.run_until_complete(run())
        finally:
            loop.run_until_complete(engine.stop())
            loop.close()

        no_ts_result, ts_result = results[0], results[1]
        self.assertNotIsInstance(no_ts_result, Exception)
        self.assertNotIsInstance(ts_result, Exception)
        self.assertNotIn("word", no_ts_result, "Non-timestamp request must not receive timestamp data")
        self.assertIn("word", ts_result, "Timestamp request must receive timestamp data")

    def test_semaphore_before_dequeue(self):
        """Requests stay in _pending until a semaphore slot is available."""
        gpu, engine = self._make_engine_with_mock(max_inflight=1, batch_wait=999)
        engine._max_batch_size = 1

        gate = asyncio.Event()
        submitted_to_gpu = []

        def mock_submit(work_type, work, loop):
            future = loop.create_future()
            submitted_to_gpu.append(work)

            async def wait_and_resolve():
                await gate.wait()
                if not future.done():
                    future.set_result([{"text": "ok"}])

            asyncio.ensure_future(wait_and_resolve())
            return future

        gpu.submit = mock_submit
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(engine.start())

            async def run():
                from unittest.mock import patch

                with patch.object(BatchEngine, '_get_audio_duration', return_value=5.0):
                    t1 = asyncio.create_task(engine.submit("/tmp/a0.wav"))
                    await asyncio.sleep(0.02)
                    t2 = asyncio.create_task(engine.submit("/tmp/a1.wav"))
                    await asyncio.sleep(0.02)

                    self.assertEqual(len(submitted_to_gpu), 1, "Only 1 batch should be on GPU (semaphore=1)")
                    self.assertGreaterEqual(len(engine._pending), 1, "Second request should still be pending")

                    gate.set()
                    return await asyncio.gather(t1, t2, return_exceptions=True)

            results = loop.run_until_complete(run())
        finally:
            loop.run_until_complete(engine.stop())
            loop.close()

        successes = [r for r in results if not isinstance(r, Exception)]
        self.assertEqual(len(successes), 2)


class TestEffectiveDuration(unittest.TestCase):
    def test_known_duration(self):
        engine = BatchEngine(_make_mock_gpu_worker())
        engine._auto_threshold_sec = 600.0
        req = _make_pending(42.0)
        self.assertEqual(engine._effective_duration(req), 42.0)

    def test_unknown_duration_uses_threshold(self):
        engine = BatchEngine(_make_mock_gpu_worker())
        engine._auto_threshold_sec = 600.0
        req = _make_pending(None)
        req.duration_sec = None
        self.assertEqual(engine._effective_duration(req), 600.0)


if __name__ == "__main__":
    unittest.main()
