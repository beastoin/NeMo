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
        self.engine._vram_available_mb = 0.0
        limit = self.engine._estimate_max_batch(300.0)
        self.assertEqual(limit, 32)


class TestFormVramSafeBatch(unittest.TestCase):
    def setUp(self):
        self.gpu = _make_mock_gpu_worker()
        self.engine = BatchEngine(self.gpu, max_batch_size=32)
        self.engine._vram_available_mb = 23034.0 * 0.8 - 5262.0
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

    def test_disabled_vram_uses_static_limit(self):
        self.engine._vram_available_mb = 0.0
        reqs = [_make_pending(300.0) for _ in range(50)]
        batch = self.engine._form_vram_safe_batch(reqs)
        self.assertEqual(len(batch), 32)

    def test_always_at_least_one(self):
        reqs = [_make_pending(1200.0)]
        batch = self.engine._form_vram_safe_batch(reqs)
        self.assertEqual(len(batch), 1)


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
