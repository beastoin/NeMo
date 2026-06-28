"""Tests for dual-model architecture routing in gpu_worker.py."""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, os.path.dirname(__file__))
from gpu_worker import GPUWorker


class FakeModel:
    """Minimal model stub that records transcribe calls."""

    def __init__(self, name="default"):
        self.name = name
        self.transcribe_calls = []

    def transcribe(
        self, audio_input, batch_size=1, timestamps=False, return_hypotheses=False, num_workers=0, verbose=False
    ):
        self.transcribe_calls.append(
            {
                "audio_input": audio_input,
                "batch_size": batch_size,
                "timestamps": timestamps,
            }
        )
        return [f"result_{self.name}_{i}" for i in range(len(audio_input))]

    def eval(self):
        return self

    def change_attention_model(self, *args, **kwargs):
        pass

    def change_subsampling_conv_chunking_factor(self, *args, **kwargs):
        pass


def _make_worker_dual(threshold_sec=600, max_file_duration_sec=0):
    """Create a GPUWorker configured for dual mode with fake models."""
    w = GPUWorker()
    w._attn_mode = "dual"
    w._attn_auto_threshold_sec = threshold_sec
    w._attn_local_context = [128, 128]
    w._max_file_duration_sec = max_file_duration_sec
    w._batch_model_compiled = FakeModel("compiled")
    w._batch_model_local = FakeModel("local")
    w._batch_model = w._batch_model_compiled
    return w


class TestDualModelRouting(unittest.TestCase):
    """Test that _batch_transcribe_dual routes files to the correct model."""

    def setUp(self):
        self.w = _make_worker_dual(threshold_sec=600)

    def _patch_durations(self, duration_map):
        """Patch _get_audio_duration_sec to return durations from a dict."""

        def get_dur(path):
            return duration_map.get(path, 0.0)

        return patch.object(self.w, '_get_audio_duration_sec', side_effect=get_dur)

    def test_all_short_files_use_compiled(self):
        durations = {"a.wav": 30.0, "b.wav": 60.0, "c.wav": 120.0}
        with self._patch_durations(durations):
            payload = {"audio_paths": ["a.wav", "b.wav", "c.wav"], "timestamps": False}
            results = self.w._batch_transcribe_dual(payload)

        self.assertEqual(len(results), 3)
        self.assertEqual(len(self.w._batch_model_compiled.transcribe_calls), 1)
        self.assertEqual(len(self.w._batch_model_local.transcribe_calls), 0)
        self.assertEqual(self.w._batch_model_compiled.transcribe_calls[0]["batch_size"], 3)

    def test_all_long_files_use_local(self):
        durations = {"a.wav": 600.0, "b.wav": 1200.0}
        with self._patch_durations(durations):
            payload = {"audio_paths": ["a.wav", "b.wav"], "timestamps": False}
            results = self.w._batch_transcribe_dual(payload)

        self.assertEqual(len(results), 2)
        self.assertEqual(len(self.w._batch_model_compiled.transcribe_calls), 0)
        self.assertEqual(len(self.w._batch_model_local.transcribe_calls), 1)

    def test_mixed_batch_partitioned(self):
        durations = {"short1.wav": 30.0, "long1.wav": 700.0, "short2.wav": 60.0}
        with self._patch_durations(durations):
            payload = {"audio_paths": ["short1.wav", "long1.wav", "short2.wav"], "timestamps": False}
            results = self.w._batch_transcribe_dual(payload)

        self.assertEqual(len(results), 3)
        self.assertEqual(len(self.w._batch_model_compiled.transcribe_calls), 1)
        self.assertEqual(len(self.w._batch_model_local.transcribe_calls), 1)
        self.assertEqual(self.w._batch_model_compiled.transcribe_calls[0]["audio_input"], ["short1.wav", "short2.wav"])
        self.assertEqual(self.w._batch_model_local.transcribe_calls[0]["audio_input"], ["long1.wav"])

    def test_mixed_batch_preserves_order(self):
        durations = {"a.wav": 30.0, "b.wav": 700.0, "c.wav": 60.0, "d.wav": 800.0}
        with self._patch_durations(durations):
            payload = {"audio_paths": ["a.wav", "b.wav", "c.wav", "d.wav"], "timestamps": False}
            results = self.w._batch_transcribe_dual(payload)

        self.assertEqual(results[0], "result_compiled_0")
        self.assertEqual(results[1], "result_local_0")
        self.assertEqual(results[2], "result_compiled_1")
        self.assertEqual(results[3], "result_local_1")

    def test_unknown_duration_routes_to_local(self):
        durations = {"known.wav": 30.0, "unknown.wav": 0.0}
        with self._patch_durations(durations):
            payload = {"audio_paths": ["known.wav", "unknown.wav"], "timestamps": False}
            results = self.w._batch_transcribe_dual(payload)

        self.assertEqual(len(results), 2)
        self.assertEqual(self.w._batch_model_compiled.transcribe_calls[0]["audio_input"], ["known.wav"])
        self.assertEqual(self.w._batch_model_local.transcribe_calls[0]["audio_input"], ["unknown.wav"])

    def test_exact_threshold_routes_to_local(self):
        durations = {"exact.wav": 600.0}
        with self._patch_durations(durations):
            payload = {"audio_paths": ["exact.wav"], "timestamps": False}
            results = self.w._batch_transcribe_dual(payload)

        self.assertEqual(len(self.w._batch_model_local.transcribe_calls), 1)
        self.assertEqual(len(self.w._batch_model_compiled.transcribe_calls), 0)

    def test_just_below_threshold_routes_to_compiled(self):
        durations = {"just_under.wav": 599.9}
        with self._patch_durations(durations):
            payload = {"audio_paths": ["just_under.wav"], "timestamps": False}
            results = self.w._batch_transcribe_dual(payload)

        self.assertEqual(len(self.w._batch_model_compiled.transcribe_calls), 1)
        self.assertEqual(len(self.w._batch_model_local.transcribe_calls), 0)

    def test_timestamps_forwarded_to_both_models(self):
        durations = {"short.wav": 30.0, "long.wav": 700.0}
        with self._patch_durations(durations):
            payload = {"audio_paths": ["short.wav", "long.wav"], "timestamps": True}
            results = self.w._batch_transcribe_dual(payload)

        self.assertTrue(self.w._batch_model_compiled.transcribe_calls[0]["timestamps"])
        self.assertTrue(self.w._batch_model_local.transcribe_calls[0]["timestamps"])

    def test_max_file_duration_rejects_before_routing(self):
        w = _make_worker_dual(threshold_sec=600, max_file_duration_sec=3600)
        durations = {"ok.wav": 30.0, "too_long.wav": 4000.0}
        with patch.object(w, '_get_audio_duration_sec', side_effect=lambda p: durations[p]):
            payload = {"audio_paths": ["ok.wav", "too_long.wav"], "timestamps": False}
            with self.assertRaises(RuntimeError) as ctx:
                w._batch_transcribe_dual(payload)
            self.assertIn("max_file_duration_sec", str(ctx.exception))

    def test_single_file_short(self):
        durations = {"single.wav": 10.0}
        with self._patch_durations(durations):
            payload = {"audio_paths": ["single.wav"], "timestamps": False}
            results = self.w._batch_transcribe_dual(payload)

        self.assertEqual(len(results), 1)
        self.assertEqual(len(self.w._batch_model_compiled.transcribe_calls), 1)

    def test_single_file_long(self):
        durations = {"single.wav": 1000.0}
        with self._patch_durations(durations):
            payload = {"audio_paths": ["single.wav"], "timestamps": False}
            results = self.w._batch_transcribe_dual(payload)

        self.assertEqual(len(results), 1)
        self.assertEqual(len(self.w._batch_model_local.transcribe_calls), 1)

    def test_all_unknown_duration_routes_all_to_local(self):
        durations = {"a.wav": 0.0, "b.wav": 0.0, "c.wav": 0.0}
        with self._patch_durations(durations):
            payload = {"audio_paths": ["a.wav", "b.wav", "c.wav"], "timestamps": False}
            results = self.w._batch_transcribe_dual(payload)

        self.assertEqual(len(results), 3)
        self.assertEqual(len(self.w._batch_model_compiled.transcribe_calls), 0)
        self.assertEqual(len(self.w._batch_model_local.transcribe_calls), 1)
        self.assertEqual(self.w._batch_model_local.transcribe_calls[0]["batch_size"], 3)


class TestDualModeDispatch(unittest.TestCase):
    """Test that _batch_transcribe dispatches to dual routing when mode is dual."""

    def test_non_dual_uses_single_model(self):
        w = GPUWorker()
        w._attn_mode = "full"
        w._batch_model = FakeModel("single")
        w._max_file_duration_sec = 0
        w._attn_auto_threshold_sec = 600
        payload = {"audio_paths": ["test.wav"], "timestamps": False, "batch_size": 1}
        with patch.object(w, '_get_audio_duration_sec', return_value=30.0):
            results = w._batch_transcribe(payload)
        self.assertEqual(len(results), 1)

    def test_dual_mode_dispatches_to_dual(self):
        w = _make_worker_dual()
        durations = {"test.wav": 30.0}
        with patch.object(w, '_get_audio_duration_sec', side_effect=lambda p: durations.get(p, 0.0)):
            payload = {"audio_paths": ["test.wav"], "timestamps": False}
            results = w._batch_transcribe(payload)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(w._batch_model_compiled.transcribe_calls), 1)


class TestDualModeConfig(unittest.TestCase):
    """Test config validation for dual mode."""

    def test_dual_with_pool_size_raises(self):
        w = GPUWorker()
        w._batch_cfg = {
            "name": "nvidia/parakeet-tdt-0.6b-v3",
            "attention_mode": "dual",
            "model_pool_size": 2,
            "device": "cuda:0",
        }
        w._stream_cfg = {}
        with self.assertRaises(RuntimeError) as ctx:
            w._load_models()
        self.assertIn("incompatible with model_pool_size", str(ctx.exception))

    def test_dual_with_prefetch_raises(self):
        w = GPUWorker()
        w._batch_cfg = {
            "name": "nvidia/parakeet-tdt-0.6b-v3",
            "attention_mode": "dual",
            "prefetch": True,
            "device": "cuda:0",
        }
        w._stream_cfg = {}
        with self.assertRaises(RuntimeError) as ctx:
            w._load_models()
        self.assertIn("incompatible with prefetch", str(ctx.exception))

    def test_invalid_attention_mode_raises(self):
        w = GPUWorker()
        w._batch_cfg = {
            "name": "nvidia/parakeet-tdt-0.6b-v3",
            "attention_mode": "bogus",
            "device": "cuda:0",
        }
        w._stream_cfg = {}
        with self.assertRaises(RuntimeError) as ctx:
            w._load_models()
        self.assertIn("Invalid attention_mode", str(ctx.exception))


class TestAttentionInfo(unittest.TestCase):
    """Test attention_info property."""

    def test_dual_mode_info(self):
        w = _make_worker_dual()
        info = w.attention_info
        self.assertEqual(info["attention_mode"], "dual")
        self.assertTrue(info["compiled_model"])
        self.assertTrue(info["local_model"])

    def test_full_mode_info(self):
        w = GPUWorker()
        info = w.attention_info
        self.assertEqual(info["attention_mode"], "full")
        self.assertNotIn("compiled_model", info)


if __name__ == "__main__":
    unittest.main()
