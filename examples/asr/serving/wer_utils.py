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
Industry-standard WER evaluation utilities.

Uses Whisper's EnglishTextNormalizer and jiwer for corpus-level WER,
matching NeMo's own evaluation methodology
(nemo/collections/speechlm2/parts/metrics/wer.py).

Requirements:
    pip install jiwer>=3.0.0 whisper-normalizer>=0.1.0
"""

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

_normalizer = EnglishTextNormalizer()


def normalize_text(text: str) -> str:
    """Normalize text using Whisper's EnglishTextNormalizer (industry standard)."""
    return _normalizer(text)


def corpus_wer(references: list[str], hypotheses: list[str]) -> float:
    """Corpus-level WER: normalize both sides, then compute WER across all samples at once.

    This is the industry standard — NOT per-sample average (macro WER).
    Empty references after normalization are skipped.
    """
    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return 0.0
    refs_filtered, hyps_filtered = zip(*pairs)
    return jiwer.wer(list(refs_filtered), list(hyps_filtered))


def pair_wer(reference: str, hypothesis: str) -> float:
    """Single-pair WER for per-sample debugging. Uses jiwer + normalization."""
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref.strip():
        return 0.0 if not hyp.strip() else 1.0
    return jiwer.wer(ref, hyp)
