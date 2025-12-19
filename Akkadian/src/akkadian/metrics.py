from __future__ import annotations

import math
from typing import Iterable, List

import sacrebleu


def corpus_bleu(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    refs = list(references)
    hyps = list(hypotheses)
    if not refs or not hyps:
        return 0.0
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return float(bleu.score)


def corpus_chrf(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    refs = list(references)
    hyps = list(hypotheses)
    if not refs or not hyps:
        return 0.0
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    return float(chrf.score)


def geometric_mean(bleu: float, chrf: float, epsilon: float = 1e-12) -> float:
    return math.sqrt(max(bleu, 0.0) * max(chrf, 0.0) + epsilon)
