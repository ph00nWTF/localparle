"""Sanity tests for WER/CER calls — proves we wired jiwer correctly."""
from __future__ import annotations


def test_jiwer_wer_known_pair():
    import jiwer

    ref = "bonjour comment ca va"
    hyp = "bonjour comment ca va"
    assert jiwer.wer(ref, hyp) == 0.0


def test_jiwer_wer_one_substitution():
    import jiwer

    ref = "bonjour comment ca va"
    hyp = "bonsoir comment ca va"
    # 1 substitution out of 4 ref words → WER = 0.25
    assert abs(jiwer.wer(ref, hyp) - 0.25) < 1e-9
