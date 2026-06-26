"""Tests for DrK_Chat.transforms.clean_* — run directly or via pytest:

    python -m tests.test_transforms
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DrK_Chat.transforms import clean_text, clean_segments


def test_collapses_immediate_word_repetition():
    assert clean_text("it's it's hard") == "it's hard"
    assert clean_text("the the the cat") == "the cat"
    assert clean_text("we we need to to talk") == "we need to talk"


def test_collapses_repeated_bigram():
    assert clean_text("i think i think so") == "i think so"


def test_repetition_collapse_is_case_insensitive():
    assert clean_text("The the dog") == "The dog"


def test_strips_fillers_but_keeps_content():
    assert clean_text("um so uh basically yeah") == "so basically yeah"
    # "like" and "you know" are content-ambiguous -> intentionally kept
    assert clean_text("I like it") == "I like it"


def test_does_not_touch_clean_text():
    s = "Anxiety is your brain trying to protect you."
    assert clean_text(s) == s


def test_does_not_collapse_across_punctuation():
    # a period between the repeats must block collapsing (not a stutter)
    assert clean_text("I am done. Done now.") == "I am done. Done now."


def test_normalizes_whitespace():
    assert clean_text("hello   world\n\n  there") == "hello world there"


def test_clean_segments_preserves_timestamps_and_drops_empty():
    segs = [
        {"start": 0.0, "end": 2.0, "text": "so so anxiety"},
        {"start": 2.0, "end": 3.0, "text": "um"},          # becomes empty -> dropped
        {"start": 3.0, "end": 5.0, "text": "is normal"},
    ]
    out = clean_segments(segs)
    assert len(out) == 2
    assert out[0] == {"start": 0.0, "end": 2.0, "text": "so anxiety"}
    assert out[1]["start"] == 3.0 and out[1]["end"] == 5.0  # timestamps intact


def test_cleaning_only_shrinks_or_keeps_length():
    # cleaning must never *grow* text (it only removes redundancy)
    for s in ["the the the", "normal sentence here", "um uh er", "it's it's it's it's"]:
        assert len(clean_text(s)) <= len(s)


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
