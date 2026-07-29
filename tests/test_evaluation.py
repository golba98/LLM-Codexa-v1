"""Assertion-based tests for generated-text evaluation helpers."""

import json
import math

from src.evaluation import (
    analyze_generated_text,
    ngram_overlap_rate,
    perplexity_from_loss,
    repeated_ngram_rate,
)


def _raises(exception_type: type[BaseException], function) -> None:
    try:
        function()
    except exception_type:
        return
    raise AssertionError(f"Expected {exception_type.__name__}.")


def main() -> None:
    assert math.isclose(perplexity_from_loss(math.log(8)), 8.0)
    _raises(ValueError, lambda: perplexity_from_loss(-1))
    _raises(ValueError, lambda: perplexity_from_loss(float("nan")))

    words = "one two three four one two three four".split()
    assert repeated_ngram_rate(words, 4) > 0
    assert repeated_ngram_rate(["short"], 4) == 0
    _raises(ValueError, lambda: repeated_ngram_rate(words, 0))

    clean = analyze_generated_text("A small bird flew over the green tree.")
    assert clean.word_count == 8
    assert clean.malformed_character_count == 0
    assert clean.longest_repeated_run == 1
    assert json.loads(json.dumps(clean.to_dict())) == clean.to_dict()

    malformed = analyze_generated_text("bad\ufffd\x00 text word word word")
    assert malformed.malformed_character_count == 2
    assert malformed.longest_repeated_run == 3

    overlap = ngram_overlap_rate(
        "the little cat sat on the warm red mat",
        "yesterday the little cat sat on the warm red mat quietly",
        ngram_size=4,
    )
    assert overlap > 0.5
    assert ngram_overlap_rate("too short", "too short", ngram_size=4) == 0

    print("All evaluation-helper tests passed.")


if __name__ == "__main__":
    main()
