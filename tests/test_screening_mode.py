"""
Unit tests for conf_for_mode — NO model, NO GPU, NO heavy imports.

Run with:
    python tests/test_screening_mode.py

All tests use plain assert statements so they work without pytest installed,
though pytest will also discover and run them normally.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on the path so we can import inference directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import conf_for_mode  # noqa: E402 — import after path setup


# ---------------------------------------------------------------------------
# Core ordering invariant
# ---------------------------------------------------------------------------

def test_screening_lt_balanced_lt_specific() -> None:
    """screening < balanced < specific — the fundamental recall/precision ordering.

    This must hold regardless of base_conf so that the API's operating-point
    semantics are always consistent for callers.
    """
    screening = conf_for_mode("screening")
    balanced  = conf_for_mode("balanced")
    specific  = conf_for_mode("specific")

    assert screening < balanced, (
        f"screening ({screening}) should be less than balanced ({balanced})"
    )
    assert balanced < specific, (
        f"balanced ({balanced}) should be less than specific ({specific})"
    )


# ---------------------------------------------------------------------------
# Absolute threshold sanity checks
# ---------------------------------------------------------------------------

def test_screening_threshold_is_low() -> None:
    """screening mode must be clearly below the default balanced threshold."""
    assert conf_for_mode("screening") < 0.25, (
        "screening threshold should be below the default balanced value of 0.25"
    )


def test_specific_threshold_is_high() -> None:
    """specific mode must be clearly above the default balanced threshold."""
    assert conf_for_mode("specific") > 0.25, (
        "specific threshold should be above the default balanced value of 0.25"
    )


def test_balanced_equals_default_base() -> None:
    """balanced should equal the default base_conf (0.25)."""
    assert conf_for_mode("balanced") == 0.25


# ---------------------------------------------------------------------------
# Unknown / unrecognised mode falls back to base_conf
# ---------------------------------------------------------------------------

def test_unknown_mode_falls_back_to_default_base() -> None:
    """An unrecognised mode string should fall back to the default base_conf."""
    assert conf_for_mode("turbo_fracture_mode") == 0.25


def test_unknown_mode_respects_custom_base_conf() -> None:
    """An unrecognised mode with an explicit base_conf should return that value."""
    custom = 0.55
    result = conf_for_mode("not_a_real_mode", base_conf=custom)
    assert result == custom, (
        f"Unknown mode with base_conf={custom} should return {custom}, got {result}"
    )


# ---------------------------------------------------------------------------
# base_conf is a fallback, not an override for known modes
# ---------------------------------------------------------------------------

def test_known_mode_ignores_base_conf() -> None:
    """Passing base_conf should NOT change the threshold for a recognised mode.

    The named operating points are clinically motivated fixed values; allowing
    base_conf to silently shift them would undermine the contract.
    """
    default_screening = conf_for_mode("screening")
    override_attempt  = conf_for_mode("screening", base_conf=0.99)
    assert default_screening == override_attempt, (
        "base_conf should not override a recognised mode's threshold"
    )


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_returns_float() -> None:
    for mode in ("screening", "balanced", "specific", "unknown"):
        result = conf_for_mode(mode)
        assert isinstance(result, float), (
            f"conf_for_mode('{mode}') should return float, got {type(result)}"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_screening_lt_balanced_lt_specific,
        test_screening_threshold_is_low,
        test_specific_threshold_is_high,
        test_balanced_equals_default_base,
        test_unknown_mode_falls_back_to_default_base,
        test_unknown_mode_respects_custom_base_conf,
        test_known_mode_ignores_base_conf,
        test_returns_float,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
