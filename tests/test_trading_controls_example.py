"""Eligibility and turnover invariants of the one-rebalance teaching example."""

from __future__ import annotations

import numpy as np
import pytest

from portfolio_optimization import trading_controls


def test_controls_preserve_constraints_and_distinguish_permission_from_retention():
    previous = np.array([0.0, 0.35, 0.45, 0.20])
    solutions = {}
    for buffer in (False, True):
        for penalty in (0.0, 0.005):
            weights = trading_controls.allocate(
                previous, buffer=buffer, penalty=penalty
            )
            solutions[buffer, penalty] = weights
            assert weights.min() >= -1e-7
            assert weights.max() <= 0.60 + 1e-7
            assert weights.sum() == pytest.approx(1.0, abs=1e-7)
            assert weights[3] == pytest.approx(0.0, abs=1e-7)
            variance = 0.20**2 * (0.70 * (weights @ weights) + 0.30)
            assert variance <= 0.18**2 + 1e-7
            if not buffer:
                assert weights[2] == pytest.approx(0.0, abs=1e-7)

    # Without reluctance, permission alone does not retain the lower-score C.
    assert solutions[True, 0.0][2] == pytest.approx(0.0, abs=1e-5)
    assert solutions[True, 0.005] == pytest.approx([0.20, 0.35, 0.45, 0.0], abs=1e-5)
    trade_without = np.abs(solutions[False, 0.0] - previous).sum()
    trade_with = np.abs(solutions[True, 0.005] - previous).sum()
    assert trade_without == pytest.approx(1.30, abs=1e-5)
    assert trade_with == pytest.approx(0.40, abs=1e-5)


def test_buffer_cannot_admit_a_lower_ranked_nonincumbent():
    weights = trading_controls.allocate(
        np.array([0.0, 0.80, 0.0, 0.20]), buffer=True, penalty=1.0
    )
    assert weights[2:] == pytest.approx([0.0, 0.0], abs=1e-7)


@pytest.mark.parametrize("penalty", [-1.0, float("nan"), float("inf")])
def test_rejects_invalid_penalty(penalty):
    with pytest.raises(ValueError, match="penalty"):
        trading_controls.allocate(
            np.array([0.0, 0.35, 0.45, 0.20]), buffer=True, penalty=penalty
        )
