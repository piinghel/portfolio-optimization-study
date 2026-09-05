"""A four-stock, long-only illustration of eligibility and trading reluctance.

This is one target-weight decision, not the article's long/short backtest.
The fixed inputs are intentionally small enough to inspect by hand. There is
no price simulation, beta or sector constraint, or historical return claim.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray


def allocate(
    previous: NDArray[np.float64], *, buffer: bool, penalty: float
) -> NDArray[np.float64]:
    """Solve the same invented decision with either control enabled or disabled."""
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and non-negative")
    if (
        previous.shape != (4,)
        or not np.isfinite(previous).all()
        or (previous < 0).any()
        or not np.isclose(previous.sum(), 1.0)
    ):
        raise ValueError(
            "previous must contain four finite non-negative weights summing to 1"
        )

    # A/B are fresh selections. C may stay under the buffer; D must exit.
    # These are already-drifted weights at the decision time, not old targets.
    rank = np.array([1, 2, 3, 4])
    score = np.array([0.200, 0.196, 0.192, 0.020])
    covariance = 0.20**2 * (0.70 * np.eye(4) + 0.30 * np.ones((4, 4)))
    eligible = (rank <= 2) | (buffer & (previous > 0) & (rank <= 3))

    weights = cp.Variable(4)
    trade = cp.norm1(weights - previous)
    problem = cp.Problem(
        cp.Maximize(score @ weights - penalty * trade),
        [
            weights >= 0,
            weights <= 0.60,
            cp.sum(weights) == 1,
            weights[~eligible] == 0,
            cp.quad_form(weights, covariance) <= 0.18**2,
        ],
    )
    problem.solve(solver="CLARABEL")
    if problem.status != cp.OPTIMAL or weights.value is None:
        raise RuntimeError(f"Example allocation failed: {problem.status}")
    return np.asarray(weights.value, dtype=np.float64)


def main() -> None:
    previous = np.array([0.0, 0.35, 0.45, 0.20])
    print("Synthetic long-only decision; weights sum to 1 and each is capped at 0.6.")
    print("Pre-trade A/B/C/D weights:", previous)
    print("Rule           A      B      C      D    Two-way trade")
    for label, buffer, penalty in (
        ("Neither", False, 0.0),
        ("Buffer only", True, 0.0),
        ("Penalty only", False, 0.005),
        ("Both", True, 0.005),
    ):
        weights = allocate(previous, buffer=buffer, penalty=penalty)
        trade = float(np.abs(weights - previous).sum())
        values = " ".join(f"{weight:z6.3f}" for weight in weights)
        print(f"{label:12s} {values}    {trade:.3f}")
    print("The buffer permits C to remain; it does not require retention.")
    print("The penalty discourages trading but cannot make D eligible.")
    print("Penalty units are sizing-score units, not a calibrated cash cost.")


if __name__ == "__main__":
    main()
