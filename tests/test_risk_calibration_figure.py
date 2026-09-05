from datetime import date

import polars as pl
import pytest

from portfolio_optimization.risk_calibration_figure import (
    LIGHT,
    build_svg,
    monthly_mean_beta,
)


def test_monthly_beta_averages_offsets_before_taking_month_end() -> None:
    beta = pl.DataFrame(
        {
            "allocator": ["b2_memoryless_mvo"] * 4,
            "allocator_label": ["Optimizer"] * 4,
            "offset": ["o0", "o1", "o0", "o1"],
            "date": [
                date(2021, 1, 2),
                date(2021, 1, 2),
                date(2021, 1, 30),
                date(2021, 1, 30),
            ],
            "realised_beta_252d": [0.1, 0.3, 0.2, 0.4],
        }
    )

    result = monthly_mean_beta(beta.lazy()).row(0, named=True)

    assert result["date"] == date(2021, 1, 30)
    assert result["realised_beta_252d"] == pytest.approx(0.3)


def test_beta_svg_contains_all_three_paths_without_target_annotation() -> None:
    rows = []
    for allocator, label, value in (
        ("b1_ranked_volscale", "Volatility-scaled rule", 0.10),
        ("b2_memoryless_mvo", "Standard optimizer", 0.08),
        ("b3_state_aware_mvo", "Optimizer + trading controls", 0.06),
    ):
        for row_date in (date(2000, 1, 3), date(2026, 1, 2)):
            rows.append(
                {
                    "allocator": allocator,
                    "allocator_label": label,
                    "date": row_date,
                    "realised_beta_252d": value,
                }
            )

    svg = build_svg(pl.DataFrame(rows), palette=LIGHT)

    assert svg.count("<path") == 3
    assert "Volatility-scaled rule" in svg
    assert "Optimizer" in svg
    assert "Optimizer + trading controls" in svg
    assert "Target band" not in svg


def test_beta_svg_rejects_values_outside_fixed_axis() -> None:
    beta = pl.DataFrame(
        {
            "allocator": [
                "b1_ranked_volscale",
                "b2_memoryless_mvo",
                "b3_state_aware_mvo",
            ],
            "allocator_label": [
                "Volatility-scaled rule",
                "Standard optimizer",
                "Optimizer + trading controls",
            ],
            "date": [date(2026, 1, 2)] * 3,
            "realised_beta_252d": [0.41, 0.1, 0.1],
        }
    )

    with pytest.raises(ValueError, match="outside the fixed display axis"):
        build_svg(beta, palette=LIGHT)


def test_beta_svg_rejects_incomplete_declared_histories() -> None:
    beta = pl.DataFrame(
        {
            "allocator": ["b1_ranked_volscale", "b2_memoryless_mvo"] * 2,
            "allocator_label": ["B1", "B2"] * 2,
            "date": [date(2025, 1, 2)] * 2 + [date(2026, 1, 2)] * 2,
            "realised_beta_252d": [0.1, 0.1, 0.2, 0.2],
        }
    )

    with pytest.raises(ValueError, match="declared B1, B2, and B3"):
        build_svg(beta, palette=LIGHT)
