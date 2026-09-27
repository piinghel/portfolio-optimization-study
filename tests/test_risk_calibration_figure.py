from datetime import date

import polars as pl
import pytest

from portfolio_optimization.risk_calibration_figure import (
    ALLOCATORS,
    LIGHT,
    build_svg,
    monthly_mean_beta,
)


def _paths(values: tuple[float, float]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "allocator": allocator,
                "allocator_label": label,
                "date": row_date,
                "realised_beta_252d": value,
            }
            for (allocator, label, _, _), value in zip(ALLOCATORS, values, strict=True)
            for row_date in (date(2000, 1, 3), date(2021, 12, 31))
        ]
    )


def test_monthly_beta_averages_offsets_before_taking_month_end() -> None:
    allocator, label, _, _ = ALLOCATORS[0]
    beta = pl.DataFrame(
        {
            "allocator": [allocator] * 4,
            "allocator_label": [label] * 4,
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


def test_beta_svg_draws_both_paths_the_limit_band_and_direct_labels() -> None:
    svg = build_svg(_paths((0.10, 0.06)), palette=LIGHT)

    assert svg.count("<path") == 2
    assert f'fill="{LIGHT.band}"' in svg
    assert "Volatility-scaled" in svg
    assert "trading controls" in svg


def test_beta_svg_rejects_values_outside_fixed_axis() -> None:
    with pytest.raises(ValueError, match="outside the fixed display axis"):
        build_svg(_paths((0.41, 0.1)), palette=LIGHT)


def test_beta_svg_rejects_incomplete_declared_histories() -> None:
    beta = _paths((0.1, 0.1)).filter(pl.col("allocator") == ALLOCATORS[0][0])

    with pytest.raises(ValueError, match="requires the histories"):
        build_svg(beta, palette=LIGHT)
