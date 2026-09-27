import polars as pl
import pytest

from portfolio_optimization.rho_ladder_figure import (
    ALLOCATOR,
    LIGHT,
    RHO_GRID,
    build_svg,
)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "allocator": ALLOCATOR,
            "rho": rho,
            "realised_to_predicted_volatility": 1.0 + rho / 2,
            "beta_mean_error": 0.06 - rho / 50,
            "executed_turnover_l1_annualized": 26 - 4 * rho,
            "net_sharpe": 1.3 - rho / 5,
        }
        for rho in RHO_GRID
    ]


def test_rho_svg_labels_metrics_and_highlights_the_chosen_value() -> None:
    svg = build_svg(pl.DataFrame(_rows()), palette=LIGHT)

    assert "Beta bias" in svg
    assert "Mean across three schedules" in svg
    assert "Two-way turnover (× capital)" in svg
    assert svg.count(f'fill="{LIGHT.selected}"') == 4


def test_rho_svg_rejects_an_incomplete_grid() -> None:
    with pytest.raises(ValueError, match="rho figure requires"):
        build_svg(pl.DataFrame(_rows()[:-1]), palette=LIGHT)
