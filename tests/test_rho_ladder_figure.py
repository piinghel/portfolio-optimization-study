import polars as pl
import pytest

from portfolio_optimization.rho_ladder_figure import LIGHT, build_svg


def test_rho_svg_labels_allocators_and_metric_units() -> None:
    rows = [
        {
            "allocator": allocator,
            "rho": rho / 10,
            "realised_to_predicted_volatility": 1.2 + rho / 100,
            "beta_mae": 0.18 + rho / 1000,
            "executed_turnover_l1_annualized": 40 - rho,
            "net_sharpe": 1.4 - rho / 100,
        }
        for allocator in ("b2", "b3")
        for rho in range(11)
    ]

    svg = build_svg(pl.DataFrame(rows), palette=LIGHT)

    assert "Optimizer" in svg
    assert "Optimizer + trading controls" in svg
    assert "Mean across three schedules" in svg
    assert "Two-way turnover (× capital)" in svg


def test_rho_svg_rejects_an_incomplete_allocator_grid() -> None:
    rows = [
        {
            "allocator": allocator,
            "rho": rho / 10,
            "realised_to_predicted_volatility": 1.2 + rho / 100,
            "beta_mae": 0.18 + rho / 1000,
            "executed_turnover_l1_annualized": 40 - rho,
            "net_sharpe": 1.3 - rho / 100,
        }
        for allocator in ("b2", "b3")
        for rho in range(11)
    ]

    with pytest.raises(ValueError, match="complete B2/B3"):
        build_svg(pl.DataFrame(rows[:-1]), palette=LIGHT)
