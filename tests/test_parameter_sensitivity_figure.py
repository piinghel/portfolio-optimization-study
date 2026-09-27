import polars as pl
import pytest

from portfolio_optimization.parameter_sensitivity_figure import (
    DARK,
    LIGHT,
    build_svg,
)

GRID = {
    "trade_coefficient": (0.0, 1.0, 2.5, 5.0),
    "holding_cutoff": (75.0, 125.0, 175.0, 225.0),
}


def _frame() -> pl.DataFrame:
    rows = []
    for family, values in GRID.items():
        for order, value in enumerate(values, start=1):
            sharpe = 1.30 + 0.01 * order
            turnover = 32.0 - 3.0 * order
            rows.append(
                {
                    "family": family,
                    "value": value,
                    "value_label": f"{value:g}",
                    "value_order": order,
                    "net_sharpe": sharpe,
                    "net_sharpe_schedule_min": sharpe - 0.03,
                    "net_sharpe_schedule_max": sharpe + 0.03,
                    "executed_turnover_l1_annualized": turnover,
                    "executed_turnover_l1_annualized_schedule_min": turnover - 1,
                    "executed_turnover_l1_annualized_schedule_max": turnover + 1,
                }
            )
    return pl.DataFrame(rows)


def test_parameter_figure_uses_the_article_terms_for_both_controls() -> None:
    svg = build_svg(_frame(), palette=LIGHT)

    assert "Trade penalty" in svg
    assert "Rank buffer" in svg
    assert "0 = no penalty" in svg
    assert "Rank-buffer cutoff (75 = no buffer)" in svg
    assert "Annual turnover (× capital)" in svg


def test_parameter_figure_highlights_the_chosen_settings_in_dark_mode() -> None:
    svg = build_svg(_frame(), palette=DARK)

    assert DARK.text in svg
    assert svg.count(f'fill="{DARK.selected}"') == 4


def test_parameter_figure_rejects_missing_grid_cell() -> None:
    with pytest.raises(ValueError, match="complete local sensitivity grid"):
        build_svg(_frame().head(5), palette=LIGHT)
