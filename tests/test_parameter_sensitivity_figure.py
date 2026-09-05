import polars as pl
import pytest

from portfolio_optimization.parameter_sensitivity_figure import (
    DARK,
    LIGHT,
    build_svg,
)


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "family": [
                "trade_coefficient",
                "trade_coefficient",
                "trade_coefficient",
                "trade_coefficient",
                "trade_coefficient",
                "trade_coefficient",
                "holding_cutoff",
                "holding_cutoff",
                "holding_cutoff",
                "holding_cutoff",
                "holding_cutoff",
            ],
            "value": [
                0.0,
                1.0,
                2.0,
                2.5,
                3.0,
                5.0,
                75.0,
                150.0,
                175.0,
                200.0,
                275.0,
            ],
            "value_label": [
                "0",
                "1",
                "2",
                "2.5",
                "3",
                "5",
                "75",
                "150",
                "175",
                "200",
                "275",
            ],
            "value_order": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5],
            "net_sharpe": [
                1.37,
                1.41,
                1.43,
                1.43,
                1.42,
                1.40,
                1.39,
                1.41,
                1.43,
                1.42,
                1.40,
            ],
            "net_sharpe_schedule_min": [
                1.35,
                1.39,
                1.42,
                1.40,
                1.39,
                1.37,
                1.37,
                1.40,
                1.40,
                1.40,
                1.38,
            ],
            "net_sharpe_schedule_max": [
                1.41,
                1.43,
                1.46,
                1.45,
                1.45,
                1.44,
                1.41,
                1.42,
                1.45,
                1.45,
                1.42,
            ],
            "executed_turnover_l1_annualized": [
                40.0,
                34.0,
                30.0,
                28.0,
                27.0,
                24.0,
                35.0,
                29.0,
                28.0,
                27.0,
                26.0,
            ],
            "executed_turnover_l1_annualized_schedule_min": [
                39.0,
                33.0,
                29.0,
                27.0,
                26.0,
                23.0,
                34.0,
                28.0,
                27.0,
                26.0,
                25.0,
            ],
            "executed_turnover_l1_annualized_schedule_max": [
                41.0,
                35.0,
                31.0,
                29.0,
                28.0,
                25.0,
                36.0,
                30.0,
                29.0,
                28.0,
                27.0,
            ],
        }
    )


def test_parameter_figure_contains_both_axes_and_selected_settings() -> None:
    svg = build_svg(_frame(), palette=LIGHT)

    assert "Trade coefficient" in svg
    assert "Holding cutoff" in svg
    assert "Points: means · whiskers: schedule range" in svg
    assert "Annual turnover (× capital)" in svg


def test_parameter_figure_supports_dark_mode() -> None:
    svg = build_svg(_frame(), palette=DARK)

    assert DARK.text in svg
    assert DARK.selected in svg


def test_parameter_figure_rejects_missing_grid_cell() -> None:
    with pytest.raises(ValueError, match="complete local sensitivity grid"):
        build_svg(_frame().head(5), palette=LIGHT)
