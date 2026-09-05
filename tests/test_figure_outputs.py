"""Figure I/O stays usable with inputs and outputs outside the checkout."""

import json
import shutil
from pathlib import Path

import pytest

from portfolio_optimization.parameter_sensitivity_figure import (
    build_parameter_sensitivity_figure,
)
from portfolio_optimization.performance_figure import build_performance_figure
from portfolio_optimization.rho_ladder_figure import build_rho_ladder_figure
from portfolio_optimization.risk_calibration_figure import build_risk_calibration_figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("builder", "input_argument", "filename"),
    [
        (
            build_parameter_sensitivity_figure,
            "summary_path",
            "article_parameter_sensitivity.csv",
        ),
        (build_rho_ladder_figure, "summary_path", "rho_ladder_summary.csv"),
        (
            build_risk_calibration_figure,
            "beta_path",
            "rolling_realised_beta_252d.csv",
        ),
        (build_performance_figure, "daily_source", "performance_path_daily.parquet"),
    ],
)
def test_figures_accept_external_inputs_and_output_directories(
    tmp_path, builder, input_argument, filename
):
    source = tmp_path / filename
    shutil.copyfile(PROJECT_ROOT / "outputs/review" / filename, source)
    review_root = tmp_path / "review"
    paths = builder(
        **{input_argument: source},
        review_root=review_root,
        figure_root=review_root / "figures",
    )

    assert all(
        path.is_file() and path.is_relative_to(review_root) for path in paths.values()
    )
    manifest = json.loads(paths["manifest"].read_text())
    assert (PROJECT_ROOT / manifest["data"]).resolve().is_file()
    assert {(PROJECT_ROOT / path).resolve() for path in manifest["files"]} == {
        path.resolve() for path in paths.values() if path.suffix == ".svg"
    }
    if builder in (build_performance_figure, build_parameter_sensitivity_figure):
        assert {"mobile_light", "mobile_dark"} <= paths.keys()
