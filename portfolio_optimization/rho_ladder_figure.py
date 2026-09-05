"""Build the single article figure for the correlation-shrinkage ladder."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from portfolio_optimization.svg_primitives import svg_line as _line
from portfolio_optimization.svg_primitives import svg_text as _text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = PROJECT_ROOT / "outputs" / "review"
SUMMARY_PATH = REVIEW_ROOT / "rho_ladder_summary.csv"
FIGURE_ROOT = REVIEW_ROOT / "figures"
WIDTH = 1200
HEIGHT = 780


@dataclass(frozen=True)
class Palette:
    text: str
    muted: str
    grid: str
    stable: str
    b2: str
    b3: str


LIGHT = Palette(
    text="#172033",
    muted="#667085",
    grid="#D9DEE8",
    stable="#EDF1F3",
    b2="#345B7E",
    b3="#378579",
)
DARK = Palette(
    text="#F3F4F6",
    muted="#AAB2C0",
    grid="#3B4250",
    stable="#202A32",
    b2="#78A0C4",
    b3="#6EB5A5",
)


@dataclass(frozen=True)
class Panel:
    title: str
    metric: str
    y_min: float
    y_max: float
    ticks: tuple[float, ...]
    formatter: str
    note: str


PANELS = (
    Panel(
        "Risk calibration",
        "realised_to_predicted_volatility",
        1.0,
        1.8,
        (1.0, 1.2, 1.4, 1.6, 1.8),
        ".2f",
        "Root-mean realized / forecast volatility",
    ),
    Panel(
        "Beta error",
        "beta_mae",
        0.17,
        0.26,
        (0.18, 0.20, 0.22, 0.24),
        ".2f",
        "Next-holding-period mean absolute error",
    ),
    Panel(
        "Annual turnover",
        "executed_turnover_l1_annualized",
        25.0,
        47.0,
        (30.0, 35.0, 40.0, 45.0),
        ".0f",
        "Two-way turnover (× capital)",
    ),
    Panel(
        "Net Sharpe",
        "net_sharpe",
        0.95,
        1.46,
        (1.0, 1.1, 1.2, 1.3, 1.4),
        ".1f",
        "Mean across three schedules",
    ),
)


def _validate(frame: pl.DataFrame) -> None:
    expected_grid = {
        (allocator, rho / 10) for allocator in ("b2", "b3") for rho in range(11)
    }
    observed_grid = {
        (str(allocator), round(float(rho), 10))
        for allocator, rho in frame.select("allocator", "rho").iter_rows()
    }
    if frame.height != len(expected_grid) or observed_grid != expected_grid:
        raise ValueError("rho figure requires the complete B2/B3 0.0-1.0 grid")
    for panel in PANELS:
        if frame.filter(
            pl.col(panel.metric).is_null()
            | ~pl.col(panel.metric).is_finite()
            | (pl.col(panel.metric) < panel.y_min)
            | (pl.col(panel.metric) > panel.y_max)
        ).height:
            raise ValueError(f"{panel.metric} falls outside its fixed figure axis")


def _panel_svg(
    frame: pl.DataFrame,
    panel: Panel,
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    palette: Palette,
    mobile: bool,
) -> list[str]:
    left = x0 + 62
    right = x0 + width - 25
    top = y0 + 70
    bottom = y0 + height - 65
    label_size = 18 if mobile else 21

    def x_position(value: float) -> float:
        return left + value * (right - left)

    def y_position(value: float) -> float:
        return bottom - (value - panel.y_min) / (panel.y_max - panel.y_min) * (
            bottom - top
        )

    elements = [
        _text(
            x0 + 8,
            y0 + 23,
            panel.title,
            fill=palette.text,
            size=20 if mobile else 24,
            weight=600,
        ),
        _text(
            x0 + 8, y0 + 47, panel.note, fill=palette.muted, size=18 if mobile else 20
        ),
        (
            f'<rect x="{x_position(0.3):.1f}" y="{top:.1f}" '
            f'width="{x_position(0.6) - x_position(0.3):.1f}" '
            f'height="{bottom - top:.1f}" fill="{palette.stable}"/>'
        ),
    ]
    for tick in panel.ticks:
        y = y_position(tick)
        elements.append(_line(left, y, right, y, stroke=palette.grid, stroke_width="1"))
        elements.append(
            _text(
                left - 10,
                y + 5,
                format(tick, panel.formatter),
                fill=palette.muted,
                size=label_size,
                anchor="end",
            )
        )
    for rho in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        x = x_position(rho)
        elements.append(
            _line(x, bottom, x, bottom + 5, stroke=palette.grid, stroke_width="1")
        )
        elements.append(
            _text(
                x,
                bottom + 23,
                format(rho, ".1f"),
                fill=palette.muted,
                size=label_size,
                anchor="middle",
            )
        )
    for allocator, color in (
        ("b2", palette.b2),
        ("b3", palette.b3),
    ):
        rows = frame.filter(pl.col("allocator") == allocator).sort("rho")
        points = [
            (x_position(float(rho)), y_position(float(value)))
            for rho, value in rows.select("rho", panel.metric).iter_rows()
        ]
        path = " ".join(
            f"{'M' if point_index == 0 else 'L'}{x:.1f},{y:.1f}"
            for point_index, (x, y) in enumerate(points)
        )
        elements.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        elements.extend(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
            for x, y in points
        )
    elements.append(
        _text(
            (left + right) / 2,
            bottom + 50,
            "Correlation shrinkage ρ",
            fill=palette.muted,
            size=label_size,
            anchor="middle",
        )
    )
    return elements


def build_svg(frame: pl.DataFrame, *, palette: Palette, mobile: bool = False) -> str:
    _validate(frame)
    width, height = (480, 1435) if mobile else (WIDTH, HEIGHT)
    label_size = 18 if mobile else 21
    legend_left = 24 if mobile else 90
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title desc">'
        ),
        '<title id="title">How much correlation shrinkage matters</title>',
        (
            '<desc id="desc">Four panels compare risk calibration, beta error, '
            "turnover, and net Sharpe for the optimizer with and without trading "
            "controls as correlation shrinkage moves from zero to one.</desc>"
        ),
        '<g font-family="DejaVu Sans, sans-serif">',
        _line(
            legend_left, 20, legend_left + 30, 20, stroke=palette.b2, stroke_width="3"
        ),
        _text(legend_left + 40, 25, "Optimizer", fill=palette.text, size=label_size),
        _line(
            legend_left if mobile else 310,
            50 if mobile else 20,
            legend_left + 30 if mobile else 340,
            50 if mobile else 20,
            stroke=palette.b3,
            stroke_width="3",
        ),
        _text(
            legend_left + 40 if mobile else 350,
            55 if mobile else 25,
            "Optimizer + trading controls",
            fill=palette.text,
            size=label_size,
        ),
    ]
    positions = (
        ((8, 90), (8, 425), (8, 760), (8, 1095))
        if mobile
        else ((45, 60), (630, 60), (45, 420), (630, 420))
    )
    for panel, (x0, y0) in zip(PANELS, positions, strict=True):
        elements.extend(
            _panel_svg(
                frame,
                panel,
                x0=x0,
                y0=y0,
                width=460 if mobile else 540,
                height=335 if mobile else 345,
                palette=palette,
                mobile=mobile,
            )
        )
    elements.extend(("</g>", "</svg>"))
    return "\n".join(elements) + "\n"


def build_rho_ladder_figure(
    *,
    summary_path: Path = SUMMARY_PATH,
    figure_root: Path = FIGURE_ROOT,
    review_root: Path = REVIEW_ROOT,
) -> dict[str, Path]:
    frame = pl.scan_csv(summary_path).sort("allocator", "rho").collect()
    review_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "light": figure_root / "rho-ladder.svg",
        "dark": figure_root / "rho-ladder_dark.svg",
        "mobile_light": figure_root / "rho-ladder_mobile.svg",
        "mobile_dark": figure_root / "rho-ladder_mobile_dark.svg",
        "caption": review_root / "rho_ladder_figure_caption.md",
        "manifest": review_root / "rho_ladder_figure_manifest.json",
    }
    paths["light"].write_text(build_svg(frame, palette=LIGHT), encoding="utf-8")
    paths["dark"].write_text(build_svg(frame, palette=DARK), encoding="utf-8")
    for theme, palette in (("light", LIGHT), ("dark", DARK)):
        paths[f"mobile_{theme}"].write_text(
            build_svg(frame, palette=palette, mobile=True), encoding="utf-8"
        )
    paths["caption"].write_text(
        "**Figure 3. Correlation shrinkage.** The optimizer "
        "and the optimizer with trading controls are rebuilt at every shrinkage "
        "value from "
        "0 to 1 using development data. The horizontal axis in every panel is "
        "correlation shrinkage. The four panels show forecast calibration, "
        "beta error over the next holding period, "
        "turnover, and net Sharpe. The curves are stable from 0.3 to "
        "0.6; moving from 0.4 to the implemented 0.5 barely changes the result.\n",
        encoding="utf-8",
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "display": "Figure 3",
                "question": "Is the correlation-shrinkage choice stable?",
                "data": os.path.relpath(summary_path, PROJECT_ROOT),
                "files": [
                    os.path.relpath(paths["light"], PROJECT_ROOT),
                    os.path.relpath(paths["dark"], PROJECT_ROOT),
                    os.path.relpath(paths["mobile_light"], PROJECT_ROOT),
                    os.path.relpath(paths["mobile_dark"], PROJECT_ROOT),
                ],
                "observation": (
                    "Risk calibration and beta error are lowest around 0.4. "
                    "Turnover and net Sharpe move little from 0.3 through 0.6, "
                    "while both endpoints are weaker."
                ),
                "supported_conclusion": (
                    "The implemented 0.5 setting lies inside a stable local "
                    "region rather than at an isolated optimum."
                ),
                "limitation": (
                    "The ladder supports a stable local region. A separate "
                    "sample would be needed to estimate an optimal value."
                ),
                "article_worthy": True,
                "supporting_only": [
                    "mean_qlike",
                    "overshoot counts",
                    "executed_weight_l1_vs_rho50",
                ],
                "mobile_specific_asset": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def main() -> None:
    for name, path in build_rho_ladder_figure().items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
