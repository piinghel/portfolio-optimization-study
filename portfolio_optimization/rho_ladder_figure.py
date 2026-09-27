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


ALLOCATOR = "optimizer_with_trading_controls"
RHO_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SELECTED_RHO = 0.5


@dataclass(frozen=True)
class Palette:
    text: str
    muted: str
    grid: str
    line: str
    selected: str


LIGHT = Palette(
    text="#172033",
    muted="#667085",
    grid="#D9DEE8",
    line="#378579",
    selected="#B98556",
)
DARK = Palette(
    text="#F3F4F6",
    muted="#AAB2C0",
    grid="#3B4250",
    line="#6EB5A5",
    selected="#C79261",
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
        0.9,
        1.6,
        (1.0, 1.2, 1.4, 1.6),
        ".1f",
        "Root-mean realized / forecast volatility",
    ),
    Panel(
        "Beta bias",
        "beta_mean_error",
        0.0,
        0.08,
        (0.0, 0.02, 0.04, 0.06, 0.08),
        ".2f",
        "Realized minus forecast beta, next holding period",
    ),
    Panel(
        "Annual turnover",
        "executed_turnover_l1_annualized",
        20.0,
        28.0,
        (20.0, 22.0, 24.0, 26.0, 28.0),
        ".0f",
        "Two-way turnover (× capital)",
    ),
    Panel(
        "Net Sharpe",
        "net_sharpe",
        1.0,
        1.4,
        (1.0, 1.1, 1.2, 1.3, 1.4),
        ".1f",
        "Mean across three schedules",
    ),
)


def _validate(frame: pl.DataFrame) -> None:
    expected_grid = {(ALLOCATOR, rho) for rho in RHO_GRID}
    observed_grid = {
        (str(allocator), round(float(rho), 10))
        for allocator, rho in frame.select("allocator", "rho").iter_rows()
    }
    if frame.height != len(expected_grid) or observed_grid != expected_grid:
        raise ValueError(f"rho figure requires {ALLOCATOR} at {RHO_GRID}")
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
    for rho in RHO_GRID:
        x = x_position(rho)
        elements.append(
            _line(x, bottom, x, bottom + 5, stroke=palette.grid, stroke_width="1")
        )
        elements.append(
            _text(
                x,
                bottom + 23,
                format(rho, "g"),
                fill=palette.muted,
                size=label_size,
                anchor="middle",
            )
        )
    rows = frame.sort("rho")
    points = [
        (float(rho), x_position(float(rho)), y_position(float(value)))
        for rho, value in rows.select("rho", panel.metric).iter_rows()
    ]
    path = " ".join(
        f"{'M' if index == 0 else 'L'}{x:.1f},{y:.1f}"
        for index, (_, x, y) in enumerate(points)
    )
    elements.append(
        f'<path d="{path}" fill="none" stroke="{palette.line}" stroke-width="3" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )
    elements.extend(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{6 if rho == SELECTED_RHO else 4}" '
        f'fill="{palette.selected if rho == SELECTED_RHO else palette.line}"/>'
        for rho, x, y in points
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
    width, height = (480, 1365) if mobile else (WIDTH, HEIGHT - 40)
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title desc">'
        ),
        '<title id="title">How much correlation shrinkage matters</title>',
        (
            '<desc id="desc">Four panels show risk calibration, beta bias, '
            "turnover, and net Sharpe for the optimizer with trading controls as "
            "correlation shrinkage moves from zero to one; the chosen 0.5 is "
            "highlighted.</desc>"
        ),
        '<g font-family="DejaVu Sans, sans-serif">',
    ]
    positions = (
        ((8, 20), (8, 355), (8, 690), (8, 1025))
        if mobile
        else ((45, 20), (630, 20), (45, 380), (630, 380))
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
        "The optimizer with trading controls rebuilt at correlation shrinkage "
        "0, 0.25, 0.5, 0.75 and 1 on development data: risk calibration, beta "
        "bias over the next holding period, annual turnover and net Sharpe. The "
        "chosen 0.5 is highlighted.\n",
        encoding="utf-8",
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "question": "How do risk, beta, turnover and Sharpe respond to shrinkage?",
                "data": os.path.relpath(summary_path, PROJECT_ROOT),
                "files": [
                    os.path.relpath(paths["light"], PROJECT_ROOT),
                    os.path.relpath(paths["dark"], PROJECT_ROOT),
                    os.path.relpath(paths["mobile_light"], PROJECT_ROOT),
                    os.path.relpath(paths["mobile_dark"], PROJECT_ROOT),
                ],
                "limitation": (
                    "The ladder supports a stable local region. A separate "
                    "sample would be needed to estimate an optimal value."
                ),
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
