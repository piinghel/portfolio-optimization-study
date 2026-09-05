"""Render the development-period trade-coefficient and holding-cutoff check."""

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
SUMMARY_PATH = REVIEW_ROOT / "article_parameter_sensitivity.csv"
FIGURE_ROOT = REVIEW_ROOT / "figures"
WIDTH = 1200
HEIGHT = 720


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
    line="#64748B",
    selected="#B98556",
)
DARK = Palette(
    text="#F3F4F6",
    muted="#AAB2C0",
    grid="#3B4250",
    line="#AAB2C0",
    selected="#C79261",
)


@dataclass(frozen=True)
class Column:
    family: str
    title: str
    x_label: str
    selected_value: float


COLUMNS = (
    Column(
        "trade_coefficient",
        "Trade coefficient",
        "Coefficient c (×10⁻⁴)",
        2.5,
    ),
    Column(
        "holding_cutoff",
        "Holding cutoff",
        "Rank cutoff",
        175.0,
    ),
)


@dataclass(frozen=True)
class Row:
    title: str
    metric: str
    y_min: float
    y_max: float
    ticks: tuple[float, ...]
    formatter: str


ROWS = (
    Row("Net Sharpe", "net_sharpe", 1.32, 1.47, (1.35, 1.40, 1.45), ".2f"),
    Row(
        "Annual turnover (× capital)",
        "executed_turnover_l1_annualized",
        20.0,
        45.0,
        (20.0, 30.0, 40.0),
        ".0f",
    ),
)


def _validate(frame: pl.DataFrame) -> None:
    expected = {
        ("trade_coefficient", 0.0),
        ("trade_coefficient", 1.0),
        ("trade_coefficient", 2.0),
        ("trade_coefficient", 2.5),
        ("trade_coefficient", 3.0),
        ("trade_coefficient", 5.0),
        ("holding_cutoff", 75.0),
        ("holding_cutoff", 150.0),
        ("holding_cutoff", 175.0),
        ("holding_cutoff", 200.0),
        ("holding_cutoff", 275.0),
    }
    observed = {
        (str(family), float(value))
        for family, value in frame.select("family", "value").iter_rows()
    }
    if frame.height != len(expected) or observed != expected:
        raise ValueError(
            "parameter figure requires the complete local sensitivity grid"
        )
    for row in ROWS:
        columns = (
            row.metric,
            f"{row.metric}_schedule_min",
            f"{row.metric}_schedule_max",
        )
        if frame.filter(
            pl.any_horizontal(
                pl.col(column).is_null() | ~pl.col(column).is_finite()
                for column in columns
            )
            | (pl.min_horizontal(*columns) < row.y_min)
            | (pl.max_horizontal(*columns) > row.y_max)
        ).height:
            raise ValueError(f"{row.metric} falls outside its fixed figure axis")


def _panel(
    frame: pl.DataFrame,
    *,
    column: Column,
    row: Row,
    x0: float,
    y0: float,
    width: float,
    height: float,
    palette: Palette,
    show_column_title: bool,
) -> list[str]:
    left = x0 + 72
    right = x0 + width - 28
    top = y0 + 52
    bottom = y0 + height - 58
    values = frame.filter(pl.col("family") == column.family).sort("value")
    x_values = values.get_column("value").to_list()
    x_min = float(min(x_values))
    x_max = float(max(x_values))

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    def y_position(value: float) -> float:
        return bottom - (value - row.y_min) / (row.y_max - row.y_min) * (bottom - top)

    elements: list[str] = []
    if show_column_title:
        elements.append(
            _text(x0 + 8, y0 + 23, column.title, fill=palette.text, size=20, weight=600)
        )
    elements.append(
        _text(x0 + 8, top - 8, row.title, fill=palette.text, size=18, weight=400)
    )
    for tick in row.ticks:
        y = y_position(tick)
        elements.extend(
            (
                _line(left, y, right, y, stroke=palette.grid, stroke_width="1"),
                _text(
                    left - 12,
                    y + 5,
                    format(tick, row.formatter),
                    fill=palette.muted,
                    size=18,
                    anchor="end",
                ),
            )
        )
    points: list[tuple[float, float]] = []
    for value, mean, low, high, label in values.select(
        "value",
        row.metric,
        f"{row.metric}_schedule_min",
        f"{row.metric}_schedule_max",
        "value_label",
    ).iter_rows():
        x = x_position(float(value))
        y = y_position(float(mean))
        points.append((x, y))
        elements.extend(
            (
                _line(
                    x,
                    y_position(float(low)),
                    x,
                    y_position(float(high)),
                    stroke=palette.line,
                    stroke_width="2",
                ),
                _line(
                    x - 5,
                    y_position(float(low)),
                    x + 5,
                    y_position(float(low)),
                    stroke=palette.line,
                    stroke_width="2",
                ),
                _line(
                    x - 5,
                    y_position(float(high)),
                    x + 5,
                    y_position(float(high)),
                    stroke=palette.line,
                    stroke_width="2",
                ),
                (
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" '
                    f'fill="{palette.selected if float(value) == column.selected_value else palette.line}"/>'
                ),
                _text(
                    x,
                    bottom + 24,
                    f"{float(label):g}",
                    fill=palette.muted,
                    size=18,
                    anchor="middle",
                ),
            )
        )
    path = " ".join(
        f"{'M' if index == 0 else 'L'}{x:.1f},{y:.1f}"
        for index, (x, y) in enumerate(points)
    )
    elements.insert(
        1,
        f'<path d="{path}" fill="none" stroke="{palette.line}" '
        'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>',
    )
    elements.append(
        _text(
            (left + right) / 2,
            bottom + 48,
            column.x_label,
            fill=palette.muted,
            size=18,
            anchor="middle",
        )
    )
    return elements


def build_svg(frame: pl.DataFrame, *, palette: Palette, mobile: bool = False) -> str:
    _validate(frame)
    width, height = (480, 1240) if mobile else (WIDTH, 660)
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title desc">'
        ),
        '<title id="title">Sensitivity of the optimizer trading controls</title>',
        (
            '<desc id="desc">Net Sharpe and annual turnover across six trade '
            "coefficients and five holding-rank cutoffs during development. "
            "Thin vertical lines show the range across three rebalance schedules.</desc>"
        ),
        '<g font-family="DejaVu Sans, sans-serif">',
        _text(
            24 if mobile else 53,
            24,
            "Points: means · whiskers: schedule range",
            fill=palette.muted,
            size=18,
        ),
        f'<circle cx="{32 if mobile else 540}" cy="{48 if mobile else 19}" r="6" fill="{palette.selected}"/>',
        _text(
            48 if mobile else 555,
            53 if mobile else 24,
            "Chosen setting",
            fill=palette.muted,
            size=18,
        ),
    ]
    positions = (
        ((8, 72), (8, 650), (8, 357), (8, 935))
        if mobile
        else ((45, 40), (630, 40), (45, 335), (630, 335))
    )
    panels = (
        (ROWS[0], COLUMNS[0]),
        (ROWS[0], COLUMNS[1]),
        (ROWS[1], COLUMNS[0]),
        (ROWS[1], COLUMNS[1]),
    )
    for index, (row, column) in enumerate(panels):
        x0, y0 = positions[index]
        elements.extend(
            _panel(
                frame,
                column=column,
                row=row,
                x0=x0,
                y0=y0,
                width=460 if mobile else 540,
                height=285 if mobile else 300,
                palette=palette,
                show_column_title=index < 2,
            )
        )
    elements.extend(("</g>", "</svg>"))
    return "\n".join(elements) + "\n"


def build_parameter_sensitivity_figure(
    *,
    summary_path: Path = SUMMARY_PATH,
    figure_root: Path = FIGURE_ROOT,
    review_root: Path = REVIEW_ROOT,
) -> dict[str, Path]:
    frame = pl.scan_csv(summary_path).sort("family", "value_order").collect()
    review_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "light": figure_root / "parameter-sensitivity.svg",
        "dark": figure_root / "parameter-sensitivity_dark.svg",
        "mobile_light": figure_root / "parameter-sensitivity_mobile.svg",
        "mobile_dark": figure_root / "parameter-sensitivity_mobile_dark.svg",
        "caption": review_root / "parameter_sensitivity_figure_caption.md",
        "manifest": review_root / "parameter_sensitivity_figure_manifest.json",
    }
    paths["light"].write_text(build_svg(frame, palette=LIGHT), encoding="utf-8")
    paths["dark"].write_text(build_svg(frame, palette=DARK), encoding="utf-8")
    for theme, palette in (("light", LIGHT), ("dark", DARK)):
        paths[f"mobile_{theme}"].write_text(
            build_svg(frame, palette=palette, mobile=True), encoding="utf-8"
        )
    paths["caption"].write_text(
        "**Figure 2. Trading-control sensitivity.** Net "
        "Sharpe and annualized two-way turnover through 2021. The trade-coefficient "
        "group holds the rank cutoff at 175; the rank-cutoff group holds the "
        "coefficient at 0.00025. Points are schedule means; whiskers span the "
        "three rebalance schedules, not confidence intervals.\n",
        encoding="utf-8",
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "display": "Figure 2",
                "question": (
                    "Are the trade coefficient and holding cutoff supported by "
                    "a stable development-period compromise?"
                ),
                "data": os.path.relpath(summary_path, PROJECT_ROOT),
                "files": [
                    os.path.relpath(paths[key], PROJECT_ROOT)
                    for key in ("light", "dark", "mobile_light", "mobile_dark")
                ],
                "observation": (
                    "Trade coefficients from 1 through 3 have similar net Sharpe "
                    "while turnover declines. Holding cutoffs from 150 through "
                    "200 are also close, with only a modest turnover change."
                ),
                "supported_conclusion": (
                    "The selected coefficient of 0.00025 and rank cutoff of 175 "
                    "sit inside broad local plateaus rather than at isolated "
                    "optima."
                ),
                "limitation": ("Each axis varies one choice at a time."),
                "mobile_specific_asset": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def main() -> None:
    for label, path in build_parameter_sensitivity_figure().items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
