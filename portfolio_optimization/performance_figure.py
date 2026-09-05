"""Build the article wealth-and-drawdown figure from matched offset returns."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from portfolio_optimization.svg_primitives import svg_line as _line
from portfolio_optimization.svg_primitives import svg_text as _text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = PROJECT_ROOT / "outputs" / "review"
FIGURE_ROOT = REVIEW_ROOT / "figures"
COMMON_START = date(1998, 9, 22)
DEVELOPMENT_END = date(2021, 12, 31)
WIDTH = 1200
HEIGHT = 780
WEALTH_AXIS_MAX = 16.0
DRAWDOWN_AXIS_MIN = -22.0


ALLOCATORS = ("b1", "b2", "b3")


@dataclass(frozen=True)
class Palette:
    text: str
    title: str
    muted: str
    grid: str
    b1: str
    b2: str
    b3: str


LIGHT = Palette(
    text="#172033",
    title="#000000",
    muted="#667085",
    grid="#D9DEE8",
    b1="#9AA4B2",
    b2="#4F7396",
    b3="#B98556",
)
DARK = Palette(
    text="#F3F4F6",
    title="#F3F4F6",
    muted="#AAB2C0",
    grid="#3B4250",
    b1="#AAB2C0",
    b2="#78A0C4",
    b3="#C79261",
)


def _validate(frame: pl.DataFrame) -> None:
    expected_allocators = set(ALLOCATORS)
    if set(frame.get_column("allocator")) != expected_allocators:
        raise ValueError("performance path requires the declared B1/B2/B3 allocators")
    counts = frame.group_by("allocator").agg(
        pl.len().alias("rows"),
        pl.col("date").min().alias("start"),
        pl.col("date").max().alias("end"),
    )
    if (
        counts.height != len(ALLOCATORS)
        or counts.get_column("rows").n_unique() != 1
        or counts.get_column("start").n_unique() != 1
        or counts.get_column("end").n_unique() != 1
    ):
        raise ValueError("performance paths do not share one matched calendar")
    if (
        frame.select("allocator", "date").n_unique() != frame.height
        or frame.group_by("date")
        .agg(pl.col("allocator").n_unique().alias("allocators"))
        .filter(pl.col("allocators") != len(ALLOCATORS))
        .height
    ):
        raise ValueError("performance paths do not share one matched calendar")
    if frame.filter(
        pl.any_horizontal(
            pl.col(column).is_null()
            | pl.col(column).is_nan()
            | pl.col(column).is_infinite()
            for column in ("net_return", "wealth", "drawdown_pct")
        )
        | pl.col("date").is_null()
        | (pl.col("net_return") <= -1)
        | (pl.col("wealth") <= 0)
        | (pl.col("drawdown_pct") > 1e-9)
    ).height:
        raise ValueError("performance path contains invalid financial values")
    if frame.filter(
        (pl.col("wealth") > WEALTH_AXIS_MAX)
        | (pl.col("drawdown_pct") < DRAWDOWN_AXIS_MIN)
    ).height:
        raise ValueError("performance path falls outside the fixed figure axes")


def _path(
    points: list[tuple[float, float]],
    color: str,
    width: float,
    *,
    opacity: float = 1.0,
) -> str:
    commands = " ".join(
        f"{'M' if index == 0 else 'L'}{x:.1f},{y:.1f}"
        for index, (x, y) in enumerate(points)
    )
    return (
        f'<path d="{commands}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-opacity="{opacity:.2f}" '
        'stroke-linecap="round" stroke-linejoin="round"/>'
    )


def _area_path(
    points: list[tuple[float, float]],
    *,
    baseline: float,
    color: str,
    opacity: float,
) -> str:
    commands = " ".join(f"L{x:.1f},{y:.1f}" for x, y in points)
    first_x, last_x = points[0][0], points[-1][0]
    closed = f"M{first_x:.1f},{baseline:.1f} {commands} L{last_x:.1f},{baseline:.1f} Z"
    return (
        f'<path d="{closed}" fill="{color}" fill-opacity="{opacity:.2f}" '
        'stroke="none"/>'
    )


def build_svg(frame: pl.DataFrame, *, palette: Palette, mobile: bool = False) -> str:
    """Render daily matched wealth and drawdown paths as one article figure."""

    _validate(frame)
    width, height = (480, 680) if mobile else (WIDTH, HEIGHT)
    left = 55.0 if mobile else 90.0
    right = 455.0 if mobile else 1140.0
    wealth_top = 165.0 if mobile else 90.0
    wealth_bottom = 410.0 if mobile else 445.0
    drawdown_top = 485.0 if mobile else 525.0
    drawdown_bottom = 640.0 if mobile else 735.0
    start_value = frame.get_column("date").min()
    end_value = frame.get_column("date").max()
    if not isinstance(start_value, date) or not isinstance(end_value, date):
        raise TypeError("performance path lacks valid date boundaries")
    start = start_value
    end = end_value
    if start >= end:
        raise ValueError("performance display requires at least two distinct dates")
    total_days = (end - start).days

    def x_position(value: date) -> float:
        return left + (value - start).days / total_days * (right - left)

    def wealth_y(value: float) -> float:
        return wealth_bottom - math.log2(value) / math.log2(WEALTH_AXIS_MAX) * (
            wealth_bottom - wealth_top
        )

    def drawdown_y(value: float) -> float:
        return drawdown_top + (-value) / -DRAWDOWN_AXIS_MIN * (
            drawdown_bottom - drawdown_top
        )

    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        ('<g font-family="DejaVu Sans, sans-serif">'),
        _line(left, 24, left + 30, 24, stroke=palette.b1, stroke_width="3"),
        _text(left + 40, 29, "Volatility-scaled", fill=palette.text, size=18),
        _line(
            left if mobile else 340,
            52 if mobile else 24,
            left + 30 if mobile else 370,
            52 if mobile else 24,
            stroke=palette.b2,
            stroke_width="3",
        ),
        _text(
            left + 40 if mobile else 380,
            57 if mobile else 29,
            "Optimizer",
            fill=palette.text,
            size=18,
        ),
        _line(
            left if mobile else 550,
            80 if mobile else 24,
            left + 30 if mobile else 580,
            80 if mobile else 24,
            stroke=palette.b3,
            stroke_width="3",
        ),
        _text(
            left + 40 if mobile else 590,
            85 if mobile else 29,
            "Optimizer + trading controls",
            fill=palette.text,
            size=18,
        ),
        _text(
            left if mobile else right,
            118 if mobile else 65,
            "Development · 1998–2021",
            fill=palette.muted,
            size=18,
            anchor="start" if mobile else "end",
        ),
        _text(
            left,
            wealth_top - 15,
            "Net growth index (log scale)",
            fill=palette.title,
            size=20,
            weight=600,
        ),
        _text(
            left,
            drawdown_top - 15,
            "Drawdown (%)",
            fill=palette.title,
            size=20,
            weight=600,
        ),
    ]
    for tick in (1.0, 2.0, 4.0, 8.0, 16.0):
        y = wealth_y(tick)
        elements.extend(
            [
                _line(left, y, right, y, stroke=palette.grid, stroke_width="1"),
                _text(
                    left - 12,
                    y + 5,
                    f"{tick:.0f}×",
                    fill=palette.muted,
                    size=18,
                    anchor="end",
                ),
            ]
        )
    for tick in (0.0, -10.0, -20.0):
        y = drawdown_y(tick)
        elements.extend(
            [
                _line(left, y, right, y, stroke=palette.grid, stroke_width="1"),
                _text(
                    left - 12,
                    y + 5,
                    f"{tick:.0f}",
                    fill=palette.muted,
                    size=18,
                    anchor="end",
                ),
            ]
        )
    for year in (2000, 2010, 2020) if mobile else (2000, 2004, 2008, 2012, 2016, 2020):
        x = x_position(date(year, 1, 1))
        elements.append(
            _text(
                x,
                height - 17,
                str(year),
                fill=palette.muted,
                size=18,
                anchor="middle",
            )
        )
    colors = {"b1": palette.b1, "b2": palette.b2, "b3": palette.b3}
    for allocator in ("b1", "b2", "b3"):
        rows = frame.filter(pl.col("allocator") == allocator).sort("date")
        wealth_points = [
            (x_position(value_date), wealth_y(float(value)))
            for value_date, value in rows.select("date", "wealth").iter_rows()
        ]
        drawdown_points = [
            (x_position(value_date), drawdown_y(float(value)))
            for value_date, value in rows.select("date", "drawdown_pct").iter_rows()
        ]
        width = 2.0 if allocator == "b1" else 2.5
        elements.extend(
            [
                _path(wealth_points, colors[allocator], width),
                _area_path(
                    drawdown_points,
                    baseline=drawdown_y(0.0),
                    color=colors[allocator],
                    opacity=0.06 if allocator == "b1" else 0.0,
                ),
                _path(
                    drawdown_points,
                    colors[allocator],
                    1.8 if allocator == "b1" else 2.1,
                    opacity=0.75 if allocator == "b1" else 0.90,
                ),
            ]
        )
    elements.extend(["</g>", "</svg>"])
    return "\n".join(elements) + "\n"


def build_performance_figure(
    *,
    review_root: Path = REVIEW_ROOT,
    figure_root: Path = FIGURE_ROOT,
    daily_source: Path = REVIEW_ROOT / "performance_path_daily.parquet",
) -> dict[str, Path]:
    """Persist daily evidence, light/dark SVGs, caption, and manifest."""

    frame = (
        pl.scan_parquet(daily_source)
        .filter(pl.col("date").is_between(COMMON_START, DEVELOPMENT_END))
        .sort("allocator", "date")
        .collect()
    )
    _validate(frame)
    source_hash = hashlib.sha256(daily_source.read_bytes()).hexdigest()
    review_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "data": review_root / "performance_path_daily.parquet",
        "light": figure_root / "performance-and-drawdowns.svg",
        "dark": figure_root / "performance-and-drawdowns_dark.svg",
        "mobile_light": figure_root / "performance-and-drawdowns_mobile.svg",
        "mobile_dark": figure_root / "performance-and-drawdowns_mobile_dark.svg",
        "caption": review_root / "performance_figure_caption.md",
        "manifest": review_root / "performance_figure_manifest.json",
    }
    frame.write_parquet(paths["data"], compression="zstd")
    paths["light"].write_text(build_svg(frame, palette=LIGHT), encoding="utf-8")
    paths["dark"].write_text(build_svg(frame, palette=DARK), encoding="utf-8")
    for theme, palette in (("light", LIGHT), ("dark", DARK)):
        paths[f"mobile_{theme}"].write_text(
            build_svg(frame, palette=palette, mobile=True), encoding="utf-8"
        )
    caption = (
        "**Figure 1. Net performance and drawdowns through time.** Daily net "
        "returns charge 5 bp per side. Each rebalance-schedule path is "
        "compounded separately, then the three wealth levels are averaged once "
        "all offsets are live, from 22 September 1998 "
        "through 31 December 2021. This path shows timing and compounding; Table 1 "
        "reports the mean of the three schedule-level metrics. The strategies "
        "retain different volatilities; cumulative growth is not a risk-matched comparison."
    )
    paths["caption"].write_text(caption + "\n", encoding="utf-8")
    manifest = {
        "display": "Figure 1",
        "question": ("How do the three allocation rules compound during development?"),
        "data": str(paths["data"].relative_to(PROJECT_ROOT))
        if paths["data"].is_relative_to(PROJECT_ROOT)
        else str(paths["data"]),
        "compact_input_sha256": source_hash,
        "files": [
            str(paths[key].relative_to(PROJECT_ROOT))
            if paths[key].is_relative_to(PROJECT_ROOT)
            else str(paths[key])
            for key in ("light", "dark", "mobile_light", "mobile_dark")
        ],
        "observation": (
            "The optimizer finishes above volatility scaling. Adding trading "
            "controls finishes highest and has the smallest major drawdown."
        ),
        "supported_conclusion": (
            "Joint sizing improves development-period performance, while the "
            "trading controls preserve that allocation with fewer trades."
        ),
        "limitation": (
            "The path averages three separately compounded schedule wealth levels; "
            "the headline table averages metrics calculated within each schedule."
        ),
        "mobile_specific_asset": True,
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--daily-source",
        type=Path,
        default=REVIEW_ROOT / "performance_path_daily.parquet",
        help="Regenerate from retained daily evidence without raw backtest trees.",
    )
    arguments = parser.parse_args()
    for name, path in build_performance_figure(
        daily_source=arguments.daily_source
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
