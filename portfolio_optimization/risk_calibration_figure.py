"""Render the article's compact realised-beta display."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from portfolio_optimization.svg_primitives import svg_line as _line
from portfolio_optimization.svg_primitives import svg_text as _text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = PROJECT_ROOT / "outputs" / "review"
BETA_PATH = REVIEW_ROOT / "rolling_realised_beta_252d.csv"
FIGURE_ROOT = REVIEW_ROOT / "figures"
WIDTH = 1200
HEIGHT = 520
DEVELOPMENT_END = date(2021, 12, 31)


@dataclass(frozen=True)
class Palette:
    text: str
    muted: str
    grid: str
    b1: str
    b2: str
    b3: str


LIGHT = Palette(
    text="#172033",
    muted="#667085",
    grid="#D9DEE8",
    b1="#64748B",
    b2="#2563EB",
    b3="#D97706",
)
DARK = Palette(
    text="#F3F4F6",
    muted="#AAB2C0",
    grid="#3B4250",
    b1="#CBD5E1",
    b2="#60A5FA",
    b3="#FBBF24",
)

ALLOCATORS = (
    ("b1_ranked_volscale", "Volatility-scaled rule", "b1", "2 5"),
    ("b2_memoryless_mvo", "Optimizer", "b2", "8 5"),
    ("b3_state_aware_mvo", "Optimizer + trading controls", "b3", None),
)
EXPECTED_ALLOCATORS = {row[0] for row in ALLOCATORS}


def monthly_mean_beta(beta: pl.LazyFrame) -> pl.DataFrame:
    """Average schedules daily, then retain the last observation of each month."""

    return (
        beta.filter(
            pl.col("allocator").is_in(EXPECTED_ALLOCATORS)
            & (pl.col("date") <= DEVELOPMENT_END)
        )
        .group_by("allocator", "allocator_label", "date")
        .agg(pl.col("realised_beta_252d").mean())
        .sort("allocator", "date")
        .with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("allocator", "allocator_label", "month", maintain_order=True)
        .agg(pl.col("date").last(), pl.col("realised_beta_252d").last())
        .sort("allocator", "date")
        .collect()
    )


def _validate(beta: pl.DataFrame) -> None:
    if set(beta.get_column("allocator")) != EXPECTED_ALLOCATORS:
        raise ValueError("beta display requires the declared B1, B2, and B3 histories")
    if (
        beta.select("allocator", "date").n_unique() != beta.height
        or beta.group_by("date")
        .agg(pl.col("allocator").n_unique().alias("allocators"))
        .filter(pl.col("allocators") != len(EXPECTED_ALLOCATORS))
        .height
    ):
        raise ValueError("beta histories do not share one matched calendar")
    if beta.filter(
        pl.col("date").is_null()
        | pl.col("realised_beta_252d").is_null()
        | ~pl.col("realised_beta_252d").is_finite()
        | (pl.col("realised_beta_252d") < -0.20)
        | (pl.col("realised_beta_252d") > 0.40)
    ).height:
        raise ValueError("realised-beta values fall outside the fixed display axis")


def build_svg(beta: pl.DataFrame, *, palette: Palette) -> str:
    """Render the three portfolio beta paths without in-plot annotations."""

    _validate(beta)
    minimum, maximum = -0.20, 0.40

    left, right, top, bottom = 95.0, 1115.0, 110.0, 435.0
    start = beta.get_column("date").min()
    end = beta.get_column("date").max()
    if not isinstance(start, date) or not isinstance(end, date):
        raise TypeError("beta dates must be parsed dates")
    if start >= end:
        raise ValueError("beta display requires at least two distinct dates")
    start_ordinal, end_ordinal = start.toordinal(), end.toordinal()

    def x_position(value: date) -> float:
        return left + (value.toordinal() - start_ordinal) / (
            end_ordinal - start_ordinal
        ) * (right - left)

    def y_position(value: float) -> float:
        return bottom - (value - minimum) / (maximum - minimum) * (bottom - top)

    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
            'aria-labelledby="title desc">'
        ),
        '<title id="title">Trailing realized market beta</title>',
        (
            '<desc id="desc">Monthly trailing 252-day realized beta for the '
            "volatility-scaled rule and both optimizers, shown against a zero "
            "reference line.</desc>"
        ),
        (
            '<g font-family="Inter, ui-sans-serif, -apple-system, '
            'BlinkMacSystemFont, Segoe UI, sans-serif">'
        ),
    ]

    for tick in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4):
        y = y_position(tick)
        elements.extend(
            [
                _line(
                    left,
                    y,
                    right,
                    y,
                    stroke=palette.muted if tick == 0.0 else palette.grid,
                    stroke_width="1.4" if tick == 0.0 else "1",
                ),
                _text(
                    left - 12,
                    y + 5,
                    f"{tick:.1f}",
                    fill=palette.muted,
                    size=18,
                    anchor="end",
                ),
            ]
        )

    for year in (2000, 2005, 2010, 2015, 2020, 2025):
        tick_date = date(year, 1, 1)
        if start <= tick_date <= end:
            elements.append(
                _text(
                    x_position(tick_date),
                    bottom + 25,
                    str(year),
                    fill=palette.muted,
                    size=18,
                    anchor="middle",
                )
            )

    elements.append(
        _text(
            left,
            34,
            "Trailing 252-day realized beta",
            fill=palette.text,
            size=23,
            weight=650,
        )
    )

    legend_positions = ((95.0, 60.0), (350.0, 60.0), (95.0, 84.0))
    for index, (allocator, label, color_name, dash) in enumerate(ALLOCATORS):
        color = getattr(palette, color_name)
        x0, y0 = legend_positions[index]
        dash_attribute = "" if dash is None else f' stroke-dasharray="{dash}"'
        elements.extend(
            [
                _line(
                    x0,
                    y0,
                    x0 + 32,
                    y0,
                    **(
                        {
                            "stroke": color,
                            "stroke_width": "3",
                            "stroke_dasharray": dash,
                        }
                        if dash is not None
                        else {"stroke": color, "stroke_width": "3"}
                    ),
                ),
                _text(
                    x0 + 42,
                    y0 + 5,
                    label,
                    fill=palette.text,
                    size=18,
                    weight=600,
                ),
            ]
        )
        rows = beta.filter(pl.col("allocator") == allocator).sort("date")
        commands = " ".join(
            f"{'M' if point_index == 0 else 'L'}{x_position(row_date):.1f},{y_position(float(value)):.1f}"
            for point_index, (row_date, value) in enumerate(
                rows.select("date", "realised_beta_252d").iter_rows()
            )
        )
        elements.append(
            f'<path d="{commands}" fill="none" stroke="{color}" '
            f'stroke-width="2.2" stroke-linejoin="round"{dash_attribute}/>'
        )

    elements.extend(
        [
            _text(
                (left + right) / 2,
                HEIGHT - 15,
                "Date",
                fill=palette.muted,
                size=18,
                anchor="middle",
            ),
            "</g>",
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def build_risk_calibration_figure(
    *,
    beta_path: Path = BETA_PATH,
    figure_root: Path = FIGURE_ROOT,
    review_root: Path = REVIEW_ROOT,
) -> dict[str, Path]:
    """Persist light/dark SVGs, caption, and one authoritative manifest."""

    beta = monthly_mean_beta(pl.scan_csv(beta_path, try_parse_dates=True))
    review_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "light": figure_root / "risk-calibration-and-beta.svg",
        "dark": figure_root / "risk-calibration-and-beta_dark.svg",
        "caption": review_root / "risk_calibration_figure_caption.md",
        "manifest": review_root / "risk_calibration_figure_manifest.json",
    }
    paths["light"].write_text(build_svg(beta, palette=LIGHT), encoding="utf-8")
    paths["dark"].write_text(build_svg(beta, palette=DARK), encoding="utf-8")
    paths["caption"].write_text(
        "**Figure 4. Realized market beta through time.** Trailing 252-day "
        "portfolio beta for the volatility-scaled rule and both optimizers, "
        "sampled monthly through 2021 and averaged across the three "
        "rebalance schedules. "
        "The point-in-time optimizer constraint uses a different beta estimate "
        "and clock, so its target band is not overlaid on this slow outcome "
        "measure.\n",
        encoding="utf-8",
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "display": "Figure 4",
                "question": (
                    "How large and persistent is realised market beta after "
                    "portfolio formation?"
                ),
                "data": os.path.relpath(beta_path, PROJECT_ROOT),
                "files": [
                    os.path.relpath(paths["light"], PROJECT_ROOT),
                    os.path.relpath(paths["dark"], PROJECT_ROOT),
                ],
                "observation": (
                    "Through 2021, all three rules carry persistent realised "
                    "beta at times; "
                    "the optimizers reduce, but do not remove, that exposure."
                ),
                "supported_conclusion": (
                    "Joint optimization improves persistent beta relative to "
                    "volatility scaling, but the point-in-time constraint does "
                    "not keep trailing realised beta close to zero."
                ),
                "limitation": (
                    "The 252-day outcome measure has long memory and is not the "
                    "point-in-time beta estimate constrained at a rebalance."
                ),
                "excluded_supporting_evidence": [
                    "volatility calibration",
                    "raw rebalance-level beta scatter",
                    "beta-estimator comparison",
                    "beta-window rerun",
                ],
                "mobile_specific_asset": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def main() -> None:
    for name, path in build_risk_calibration_figure().items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
