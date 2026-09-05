"""Small, pure SVG primitives shared by the article figure builders."""

from __future__ import annotations

import html


def svg_text(
    x: float,
    y: float,
    value: str,
    *,
    fill: str,
    size: int,
    anchor: str = "start",
    weight: int = 400,
    rotate: int | None = None,
) -> str:
    """Return an escaped SVG text element with stable numeric formatting."""
    transform = (
        "" if rotate is None else f' transform="rotate({rotate} {x:.1f} {y:.1f})"'
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}"{transform}>'
        f"{html.escape(value)}</text>"
    )


def svg_line(x1: float, y1: float, x2: float, y2: float, **attrs: object) -> str:
    """Return an SVG line element with CSS-style attribute names."""
    attributes = " ".join(
        f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items()
    )
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {attributes}/>'
    )
