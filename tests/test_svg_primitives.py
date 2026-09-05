from portfolio_optimization.svg_primitives import svg_line, svg_text


def test_svg_text_escapes_content_and_supports_rotation() -> None:
    rendered = svg_text(
        1,
        2,
        "A & B",
        fill="#000",
        size=12,
        anchor="middle",
        weight=600,
        rotate=-90,
    )

    assert rendered == (
        '<text x="1.0" y="2.0" fill="#000" font-size="12" font-weight="600" '
        'text-anchor="middle" transform="rotate(-90 1.0 2.0)">A &amp; B</text>'
    )


def test_svg_line_translates_python_attribute_names() -> None:
    assert svg_line(1, 2, 3, 4, stroke_width=2, stroke="#fff") == (
        '<line x1="1.0" y1="2.0" x2="3.0" y2="4.0" stroke-width="2" stroke="#fff"/>'
    )
