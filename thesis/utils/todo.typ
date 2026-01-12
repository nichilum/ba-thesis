#let TODO(body, color: red.lighten(50%), width: 100%, breakable: true) = {
    block(
      width: width,
      radius: 3pt,
      stroke: 0.5pt,
      fill: color,
      inset: 10pt,
      breakable: breakable,
    )[
      #body
    ]
}
