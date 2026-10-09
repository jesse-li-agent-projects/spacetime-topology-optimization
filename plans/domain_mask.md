# Non-rectangular design domains for `stto`

Give `stto` a design domain short of its mesh, and an L-bracket load case for the
validation sweep. Ask the user before any design decision this plan does not make.

## L bracket (user, 2026-10-09)

An L tromino: the bounding box without its top-right quarter.
- Supports: the right edge of the bottom half is clamped.
- Load: horizontal, +x (the sign does not change a compliance optimum), spread over
  `load_length_m` down the vertical edge at either the top-left corner
  (`l_bracket_corner`, A) or the top of the arm's right edge (`l_bracket_middle`, B).
- Print base: the clamped edge (`right_edge`); printed right to left.
- Validation: 120x120 box, 1 mm elements (10 800 active), volfrac 0.5, both load points.

```
  A----B
  |    |
  |    +-----+
  |          |#
  |          |#  # = clamped
  +----------+#
```

## Design (implemented)

- **The load case defines the domain** (`load_cases.active_mask`), so a config cannot
  pair an L load with a rectangle. Every older case uses the whole mesh.
- **Outside the domain is void.** `xPhys = 0` there, and the density filter reads it
  as void (user: padded, not truncated at the mask edge as at the mesh edge). No row's
  gradient reaches the density variables outside.
- `volfrac` is a fraction of the domain; so are the volume and grey fractions of the
  design checks.
- The time filter averages over the domain only, which extends `t` to the void next to
  it, where the time-field gradient reads it. So a domain short of the mesh needs
  `time_filter_rmin_m > 0`.
- The print base is the time field's base elements inside the domain.
- Refused on a domain short of the mesh: the stage volume bounds (deprecated, measured
  against the whole mesh) and the virtual-heat variants.

## Open

- L cells in the validation sweep.
