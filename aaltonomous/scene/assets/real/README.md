# Organiser-supplied geometry

`lid_shape_holes_200mm.stl` (received 2026-11-14, mm, printed plate-down). Measured from the mesh:

- plate 200 x 200 mm, 4 mm thick; a locating rim 185.6-191.6 mm square (3 mm wall), 12 mm tall from the plate bottom
- holes, all 60 mm across (objects are 45 mm -> 15 mm clearance, 7.5 mm per side):
  - circle d 60 at (+48, -35)
  - square 60 x 60 (rounded corners) at (-48, -35)
  - triangle 60 wide x 54.6 tall at (0, +45)
- 4 small corner features (3.5 mm) on the rim

Orientation on the real bin (which edge faces the robot) still has to be checked at the venue. The sim lid
(`../lid_shapes.stl`, generated from scene.yaml) is NOT this part yet: 210 mm, 3 mm, 10 mm clearance, holes
near the far edge.
