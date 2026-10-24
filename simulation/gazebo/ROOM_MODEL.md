# Photo-informed robotics teaching laboratory

The Gazebo room is reconstructed from the user's approximately 20 second slow panorama and the drone photographs. The user supplied an approximate **12 × 24 m** room size. This is a recognizable scene reconstruction, not a measured digital survey.

## Geometry and layout

- World X is the long 24 m axis, Y the 12 m axis, Z upward. The concrete surface is at Z = 0, bounded by X ±12 m and Y ±6 m.
- The central 6 × 4 m patch is left open for the UGV and stationary drone demonstration. No navigation controller or autonomous driving behavior is added by the room model.
- The north wall has a long framed whiteboard run, faint formula strokes, marker trays, low white cupboards, wall conduit, sockets, and a teaching table/chair group.
- The south long wall has tall frosted window/blind panels, narrow mullions, pull cords, structural piers and two grey exit doors with bars and green exit signs.
- The northeast corner has black lockers and the east wall has, computer workstations, dual monitors and office chairs. Shop islands, grey storage cabinets, shelving, colored bins, red rolling tool chests, cardboard cartons and spare boards occupy the other perimeter zones.
- The Canadian flag partition, curved red and black sled chairs, pale wood table tops and dark metal legs reproduce particularly recognizable objects from the panorama.
- A wheeled whiteboard, wooden wing test frame, metal survey tripod, aircraft parts and workshop clutter provide intermediate scale references without filling the demonstration floor.
- The ceiling is estimated at 3.8 m and has exposed beams, cylindrical duct runs/collars, hanging linear fixtures, circular diffusers and six interior point lights. `category='ceiling'` makes these removable in presentation cutaways; they remain present in Gazebo.

Table tops are approximately 0.75 m high, normal chairs approximately 0.45 m at the seat, and the whiteboards approximately 1.24 m high. Most furniture sizes, spacings, door positions and room orientation are inferred. The panorama does not reveal every wall boundary clearly, and using the full requested 24 × 12 m envelope requires extrapolating the distant perimeter. No people are modeled. Tiny partly obscured objects are represented as equipment clutter rather than claims of exact identity.

## Reuse

`room_model.room_assembly()` returns flat part dictionaries compatible with the car/drone render pipeline: `name`, `shape`, `size`, `pos`, `rot`, `color`, `collision`; optional `mesh` and `category`. Colors are in `ROOM_COLORS`. `room_furniture.furniture_assembly()` supplies the detailed furniture and `FURNITURE_COLORS`.

`room_model.add_room(world, node)` inserts a static finite room and lights into SDF. Hard furniture includes separate local collision shapes, allowing driving under desks and between chair legs. Visual surface wear, paper, screens, cupboard handles, thin cables, artwork and other ornamental details have no collision. The room floor is a finite box, not an infinite plane. The room uses no downloaded assets or external Python packages.

Procedural OBJ assets live in `assets/room/` and are generated deterministically when the assembly is built. They use metres. `room_dimensions.json` records the scale assumptions. To produce the combined room scene, use the repository's generator with `--room` and the existing drone option. Preview render integration is handled by the shared scene renderer.

## Validation

The room and furniture modules compile, parts have unique names, all material keys resolve, and the central demonstration area has no furniture collision centers. Final scene inspection and Gazebo validation are recorded by the parent integration task.

## Panorama comparison refinement

The first room renders were compared directly with extracted video frames at 3 s (teaching), 12 s (glazing/test stand) and 16 s (workshop/flag). The continuous sweep establishes the approximate clockwise order: whiteboards, computing/lockers, exits, opposite glazing, fabrication, flag and back to teaching tables. The glazing was moved to the south **long** wall opposite the north teaching wall; computing moved to the east wall, lockers to the northeast and fabrication to the southwest. This corrects the first exploratory layout, which placed the glazing on a short wall. Two whiteboard/cabinet modules avoid overlap with the northeast lockers.

The inferred stationary phone camera is `(2, 2, 1.62)` metres, with teaching target `(3.5, 5.9, 1.55)`, glazing target `(0, -6, 1.35)` and workshop target `(-8, -1.5, 1.3)`. These are comparison viewpoints, not recovered calibration. Near-camera open laptops, a power strip/cable and backpack, workshop stools, folded tables and stacked chairs improve the relative furniture density seen in the source. The wing test stand and tripod now sit against the south glazing.

Source materials were darkened to medium grey concrete, warm grey walls and a darker industrial ceiling after the first interior render appeared overexposed. This correction also applies to the Gazebo materials; the rendering script additionally adjusts illumination. Room size and unobserved space remain estimates.

## User direction correction and missing-object pass

The user's explicit anchors supersede the earlier inferred cardinal layout: 4s is the east long teaching wall, 8s is the south short end, 11s the west long glazed side, and 17s the north back/workshop. World X now runs east/west across 12m and Y north/south along 24m. Glazing is L-shaped across both south and west. The comparison camera is at (2,-2,1.62), and the drone foreground position and skid angle are estimated from the south-facing8s frame (room_dimensions.json).

The wheeled whiteboard sits behind the drone at (1.8,-6.2), with the south drawer cabinet/gantry to its east and meeting tables to its west. The large teaching-table cluster is offset to the left of the 4s view rather than covering its full foreground; board/cabinet widths are revised to approximately 2.45m each. New modules room_details.py and room_south_details.py add south stored tables/chairs, tray racks, a drawer bench/gantry, a steering-wheel workstation, five narrow black lockers, folding aircraft stands, north workbenches, storage/vacuum equipment, a delta wing, and two adjacent flag-divider panels. ROOM_DETAILS.md records the full per-second inventory.

The protected driving route now follows the actual drone approach, X 1.55..2.05m and Y -4.9..-0.5m, rather than the original arbitrary central rectangle. The room footprint remains the user's approximate 12 by 24m; object depths and camera translation remain estimates.
