# Photo-informed drone parking target

The static quadcopter recreates the supplied seven photos for UGV perception and drive-under parking work. It includes four carbon booms, motor brackets and cans, shaped two-blade propellers, stacked center plates, visible electronics and wiring, a GPS mast, suspended payload, exactly two splayed legs (one per skid), and foam landing skids with red rings. Dimensions are estimates, rather than measurements of the real drone.

## Interfaces

`drone_model.drone_assembly()` returns a flat list of part dictionaries with `name`, `shape`, `pos`, `size`, `rot`, `color`, and `collision`. Shapes are `box`, `cylinder`, and `mesh`; mesh parts additionally contain `mesh`, a path relative to the Gazebo directory. `DRONE_COLORS` maps material names to RGB tuples. Coordinates use metres and radians, with X along the skids, Y left, Z up, and the origin on the floor at the drone center.

`add_drone(world, node, pose=(.8,0,0,0,0,0))` inserts a static SDF model named `photo_quadcopter`. `node` follows the XML helper interface in `generate.py`. Calling the assembly generator also deterministically creates `assets/drone/propeller.obj`; the SDF uses its portable relative URI.

## Built dimensions

| Feature | Estimated model dimension |
| --- | --- |
| Skid end-to-end length | 480 mm, including endcaps |
| Skid collision core length | 470 mm |
| Foam sleeve span | 457.5 mm |
| Skid center spacing / outer foam radius | 340 mm / 15 mm |
| Clear gap between skids at floor level | 310 mm |
| Center plate diameter / plate center heights | 235 mm / 410 and 438 mm |
| Motor center radius | 360 mm |
| Propeller diameter | 380 mm |
| GPS top | 694 mm above floor |
| Lowest suspended solid payload collision | 225 mm above floor |
| Overhead marker card bottom | 168 mm above floor, visual only |
| Loose servo cable bottom | approximately 195 mm, visual only |

Each skid meets a single splayed tube at its center (local X=0); the two tubes converge toward the central deck. There are no fore/aft duplicate legs. Landing legs narrow the corridor with height. The JSON records a conservative approximately 255 mm center gap around 200 mm above the floor. The UGV's lower driving envelope has more lateral room. Individual tubes, skids, plates, motor cans, and payload blocks supply collisions; there is no filled box through the landing gear corridor. Marker cards, fine wiring, propeller blades, and other fine appearance details are visual only. These estimates support scene development and are not physical clearance proof for the real vehicles.

`drone_dimensions.json` describes the authored geometry. Only `skid_center_spacing` and `deck_diameter` currently directly control geometry; detailed coordinates and other dimensions are authored in `drone_model.py`. Changing metadata alone does not rescale the assembly.

## Fiducials and simulation scope

Four separate physical cards carry actual `DICT_4X4_50` markers: IDs **9** and **12** on the two landing legs, plus IDs **0** and **1** on opposite X sides of the suspended platform. Each has a 50 mm black square including its one-cell black border, plus a 9 mm white quiet zone on each side (68 mm card). Each card has one printed outward face. Payload ID 0 points toward local -X along the approach direction, and payload ID 1 points toward local +X. The two leg cards sit near the bases of the two central leg tubes. The literal grids were checked with OpenCV's ArUco detector; OpenCV is not required to build or load the model.

This is a stationary obstacle and perception target. It has no aerial dynamics or spinning propellers. The car geometry and drive dynamics come from the existing UGV modules; the drone scene itself does not add autonomous driving.

Validation performed: Python compilation, successful assembly construction, standalone SDF validation with `gz sdf -k`, and detection of all four literal ArUco grids. Appearance and scene-level checks are performed separately through the docking world and Blender previews.
