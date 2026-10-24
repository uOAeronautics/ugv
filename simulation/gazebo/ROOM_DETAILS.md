# Video anchor inventory and second detail pass

`room_details.py` and `room_south_details.py` use the corrected **cardinal coordinates** directly: X east/west ±6 m, Y north/south ±12 m. They must be appended after any legacy room rotation. `room_details()` returns the standard flat parts schema and `DETAIL_COLORS` supplies its materials. These modules do not modify the room shell. Meshes use atomic replacement and skip unchanged content.

The user supplies the directional anchors: **4 s east whiteboard long side; 8 s south short end; 11 s west window long side; 17 s north/back short end.** Both the south end and west long wall have frosted glazing, forming an L. The first inferred azimuth assignment was wrong and was superseded by these anchors.

## Observed panorama inventory

Times identify approximate one-second samples, not calibrated survey bearings. Objects reappear in overlapping frames and are not counted repeatedly.

| Second | Visible objects / placement evidence |
|---|---|
| 0 | Two adjacent grey framed cork room dividers, Canadian flag on left, blue lettering on right, yellow clips along tops; tall red narrow cabinet at left; boxes, bucket and bags underneath; laptop table foreground. |
| 1 | Divider at left, two joined pale teaching tables, one open laptop, power adapters, leads, pens, tape roll; red sled chairs; small metal wire equipment cart against east wall. |
| 2 | First east whiteboard, approximately four red chairs around teaching tables, wall conduit/outlet, low cabinet begins at right. |
| 3 | Two adjacent whiteboards with faint multicolor notes, marker trays; white cupboard with dark grey top and four doors; loose floor cable. |
| 4 | Second whiteboard and cupboard, conduit rising toward ceiling, desktop monitor station appears to right. |
| 5 | East station: two screens, steering-wheel/controller, pale desk with grey T feet and a black chair; black narrow locker bank behind. |
| 6 | Drawer work island with substantial wood top, tall grey steel overhead gantry, two cardboard boxes, coiled black lead; east lockers and computing behind. |
| 7 | South frosted glazed end; grey exit door left, wheeled whiteboard in front of stored pale tables and white chairs; folded black table frames upright. Drone enters foreground. |
| 8 | South anchor: wheeled whiteboard centered, island left, stored tables/chairs behind; second exit right; black equipment shelving and many grey trays near right corner; trash bin. Drone clearly foreground. |
| 9 | South-west transition: pale meeting table group with black sled chairs, black chair stacks and red chairs; grey exit beside frosted glazing. |
| 10 | West meeting tables with electronics and paper; at least three black chairs on near edge and two red chairs at far end; structural window pier. |
| 11 | West long-side anchor: wooden skeletal wing test frame, survey tripod/instrument, tall frosted glazing with hanging blind cords; meeting table group to left. |
| 12 | Test stand and tripod centered; folding instrument table with black equipment crate to right; red X aircraft stand; rolling shop stool. |
| 13 | West workshop transition: red X stand and small silver aircraft, black work table, larger metallic aircraft/airframe on table, display board, long white foam stock. |
| 14 | Red rolling tool chest with multiple drawers; white foam panels leaning against black work table; rear computing benches appear; blue delta aircraft on back wall. |
| 15 | Back work area: two adjoining computer/electronics benches, monitors, rolling stools, task equipment, leaning wood stock; open wood shelving on right. People are intentionally excluded. |
| 16 | North/back anchor: back benches, large blue delta wing on wall, spare white wing pieces, open wooden shelf, orange shop vacuum, white/black containers; red narrow cabinet next to divider. |
| 17 | North/back and divider: two divider panels clearly in front of rear wall, flag left panel, yellow clips, under-panel storage; foreground red chair and dark backpack next to teaching table. |
| 18 | Return toward start: flag divider, red chair, backpack, laptop table with adapters/wires; a second red chair enters. |
| 19 | Camera tilts downward: red chair, backpack with straps, tabletop power adapter, laptop, small tape roll, pens and cables. |

## Detail implementation and size assumptions

The second pass adds a tall narrow wheeled whiteboard at `(1.8, -6.2)`, a wood-topped drawer island and overhead gantry at `(4, -5)`, folded tables/chairs/frames at the south end, and four-shelf equipment rack, twelve trays and bin in the southwest corner. The island is a storage cabinet rather than a generic open table.

The east wall gets a single dual-monitor steering-controller workstation and **five narrow black lockers**. West-side additions include an instrument table/open black case, red X aircraft stand and small silver plane, larger aircraft on a black fabrication table, and a rolling stool. Two north workbenches carry monitors and equipment; nearby are open wood shelves, orange shop vacuum with hose, containers, spare wings and a distinct triangular blue delta aircraft. Two cork room-divider panels with grey framing, casters, yellow clips, flag and under-panel storage are placed near `(3, 3)` facing south.

Approximate standard scales: work surface 0.76–0.89 m, locker 2 m, instrument table 0.86 m, divider 1.8 m, whiteboard approximately 1.9 m. These estimates follow familiar furniture proportions and the supplied 12 × 24 m envelope; no exact dimensions are claimed. The preliminary video camera `(2, -2, 1.62)` is a presentation estimate. Distances should be refined by comparing the same camera against the directional source frames.

`room_details()` currently supplies 423 parts with 207 local collision components. Names are unique, material keys resolve, modules compile and no detail collider center enters the UGV/drone route X `[-1, 3]`, Y `[-5, 0]`. Parent integration is responsible for removing superseded generic furniture, matching camera views and final Gazebo validation.

## Video review revision 3

- 6 s: shifted the two-monitor/steering desk north to y=-4 m so it is visible beside the lockers; shortened the drawer island and gantry to 72% of their previous length.
- 9–11 s: moved the joined meeting table to (-3.6,-3.9), enlarged its length to 2.4 m, and arranged three chairs on each outer edge.
- 13–15 s: replaced the generic northwest aircraft island and foam stock with the coherent bench, airframe, project display, leaning foam, separate plane stand and stool in `room_west_bays.py`; moved the adjacent red tool chest inward.
- Re-rendered the full second-by-second comparison under `preview/room/every-second-v3`. Positions and dimensions remain estimates from the rotating reference footage.

The 9–10 s camera headings were refined between the south and west anchors; stacked chairs were added behind the meeting table in the 11 s window bay.

## Walking reference refinement

The second 17.43s video establishes the sequence from the aircraft construction area toward the south-west door. Added a separate aircraft build island, tool pegboard bench, printer station, open tool chest, deep lattice beam on two trestles, and striped aircraft. Replaced the earlier flat wing stand and closed chest to avoid duplicates. Door bay now includes folded grid frames, labelled lidded tubs, and meeting-table laptop, prototype, papers, transmitter and cables. See ROOM_WEST_BAYS.md for detailed observed objects.

`walk_camera_estimates.json` contains an explicitly estimated moving camera path, not a calibrated reconstruction. `render_video_comparison.py --poses walk_camera_estimates.json --end 17` renders its views. Room footprint remains the user-estimated 12 by 24 metres.
