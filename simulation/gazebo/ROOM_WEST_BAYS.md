# West bays: comparison refinement at 11, 13 and 15 seconds

This pass was checked against the source panorama and `preview/room/every-second-v2/pairs/11.png`, `13.png`, and `15.png`. The comparison showed that listing the right object classes was insufficient: the fabricated aircraft bay needed to project into the room as a coherent cluster, with its stool, display, foam stock and separate aircraft support occupying the same bay.

`room_west_bays.bay_parts()` returns 153 standard box/cylinder parts in the corrected cardinal frame, with `BAY_COLORS`. Names begin `westbay_`. It has no external dependencies or generated meshes. There are 18 local collision components; ornament and aircraft detail are visual only. The UGV/drone corridor remains clear.

## Observed and modeled relationships

| Source view | Observed relationship | New model placement |
|---|---|---|
| 11 s | Meeting tables and chair stacks fill the left window bay; wing frame and tripod occupy the next bay to the right. | Existing meeting cluster must remain visible at left. No duplicate meeting tables or wing frame are added by this module. |
| 13 s | Beyond tripod: red X stand, small folding instrument table, open black equipment case, then a black work table with aircraft. A chrome stool, separate small RC plane stand, tall display board and white foam stock belong to the same work area. | Instrument accessories near `(-5, 1.9)` supplement the existing table/case. Black aircraft bench centered `(-3.8, 4.4)`, footprint 1.1 × 2.2 m. Separate silver T aircraft stand `(-3.35, 3.13)` and backed rolling stool `(-4.55, 2.95)`. |
| 15 s | Fabrication table and leaning white boards fill the left foreground, red rolling tool chest immediately beyond their right edge. Rear benches are visible through the gap, not isolated at a bare distant wall. | Three foam boards near `(-2.93, 3.94)`; project display near `(-4.15, 5.62)`; tall red/white aircraft parts behind the table. Existing tool chest should be near `(-2.1, 4.7)`, facing south toward the camera. |

The main bench carries a slender metal fuselage with exposed wing spars/ribs, white skin, tail surfaces, a red stripe and loose spare wing pieces. The nearby small RC plane has red/white wings, a blue nose, tail and its own silver T stand. The shop stool has a black backrest, chrome foot ring, five legs and casters. The folding instrument table gets small white instruments, knobs, a compact open laptop and coiled black leads.

The precise identities of partly obscured equipment are uncertain. The source clearly supports the furniture/support/aircraft relationships and approximate proportions; exact internal component shapes remain estimates. People remain excluded.

## Integration

Append `bay_parts()` after the cardinal room transform. Remove superseded `detail_northwest_shop_`, `detail_large_airframe_` and legacy `white_foam_stock_` parts. Keep the existing `detail_west_folding_`, `detail_west_crate_`, red X stand, wooden wing test frame and tripod. `BAY_COLORS` must be merged into the scene material map.

Source compilation, unique names, material lookup and route clearance were checked. Parent integration handles matching camera renders and final Gazebo validation.

## Walk-through evidence, second video

The new 17.43 s walking video materially improves the depth ordering. It shows that the northern construction island, long pegboard tool bench and aircraft/display bench are **separate surfaces**. Combining them into one generic table omitted much of the left side.

- At approximately 0–2 s, a large black-topped island carries a long unfinished wooden lattice wing, adhesive bottles, papers, small supplies and a brown supply bag. A backed chrome shop stool sits beside it. This island is now centered `(-3.25, 7.4)` and has separate local legs and a lower shelf.
- Behind it, a long pale tool bench has a wooden pegboard, rows of bright yellow-green/blue/red holders and tools, compact drawer towers, clear parts trays and a shelf underneath. This is modeled independently at `(-5.03, 6.62)`, 2.75 m along Y.
- A rear side table carries two compact desktop 3D printers with open towers, beds, extruder heads and filament spools, modeled at `(-5.02, 9.12)`. Their exact make is not asserted.
- At approximately 4–7 s, the red tool chest has an open top lid, badges and a pulled-out bottom drawer. The aircraft on the separate silver support has blue fuselage/canopy tones and red/white/blue wings. A tall stack of white foam stock leans next to the display/aircraft table.
- At approximately 8–10 s, the test article is a substantial wooden box/lattice beam approximately 3 m long, supported by two freestanding trestles. Four longerons, repeated vertical ribs, diagonal members and a central plywood patch define its silhouette. The replacement is centered `(-4.25, 0.10)`. The survey tripod remains separate.
- At approximately 11–17 s, the camera passes the meeting tables, red/grey chairs, stacked black chairs, folded table frames, doorway and storage rack. Parent integration owns these objects to avoid duplicated geometry.

The walking video shows darker roller shades than the earlier bright panorama. Window/shade material changes belong to the room shell and are not authored by this module.

The module now returns **490 parts with 44 local collision components**. New source is stdlib only and names/materials validate. Parent integration should remove old `wing_stand_`, `wing_spar_` and `wing_rib_` parts, preserving the separate tripod. Remove the former closed tool chest at `(-2.1, 4.7)` (typically `tools0_`) before appending the new open-lid/open-drawer chest. The protected drone corridor X `[1.55, 2.05]`, Y `[-4.9, -0.5]` is unaffected.
