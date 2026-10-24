"""Check room collision envelopes and offline assets for the UGV scene."""
import unittest
import json
from pathlib import Path
from room_model import room_assembly, ROOM_COLORS
from test_drone_model import bounds


class RoomGeometry(unittest.TestCase):
    def test_user_direction_anchors_and_l_shaped_glazing(self):
        config=json.loads((Path(__file__).parent/'room_dimensions.json').read_text())
        headings=config['panorama_yaws_deg']
        for second, expected in ((4,0),(8,-90),(11,-180),(17,-270)):
            self.assertAlmostEqual(headings[second],expected)
        parts={p['name']:p for p in room_assembly()}
        self.assertAlmostEqual(parts['south_facade_glass_4']['pos'][1],-12)
        self.assertAlmostEqual(parts['frosted_glazing_4']['pos'][0],-6.0025)

    def test_finite_floor_matches_user_scale(self):
        floor = next(p for p in room_assembly() if p['name'] == 'concrete_slab')
        extent = bounds(floor)
        self.assertAlmostEqual(extent[0][1]-extent[0][0], 12)
        self.assertAlmostEqual(extent[1][1]-extent[1][0], 24)
        self.assertTrue(floor['collision'])
        self.assertAlmostEqual(floor['pos'][2]+floor['size'][2]/2, 0)

    def test_demo_floor_is_clear_at_ugv_height(self):
        config=json.loads((Path(__file__).parent/'room_dimensions.json').read_text())
        area=config['free_demonstration_area']
        corridor = (area['x'], area['y'], (.001, .125))
        for part in room_assembly():
            if not part['collision']:
                continue
            overlap = all(a < d and c < b for (a,b), (c,d) in zip(bounds(part), corridor))
            self.assertFalse(overlap, 'Demo floor blocked by '+part['name'])

    def test_assets_and_materials_exist_offline(self):
        root = Path(__file__).resolve().parent
        parts = room_assembly()
        self.assertEqual(len({p['name'] for p in parts}), len(parts))
        for part in parts:
            self.assertIn(part['color'], ROOM_COLORS)
            if part['shape'] == 'mesh':
                self.assertTrue((root/part['mesh']).is_file())


if __name__ == '__main__':
    unittest.main()
