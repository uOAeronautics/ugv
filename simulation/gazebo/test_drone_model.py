"""Geometry contracts for the photo estimated drone docking obstacle."""
import math
import json
import unittest
from pathlib import Path
from drone_model import drone_assembly


def bounds(part):
    rx, ry, rz = part['rot']
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    rotation = ((cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx),
                (sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx),
                (-sy, cy*sx, cy*cx))
    if part['shape'] == 'box':
        extent = [sum(abs(row[j])*part['size'][j]/2 for j in range(3)) for row in rotation]
    else:
        radius, length = part['size']
        extent = [abs(row[2])*length/2+radius*math.hypot(row[0], row[1]) for row in rotation]
    return [(x-e, x+e) for x, e in zip(part['pos'], extent)]


class DroneGeometry(unittest.TestCase):
    def test_two_central_landing_legs_and_four_marker_cards(self):
        parts = drone_assembly()
        legs = [p for p in parts if p['name'].startswith('landing_leg_')]
        cards = [p for p in parts if p['name'].endswith('_paper')]
        self.assertEqual(len(legs), 2)
        self.assertEqual(len(cards), 4)
        self.assertTrue(all(abs(p['pos'][0]) < 1e-9 for p in legs))

    def test_ugv_center_approach_has_clear_structural_corridor(self):
        # Use current car dimensions, including independently configured
        # mecanum width/track, with 5mm per side and 125mm vertical clearance.
        p = json.loads((Path(__file__).parent/'dimensions.json').read_text())
        tire = p.get('wheel_width_mm_assumed', (p['overall_width_in']-p['chassis_width_in'])*12.7)/1000
        track = p.get('wheel_track_mm_assumed', (p['overall_width_in']*.0254-tire)*1000)/1000
        half_width = (track+tire)/2+.005
        corridor = ((-.45, .45), (-half_width, half_width), (.001, .125))
        for part in drone_assembly():
            if not part['collision']:
                continue
            pieces = [part]
            if part['shape'] == 'cylinder':
                # Slice tilted legs so their high, inward ends do not make
                # a false floor-level obstruction through one large AABB.
                radius, length = part['size']
                rx, ry, rz = part['rot']
                axis = (math.cos(rz)*math.sin(ry)*math.cos(rx)+math.sin(rz)*math.sin(rx),
                        math.sin(rz)*math.sin(ry)*math.cos(rx)-math.cos(rz)*math.sin(rx),
                        math.cos(ry)*math.cos(rx))
                count = max(1, math.ceil(length/.005))
                pieces = [dict(part, size=(radius,length/count),
                               pos=tuple(part['pos'][j]+axis[j]*length*((i+.5)/count-.5) for j in range(3)))
                          for i in range(count)]
            overlap = any(all(a < d and c < b for (a,b), (c,d) in zip(bounds(piece), corridor)) for piece in pieces)
            self.assertFalse(overlap, 'Docking corridor blocked by '+part['name'])

    def test_visual_meshes_have_valid_triangle_indices(self):
        root = Path(__file__).resolve().parent
        for part in drone_assembly():
            if part['shape'] != 'mesh':
                continue
            lines = (root / part['mesh']).read_text().splitlines()
            vertices = [line for line in lines if line.startswith('v ')]
            faces = [line.split()[1:] for line in lines if line.startswith('f ')]
            self.assertTrue(vertices and faces)
            for face in faces:
                indices = [int(i) for i in face]
                self.assertEqual(len(set(indices)), 3)
                self.assertTrue(all(1 <= i <= len(vertices) for i in indices))


if __name__ == '__main__':
    unittest.main()
