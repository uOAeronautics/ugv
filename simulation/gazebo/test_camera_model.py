"""Camera geometry, generated optics, image units and bounded frame handling."""
import json
import math
from pathlib import Path
import struct
import unittest
import xml.etree.ElementTree as E
import zlib
from camera_model import camera_geometry,load_camera
from camera_stream import gray_png

HERE=Path(__file__).resolve().parent

class CameraTest(unittest.TestCase):
    def test_new_mount_is_on_board_with_lens_centered_in_40mm_stand(self):
        dims=json.loads((HERE/'dimensions.json').read_text());p=load_camera();g=camera_geometry(dims,p)
        self.assertAlmostEqual(dims['chassis_length_in']*.0254/2-g['stand_x'],.04)
        self.assertAlmostEqual(g['center_z']-g['board_z'],.02)
        self.assertAlmostEqual(p['mount']['width_m'],.04)
        self.assertAlmostEqual(p['mount']['height_m'],.04)
        self.assertGreater(g['optical_xyz'][0],g['stand_x'])

    def test_native_sensor_matches_observed_stream_and_nominal_real_vision_intrinsics(self):
        camera=E.parse(HERE/'freenove_world.sdf').find('.//sensor[@name="arducam_front"]')
        p=load_camera()
        self.assertEqual(camera.findtext('update_rate'),'30')
        self.assertEqual(camera.findtext('camera/image/format'),'L8')
        for k in ('width','height'):self.assertEqual(int(camera.findtext('camera/image/'+k)),p[k])
        fov=float(camera.findtext('camera/horizontal_fov'))
        self.assertAlmostEqual(p['width']/(2*math.tan(fov/2)),968)
        for k in ('fx','fy','cx','cy'):self.assertEqual(float(camera.findtext('camera/lens/intrinsics/'+k)),p[k])
        self.assertFalse(p['intrinsics_confirmed'])

    def test_png_preserves_grayscale_rows_and_stride(self):
        image=gray_png(2,2,b'\x00\xffpad\x20\x80pad',5)
        self.assertEqual(image[:8],b'\x89PNG\r\n\x1a\n')
        offset=8;compressed=b''
        while offset<len(image):
            n=struct.unpack('!I',image[offset:offset+4])[0];kind=image[offset+4:offset+8];payload=image[offset+8:offset+8+n]
            if kind==b'IHDR':self.assertEqual(struct.unpack('!2I5B',payload),(2,2,8,0,0,0,0))
            if kind==b'IDAT':compressed+=payload
            offset+=n+12
        self.assertEqual(zlib.decompress(compressed),b'\x00\x00\xff\x00\x20\x80')
        with self.assertRaises(ValueError):gray_png(10,10,b'bad')

if __name__=='__main__':unittest.main()
