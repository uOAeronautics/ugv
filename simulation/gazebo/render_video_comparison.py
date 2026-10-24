"""Render integer-second views from the saved lab scene.

blender -b preview/room/lab-docking.blend --python render_video_comparison.py -- OUTPUT
Camera angles are visual estimates, not recovered camera calibration.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import bpy
from mathutils import Vector

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('output', type=Path)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=19)
parser.add_argument('--engine', choices=('CYCLES','BLENDER_EEVEE'), default='CYCLES')
parser.add_argument('--poses', type=Path, help='JSON list of estimated per-second camera poses')
parser.add_argument('--samples', type=int, default=16)
args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
args.output.mkdir(parents=True, exist_ok=True)
scene = bpy.context.scene
scene.render.engine = args.engine
scene.render.resolution_x, scene.render.resolution_y = 576, 1024
scene.render.resolution_percentage = 100
scene.cycles.samples = args.samples
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 4
scene.cycles.diffuse_bounces = 2
scene.cycles.glossy_bounces = 2
camera = scene.camera
camera.data.type = 'PERSP'
camera.data.sensor_fit = 'HORIZONTAL'
camera.data.sensor_width = 36
camera.data.lens = 37
fit = json.loads((Path(__file__).parent/'room_dimensions.json').read_text())
camera.location = fit['panorama_camera_estimate']
# Clockwise sweep inferred from the observed ordering of teaching, computing,
# exit, glazing and workshop landmarks. Last seconds include the phone tilt.
yaws = [math.radians(value) for value in fit['panorama_yaws_deg']]
pitches = [math.radians(value) for value in fit['panorama_pitches_deg']]
poses=[]
for second,(yaw,pitch) in enumerate(zip(yaws,pitches)):
    poses.append(dict(second=second,position=list(camera.location),yaw_rad=yaw,
                      pitch_rad=pitch,lens_mm=37,estimated=True))
if args.poses:
    poses=json.loads(args.poses.read_text())
    args.end=min(args.end,len(poses)-1)
(args.output/'camera_poses.json').write_text(json.dumps(poses,indent=2)+'\n')
for second in range(args.start, args.end+1):
    pose=poses[second]
    yaw,pitch=pose['yaw_rad'],pose['pitch_rad']
    camera.location=pose['position']
    camera.data.lens=pose.get('lens_mm',37)
    direction = Vector((math.cos(pitch)*math.cos(yaw),math.cos(pitch)*math.sin(yaw),math.sin(pitch)))
    camera.rotation_euler = direction.to_track_quat('-Z','Y').to_euler()
    scene.render.filepath = str(args.output/f'{second:02d}.png')
    bpy.ops.render.render(write_still=True)
