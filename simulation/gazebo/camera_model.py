"""Shared stand/optical geometry and observed Arducam stream settings."""
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent

def load_camera():
    return json.loads((HERE/'camera_profile.json').read_text())

def camera_geometry(dimensions,profile=None):
    p=profile or load_camera();m=p['mount'];board=dimensions['main_board_height_in']*.0254
    stand_x=dimensions['chassis_length_in']*.0254/2-m['inset_from_board_front_m']
    z=board+m['height_m']/2
    return dict(stand_x=stand_x,board_z=board,center_z=z,
                optical_xyz=[stand_x+m['lens_front_offset_m_assumed'],0,z],
                horizontal_fov=2*math.atan(p['width']/(2*p['fx'])))

def add_camera_sensor(link,node,dimensions):
    p=load_camera();g=camera_geometry(dimensions,p);m=p['mount']
    sensor=node(link,'sensor',name='arducam_front',type='camera')
    node(sensor,'pose',' '.join(map(str,(*g['optical_xyz'],0,m['pitch_rad_assumed'],m['yaw_rad_assumed']))))
    node(sensor,'always_on','true');node(sensor,'update_rate',p['update_rate_hz'])
    node(sensor,'topic','/model/freenove/camera/image')
    camera=node(sensor,'camera');node(camera,'horizontal_fov',g['horizontal_fov'])
    image=node(camera,'image')
    for key in ('width','height','format'):node(image,key,p[key])
    clip=node(camera,'clip');node(clip,'near',p['near_clip_m']);node(clip,'far',p['far_clip_m'])
    lens=node(camera,'lens');intrinsics=node(lens,'intrinsics')
    for key in ('fx','fy','cx','cy'):node(intrinsics,key,p[key])
    node(intrinsics,'s',0)
