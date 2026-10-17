"""Render known targets in an isolated world and verify mono optics and timing."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as E

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));os.environ['GZ_PARTITION']='ugv-camera-check-'+uuid.uuid4().hex
from camera_model import camera_geometry,load_camera
from camera_stream import CameraStream
from build_motor_plugin import plugin_environment
from gz.transport import Node
from gz.msgs.image_pb2 import Image
from gz.msgs.camera_info_pb2 import CameraInfo

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'.control-logs');args=parser.parse_args()
p=load_camera();dims=json.loads((HERE/'dimensions.json').read_text());g=camera_geometry(dims,p)
root=E.parse(HERE/'freenove_world.sdf');world=root.find('world')
model=E.SubElement(world,'model',name='temporary_camera_target');E.SubElement(model,'static').text='true'
link=E.SubElement(model,'link',name='target')
def rectangle(name,x,y,z,width,height,color):
    visual=E.SubElement(link,'visual',name=name);E.SubElement(visual,'pose').text=f'{x} {y} {z} 0 0 0'
    box=E.SubElement(E.SubElement(visual,'geometry'),'box');E.SubElement(box,'size').text=f'.001 {width} {height}'
    mat=E.SubElement(visual,'material')
    for tag in ('ambient','diffuse'):E.SubElement(mat,tag).text=f'{color} {color} {color} 1'
    E.SubElement(mat,'specular').text='0 0 0 1'
optical_x,_,optical_z=g['optical_xyz'];distance=.5
rectangle('white_board',optical_x+distance+.002,0,optical_z+.04,.35,.14,1)
targets=[(.1,.03),(0,.06),(-.1,.03)]
for i,(y,z) in enumerate(targets):rectangle('black_'+str(i),optical_x+distance,y,optical_z+z,.015,.015,0)
node=Node();camera=CameraStream();stamps=[];infos=[]
def image(m):camera.update(m);stamps.append(m.header.stamp.sec+m.header.stamp.nsec*1e-9)
node.subscribe(Image,'/model/freenove/camera/image',image)
for topic in ('/model/freenove/camera/camera_info','/model/freenove/camera/image/camera_info','/model/freenove/camera/image_info'):
    node.subscribe(CameraInfo,topic,lambda m:infos.append(m))
tmp=tempfile.TemporaryDirectory(prefix='ugv-camera-');path=Path(tmp.name)/'world.sdf';root.write(path)
log=(HERE/'.control-logs/camera-validation.log').open('w');env=plugin_environment();env['GZ_SIM_RESOURCE_PATH']=str(HERE)
proc=subprocess.Popen(['gz','sim','-s','-r',str(path)],env=env,cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
try:
    deadline=time.monotonic()+45
    while (len(stamps)<61 or not infos) and time.monotonic()<deadline:
        if proc.poll() is not None:raise RuntimeError('Camera renderer exited; see camera-validation.log')
        time.sleep(.1)
    assert len(stamps)>=61,'No camera frame stream'
    assert infos,('Missing CameraInfo',node.topic_list())
    info=infos[-1];assert (info.width,info.height)==(1280,720)
    k=list(info.intrinsics.k);assert all(abs(a-b)<.01 for a,b in zip(k,[968,0,640,0,968,360,0,0,1])),k
    rate=(len(stamps)-1)/(stamps[-1]-stamps[0]);assert 28<rate<=31,rate
    with camera.lock:width,height,pixels,step,_=camera.frame
    assert (width,height)==(1280,720)
    # Detect the three black squares in an upper-middle ROI, away from floor markings.
    dark={(x,y) for y in range(160,340) for x in range(300,980) if pixels[y*step+x]<25}
    blobs=[]
    while dark:
        point=dark.pop();todo=[point];blob=[point]
        while todo:
            x,y=todo.pop()
            for neighbor in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                if neighbor in dark:dark.remove(neighbor);todo.append(neighbor);blob.append(neighbor)
        if len(blob)>200:blobs.append([statistics.mean(v[i] for v in blob) for i in (0,1)])
    assert len(blobs)==3,blobs
    predicted=sorted([[p['cx']-p['fx']*y/distance,p['cy']-p['fy']*z/distance] for y,z in targets])
    measured=sorted(blobs);errors=[math.dist(a,b) for a,b in zip(measured,predicted)]
    assert max(errors)<3,(measured,predicted,errors)
    out=args.output
    out.mkdir(parents=True,exist_ok=True);(out/'ugv-camera-target-check.png').write_bytes(camera.png())
    report=dict(width=width,height=height,pixel_format='L_INT8',rate_per_sim_second_hz=rate,
        camera_info_intrinsics=k,target_projection_errors_px=errors,
        optical_origin_body_m=g['optical_xyz'],intrinsics_confirmed_on_real_camera=False)
    (HERE/'camera_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));print('PASS: grayscale camera, frame rate, CameraInfo and known-target projection')
finally:
    if proc.poll() is None:
        os.killpg(proc.pid,signal.SIGINT)
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM)
    log.close();tmp.cleanup()
