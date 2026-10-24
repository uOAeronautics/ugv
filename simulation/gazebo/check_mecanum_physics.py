"""Isolated passive-roller contact check; no hardware, no calibrated-car claims.
Run with .control-venv/bin/python check_mecanum_physics.py.
Uses a simple 550 g rigid probe body to isolate roller handedness/contact.
"""
import sys,math,json,xml.etree.ElementTree as E
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
(HERE/'.control-logs').mkdir(exist_ok=True)
from mecanum_geometry import add_passive_rollers
from generate import node
root=E.Element('sdf',version='1.9');world=node(root,'world',name='freenove_lab')
physics=node(world,'physics',name='default',type='ignored');node(physics,'max_step_size',.001)
for kind,cls in [('physics','Physics'),('scene-broadcaster','SceneBroadcaster')]:node(world,'plugin',filename='gz-sim-'+kind+'-system',name='gz::sim::systems::'+cls)
floor=node(world,'model',name='ground');node(floor,'static','true');link=node(floor,'link',name='floor');collision=node(link,'collision',name='floor');box=node(node(collision,'geometry'),'box');node(box,'size','20 20 .02');node(collision,'pose','0 0 -.01 0 0 0')
model=node(world,'model',name='freenove');node(model,'pose','0 0 .001 0 0 0')
body=node(model,'link',name='chassis');node(body,'pose','0 0 .05 0 0 0');inertial=node(body,'inertial');node(inertial,'mass',.55);tensor=node(inertial,'inertia')
for key,value in zip(('ixx','iyy','izz'),(.001,.002,.003)):node(tensor,key,value)
groups={}
for tag,x,y in [('front_left',.06985,.07),('front_right',.06985,-.07),('rear_left',-.06985,.07),('rear_right',-.06985,-.07)]:
 name=tag+'_wheel';groups[name]={'origin':(x,y,.03175)}
 link=node(model,'link',name=name);node(link,'pose',f'{x} {y} .03175 0 0 0');inertial=node(link,'inertial');node(inertial,'mass',.02);tensor=node(inertial,'inertia')
 for key in ('ixx','iyy','izz'):node(tensor,key,.000005)
 joint=node(model,'joint',name=name+'_joint',type='revolute');node(joint,'parent','chassis');node(joint,'child',name);axis=node(joint,'axis');node(axis,'xyz','0 1 0')
add_passive_rollers(model,node,groups)
plugin=node(model,'plugin',filename='ugv_tt_motor',name='ugv::TTMotorSystem');node(plugin,'mecanum','true');node(plugin,'wheelbase',.1397);node(plugin,'wheel_separation',.14)
E.ElementTree(root).write(HERE/'.control-logs/mecanum_probe.sdf')

"""Isolated live Gazebo regression: requires the control panel Python environment."""
import os,sys,time,math,threading,subprocess,json,signal
from pathlib import Path
p=Path(__file__).resolve().parent;sys.path.insert(0,str(p))
from build_motor_plugin import plugin_environment
import uuid
os.environ['GZ_PARTITION']='ugv-tt-check-'+uuid.uuid4().hex
env=plugin_environment();env['GZ_PARTITION']=os.environ['GZ_PARTITION']
from gz.transport import Node
from gz.msgs.twist_pb2 import Twist
from gz.msgs.pose_v_pb2 import Pose_V
from gz.msgs.double_v_pb2 import Double_V
n=Node();pub=n.advertise('/model/freenove/cmd_vel',Twist)
duty_pub=n.advertise('/model/freenove/wheel_duty',Double_V)
pose={}; samples=[]
def poses(m):
 for q in m.pose:
  if q.name=='freenove':
   pose.update(x=q.position.x,y=q.position.y,z=q.position.z,yaw=math.atan2(2*q.orientation.w*q.orientation.z,1-2*q.orientation.z*q.orientation.z))
def motors(m): samples.append(list(m.data))
n.subscribe(Pose_V,'/world/freenove_lab/dynamic_pose/info',poses)
n.subscribe(Double_V,'/model/freenove/motor_state',motors)
(p/'.control-logs').mkdir(exist_ok=True)
log=(p/'.control-logs/mecanum-physics-check.log').open('w')
proc=subprocess.Popen(['gz','sim','-s','-r','-v','3',str(p/'.control-logs/mecanum_probe.sdf')],cwd=p,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
def drive(v,w,secs):
 end=time.monotonic()+secs
 while time.monotonic()<end:
  m=Twist();m.linear.x=v;m.angular.z=w;pub.publish(m);time.sleep(.05)
try:
 deadline=time.monotonic()+15
 while (not pose or not samples) and time.monotonic()<deadline: time.sleep(.1)
 assert pose and samples,'Motor system or physics did not initialize'
 drive(0,0,.5)
 results={}
 for name,vx,vy,w in [('forward',.1,0,0),('left',0,.1,0),('ccw',0,0,.5)]:
  before=dict(pose);end=time.monotonic()+1.5
  while time.monotonic()<end:
   m=Twist();m.linear.x=vx;m.linear.y=vy;m.angular.z=w;pub.publish(m);time.sleep(.04)
  after=dict(pose)
  dx=after['x']-before['x'];dy=after['y']-before['y'];yaw=before['yaw']
  results[name]=dict(dx_body=dx*math.cos(yaw)+dy*math.sin(yaw),dy_body=-dx*math.sin(yaw)+dy*math.cos(yaw),dyaw=math.atan2(math.sin(after['yaw']-yaw),math.cos(after['yaw']-yaw)),z=after['z'])
  drive(0,0,.5)
 print(json.dumps(results,indent=2),flush=True)
 assert results['forward']['dx_body']>.05
 assert results['left']['dy_body']>.04
 assert results['ccw']['dyaw']>.15
 print('PASS passive-roller physics: forward, left strafe, counterclockwise torque drive',flush=True)

finally:
 pub.publish(Twist());os.killpg(proc.pid,signal.SIGINT)
 try: proc.wait(timeout=5)
 except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGTERM)
