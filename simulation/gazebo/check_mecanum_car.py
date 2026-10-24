"""Full generated mecanum car mechanics regression in isolated Gazebo.
Run with simulation/gazebo/.control-venv/bin/python simulation/gazebo/check_mecanum_car.py.
Does not actuate hardware or establish real-car dynamics calibration.
"""
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
   pose.update(t=m.header.stamp.sec+m.header.stamp.nsec*1e-9,x=q.position.x,y=q.position.y,z=q.position.z,yaw=math.atan2(2*q.orientation.w*q.orientation.z,1-2*q.orientation.z*q.orientation.z))
def motors(m): samples.append(list(m.data))
n.subscribe(Pose_V,'/world/freenove_lab/dynamic_pose/info',poses)
n.subscribe(Double_V,'/model/freenove/motor_state',motors)
(p/'.control-logs').mkdir(exist_ok=True)
log=(p/'.control-logs/mecanum-car-check.log').open('w')
proc=subprocess.Popen(['gz','sim','-s','-r','-v','3',str(p/'freenove_world.sdf')],cwd=p,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
def drive(v,w,secs):
 end=pose['t']+secs;deadline=time.monotonic()+secs*15+10
 while pose['t']<end:
  assert time.monotonic()<deadline,'Gazebo clock stopped'
  m=Twist();m.linear.x=v;m.angular.z=w;pub.publish(m);time.sleep(.05)
try:
 deadline=time.monotonic()+15
 while (not pose or not samples) and time.monotonic()<deadline: time.sleep(.1)
 assert pose and samples,'Motor system or physics did not initialize'
 drive(0,0,.5)
 results={}
 for name,vx,vy,w in [('forward',.1,0,0),('reverse',-.1,0,0),('left',0,.1,0),('right',0,-.1,0),('ccw',0,0,.5),('cw',0,0,-.5)]:
  before=dict(pose);end=pose['t']+1.5;deadline=time.monotonic()+30
  while pose['t']<end:
   assert time.monotonic()<deadline,'Gazebo clock stopped'
   m=Twist();m.linear.x=vx;m.linear.y=vy;m.angular.z=w;pub.publish(m);time.sleep(.04)
  after=dict(pose)
  dx=after['x']-before['x'];dy=after['y']-before['y'];yaw=before['yaw']
  results[name]=dict(dx_body=dx*math.cos(yaw)+dy*math.sin(yaw),dy_body=-dx*math.sin(yaw)+dy*math.cos(yaw),dyaw=math.atan2(math.sin(after['yaw']-yaw),math.cos(after['yaw']-yaw)),z=after['z'],duration_sim_s=after['t']-before['t'])
  drive(0,0,.5)
 print(json.dumps(results,indent=2),flush=True)
 assert results['forward']['dx_body']>.05
 assert results['left']['dy_body']>.04
 assert results['right']['dy_body']<-.04
 assert results['reverse']['dx_body']<-.05
 assert results['ccw']['dyaw']>.15
 assert results['cw']['dyaw']<-.15
 assert all(abs(r['z'])<.01 for r in results.values())
 # Four-wheel full power and heartbeat loss exercise finite motor forces.
 before=dict(pose);end=time.monotonic()+.4
 while time.monotonic()<end:
  message=Double_V();message.data.extend([1.]*4);duty_pub.publish(message);time.sleep(.03)
 time.sleep(.9);stopped=dict(pose);time.sleep(.35);after=dict(pose)
 assert math.hypot(stopped['x']-before['x'],stopped['y']-before['y'])>.04
 assert math.hypot(after['x']-stopped['x'],after['y']-stopped['y'])<.005
 assert all(abs(samples[-1][i])<1e-9 for i in (3,7,11,15))
 max_torque=max(abs(row[i]) for row in samples if len(row)==16 for i in (2,6,10,14))
 from motor_physics import load_profile
 assert max_torque<=load_profile()['stall_nm']+1e-8
 import xml.etree.ElementTree as E
 calibration=E.parse(p/'freenove_world.sdf').find('.//model[@name="freenove"]/plugin[@name="ugv::TTMotorSystem"]')
 report=dict(scope='Full generated car mechanics smoke check; does not establish real2sim acceptance',
             motor_profile=calibration.findtext('calibration_profile'),calibration_status=calibration.findtext('calibration_status'),results=results,
             full_power_watchdog=dict(displacement_m=math.hypot(stopped['x']-before['x'],stopped['y']-before['y']),
             final_drift_m=math.hypot(after['x']-stopped['x'],after['y']-stopped['y']),max_torque_nm=max_torque))
 (p/'mecanum_car_validation.json').write_text(json.dumps(report,indent=2))
 print('PASS full car: six directions, stable support, full duty, finite torque and 400ms watchdog',flush=True)

finally:
 pub.publish(Twist());os.killpg(proc.pid,signal.SIGINT)
 try: proc.wait(timeout=5)
 except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGTERM)
