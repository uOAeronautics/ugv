"""Initialize native Gazebo motor/contact parameters from training data only.
Requires scipy and numpy. Full actual dynamics are evaluated separately by
`evaluate_mecanum_gazebo.py`; this surrogate fit alone is not acceptance.
"""
import sys,json,numpy as np
from scipy.optimize import least_squares
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/calibration'),str(ROOT/'src')]
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('folder',type=Path)
parser.add_argument('--output',type=Path,default=ROOT/'simulation/gazebo/mecanum_native_profile.json')
args=parser.parse_args()
from mecanum_fit import extract,extract_visual_segments
folder=args.folder
trials,m=extract(folder);segments,_=extract_visual_segments(folder)
# Fit only observed training intervals; preserve actual full history for lag.
data=[]
for tr in trials:
 if tr['split']!='train':continue
 t=np.array(tr['t']);w=-np.array(tr['wheels'])[:,[0,2,1,3]]
 ref=np.array(tr['velocity']);mask=np.zeros(len(t),bool)
 for seg in segments:
  if seg['base_trial']!=tr['id']:continue
  st=np.array(seg['t']);sv=np.array(seg['velocity']);inside=(t>=st[0])&(t<=st[-1]);mask|=inside
  for k in (0,1):ref[inside,k]=np.interp(t[inside],st,sv[:,k])
 data.append((t,w,ref,mask))
# Kinematic initial fit; lateral factor is diagnostic for physical bearing drag.
M=np.array([[1,1,1,1],[-1,1,1,-1],[-1,1,-1,1]])
def predict(t,w,p):
 gains=np.where(w>=0,np.exp(p[:4]),np.exp(p[4:8]));target=np.sign(w)*np.abs(w)**p[10]*gains
 q=np.zeros_like(target);tau=np.exp(p[8])
 for i in range(1,len(t)):q[i]=q[i-1]+(1-np.exp(-(t[i]-t[i-1])/tau))*(target[i]-q[i-1])
 v=q@M.T*.03175*26.17994/4;v[:,1]*=np.exp(p[9]);v[:,2]/=.13985
 return v

def residual(p):
 rows=[]
 for t,w,ref,mask in data:
  err=predict(t,w,p)-ref
  rows.extend((err[mask,:2]/.15).ravel());rows.extend(err[:,2]/.7)
 rows.extend(p[:8]*.5) # weak equal motor prior; cannot identify eight gains independently here
 return np.array(rows)
p=np.r_[np.zeros(8),np.log(.08),np.log(.65),1.]
fit=least_squares(residual,p,bounds=(np.r_[np.full(8,np.log(.4)),np.log(.01),np.log(.15),.3],np.r_[np.full(8,np.log(3)),np.log(.4),np.log(1.5),2.4]),max_nfev=60)
p=fit.x
print('gains',np.exp(p[:4]),np.exp(p[4:8]),'tau',np.exp(p[8]),'latfactor',np.exp(p[9]))
gains=np.r_[np.exp(p[:4]),np.exp(p[4:8])]
scale=float(np.mean(gains))
omega0=26.17994*scale
# Match the fitted lateral attenuation with an equivalent passive roller
# viscous load. This is an initialization for physical Gazebo validation,
# not an independently identified bearing measurement.
wheel_drag=4*.0784532/(omega0*.03175**2)
roller_damping=max(0.,wheel_drag*(1/float(np.exp(p[9]))-1)*.006**2/8)
params=dict(wheel_duty_exponent=[float(p[10])]*4,
            wheel_voltage_gain=(np.exp(p[:4])/scale).tolist(),
            wheel_reverse_voltage_gain=(np.exp(p[4:8])/scale).tolist(),
            no_load_speed=omega0,stall_torque=.0784532,
            wheel_drive_tau=[float(np.exp(p[8]))]*4,wheel_brake_tau=[.12]*4,
            wheel_deadzone=[0.]*4,wheel_gain_stddev=[0.]*4,
            roller_bearing_damping=roller_damping,roller_bearing_friction=.000001,
            random_seed=0)
output=dict(schema_version=1,gazebo_motor_parameters=params,
            scope='Provisional physically structured training initialization; acceptance requires native heldout replay',
            training_run=str(folder.resolve()),heldout_used=False,
            wheel_order=['front-left','front-right','rear-left','rear-right'],
            training_trial_ids=[tr['id'] for tr in trials if tr['split']=='train'],
            fitting_method='bounded least-squares first-order mecanum motor surrogate, then equivalent passive bearing load',
            fitted_surrogate_gains_positive=np.exp(p[:4]).tolist(),fitted_surrogate_gains_negative=np.exp(p[4:8]).tolist(),
            fitted_surrogate_lateral_factor=float(np.exp(p[9])),
            fitted_surrogate_duty_exponent=float(p[10]),
            optimizer_cost=float(fit.cost),optimizer_success=bool(fit.success),
            limitations=['Only contiguous accepted visual segments fit translation; gyro yaw fits full training trajectories.',
                         'Visual inertial scale is not independent external metric truth.',
                         'Motor speed, gain, slip and passive bearing drag are not separately identifiable.',
                         'Gaussian randomization defaults to zero; no arbitrary uncertainty claimed as calibrated.'])
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(output,indent=2))
print(args.output)
