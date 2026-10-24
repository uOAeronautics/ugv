"""Stationary IMU noise and current tilt calibration, without six-pose claims."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def allan_deviation(values, sample_period, taus=(.01,.02,.05,.1,.2,.5,1.,2.,5.)):
    """Overlapping Allan deviation of sampled rate/acceleration, not integrated angles."""
    a=np.asarray(values,dtype=float);cs=np.vstack((np.zeros(a.shape[1]),np.cumsum(a,axis=0)));out=[]
    for tau in taus:
        m=max(1,round(tau/sample_period))
        if 2*m>=len(a):continue
        means=(cs[m:]-cs[:-m])/m;delta=means[m:]-means[:-m]
        out.append(dict(tau_s=m*sample_period,averaging_samples=m,difference_count=len(delta),deviation=np.sqrt(np.mean(delta**2,axis=0)/2).tolist()))
    return out


def calibrate(rest_folder, probe_folder, output):
    from mecanum_fit import extract
    blob=(rest_folder/'samples.jsonl').read_bytes();rows=[json.loads(l) for l in blob.splitlines() if l.strip()]
    rows=[r for r in rows if not any(r['wheels'].values()) and r.get('stopped',True)]
    t=np.array([r['t'] for r in rows]);dt=float(np.median(np.diff(t)));channels={}
    for field in ('accel_g','gyro_dps'):
        a=np.array([r[field] for r in rows]);mean=a.mean(axis=0);residual=a-mean;cov=np.cov(residual.T)
        bins=[]
        for start in np.arange(t[0],t[-1]-.5,.5):
            selected=a[(t>=start)&(t<start+.5)]
            if len(selected)>=30:bins.append(selected.mean(axis=0))
        bins=np.array(bins);centered=bins-bins.mean(axis=0)
        rho=np.sum(centered[:-1]*centered[1:],axis=0)/np.maximum(np.sum(centered[:-1]**2,axis=0),1e-20);rho=np.clip(rho,0,.99)
        bias_std=np.sqrt(np.maximum(0,np.var(bins,axis=0)-np.diag(cov)*dt/.5));tau=np.where(rho>.05,-.5/np.log(np.maximum(rho,1e-6)),.5)
        windows=[a[(t>=start)&(t<start+10)].mean(axis=0) for start in np.arange(t[0],t[-1]-9,10)]
        channels[field]=dict(mean=mean.tolist(),white_covariance=cov.tolist(),white_std=np.sqrt(np.diag(cov)).tolist(),rest_window_mean_std=np.std(windows,axis=0).tolist(),short_term_bias_std=bias_std.tolist(),short_term_bias_tau_s=tau.tolist(),half_second_lag1=rho.tolist(),tau_note='Empirical lag of 0.5s stationary block means; values at0.5s are unresolved, not measured longterm drift.',allan_deviation=allan_deviation(a,dt),first_difference_covariance=(np.cov(np.diff(a,axis=0).T)/2).tolist())
    _,mount=extract(probe_folder);up=np.array(channels['accel_g']['mean']);up/=np.linalg.norm(up);forward=np.array(mount['body_axes_sensor'])[:,0];forward-=up*forward.dot(up);forward/=np.linalg.norm(forward);left=np.cross(up,forward);matrix=np.column_stack((forward,left,up))
    # Preserve the simulator's existing compatible rendering/noise configuration.
    old=json.loads(output.read_text()) if output.exists() else {}
    old.update(schema_version=1,sensor='Nano 33 BLE Sense Rev2 BMI270',seed=5774,update_rate_hz=100,ideal_physics_rate_hz=1000,anti_alias_cutoff_hz_assumed=25,source_recordings=[dict(session=rest_folder.name,sha256=hashlib.sha256(blob).hexdigest(),rest_samples=len(rows),role='current_stationary_noise'),dict(session=probe_folder.name,sha256=hashlib.sha256((probe_folder/'samples.jsonl').read_bytes()).hexdigest(),rest_samples=0,role='signed_forward_launch_mount')],rest_spans=1,rest_samples=len(rows),settled_rest_s=float(t[-1]-t[0]),selection='Dedicated stationary recording: stopped and all actual wheel PWMzero. No moving samples used as sensornoise.',channels=channels,body_to_sensor_matrix=matrix.tolist(),accel_bias_g=(np.array(channels['accel_g']['mean'])-up).tolist(),gyro_bias_dps=channels['gyro_dps']['mean'],sample_period_median_s=dt,max_sample_gap_s=float(np.max(np.diff(t))),mount_calibration=dict(forward_launch_signal_g=mount['forward_launch_signal_g'],forward_axis_origin='Probe signed launch direction, reorthogonalized against newer stationarygravity',gravity_norm_g=float(np.linalg.norm(channels['accel_g']['mean'])),mount_change_note='Stationarygravity differs from probe; reorthogonalization updates tilt but cannot distinguish sensorfixture shift from chassis pitch.'),bias_model='Measured stationary mean gyro offset; full residual samplecovariance plus empirical shortterm OU blockmeans. No independent powercycle biasdistribution inferred.',limits=['Single stationary60s record; temperature drift, powercycle repeatability, longterm biasinstability and sixpose accelerometer gains/offsets remain uncalibrated.','Overlapping Allan deviations cover10ms–5s; overlapping differences are correlated and do not supply independent confidenceintervals.','Rest covariance contains firmwarefiltering and environmentalvibration, not identified intrinsic sensornoisedensity.','Motoron fluctuations combine true chassisacceleration, wheelroller ripple, mountingflex and sensornoise; they must not all be labeled measurementnoise.','Gravity norm residual is assigned alonggravity for simulatorcompatibility; accelerometer scale/offset are not independently identifiable in onepose.','Forwarddirection from earlierlaunch projected onto newgravity; mountshift uncertainty remains.'])
    output.write_text(json.dumps(old,indent=2)+'\n')
    (rest_folder/'imu_calibration.json').write_text(json.dumps(old,indent=2)+'\n')
    print(json.dumps(dict(source=rest_folder.name,samples=len(rows),duration_s=old['settled_rest_s'],body_to_sensor_matrix=old['body_to_sensor_matrix'],gyro_bias_dps=old['gyro_bias_dps'],gyro_std=channels['gyro_dps']['white_std'],accel_std=channels['accel_g']['white_std']),indent=2))
    return old

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('rest',type=Path);p.add_argument('probe',type=Path);p.add_argument('--output',type=Path,default=Path('simulation/gazebo/imu_noise_profile.json'));a=p.parse_args();calibrate(a.rest,a.probe,a.output)
