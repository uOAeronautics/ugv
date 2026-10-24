"""Fit body response from a timestamped mecanum capture; never moves hardware.

Optional reference.jsonl rows: {t, body_velocity:[forward,left,ccw], source}.
Translation is independently observed ONLY when source is external_metric_pose;
otherwise accelerometer integration is clearly marked provisional.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from ugv.mecanum_dynamics import WHEEL_ORDER, fit, fit_bayesian


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def integral(a,t):
    return np.vstack((np.zeros(a.shape[1]),np.cumsum((a[1:]+a[:-1])*.5*np.diff(t)[:,None],axis=0)))


def extract(folder):
    samples=read(folder/'samples.jsonl');events=read(folder/'events.jsonl')
    starts=[e for e in events if e.get('event')=='trial_start']
    def key(e):return e.get('trial',e.get('id'))
    def command(e):
        keys=e.get('keys','')
        return np.array([e.get('forward',int('W'in keys)-int('S'in keys)),e.get('strafe',int('A'in keys)-int('D'in keys))])
    candidates=[];up_vectors=[]
    for e in starts:
        split=e.get('split','heldout' if int(e.get('repeat',0))%3==2 else 'train')
        if split in ('heldout','test','validation','holdout'):continue
        before=[s for s in samples if e['t']-.6<s['t']<e['t']-.04]
        launch=[s for s in samples if e['t']+.02<s['t']<e['t']+.18]
        if len(before)<5 or len(launch)<3:continue
        gravity=np.median([s['accel_g'] for s in before],axis=0);up=gravity/np.linalg.norm(gravity);up_vectors.append(up)
        a=np.mean([s['accel_g'] for s in launch],axis=0)-gravity;a-=np.dot(a,up)*up
        c=command(e)
        if np.max(abs(c))<.01:
            c=np.median([[s.get('forward',0),s.get('strafe',0)] for s in launch],axis=0)
        if abs(c[0])>.5 and abs(c[1])<.01:candidates.append(a*np.sign(c[0]))
    mount_source='signed_training_launches'
    if not candidates:
        mount_path=folder/'imu_calibration.json'
        if not mount_path.exists():mount_path=Path(__file__).resolve().parents[2]/'simulation/gazebo/imu_noise_profile.json'
        if not mount_path.exists():raise ValueError('No signed forward launches or prior stationary/launch mount profile')
        prior=json.loads(mount_path.read_text());candidates=[np.asarray(prior['body_to_sensor_matrix'])[:,0]];mount_source=str(mount_path)
        if not up_vectors:up_vectors=[np.asarray(prior['body_to_sensor_matrix'])[:,2]]
    up=np.median(up_vectors,axis=0);up/=np.linalg.norm(up)
    forward=np.median(candidates,axis=0);signal=float(np.linalg.norm(forward));forward-=forward.dot(up)*up
    forward/=max(np.linalg.norm(forward),1e-12);left=np.cross(up,forward);left/=np.linalg.norm(left)
    basis=np.column_stack((forward,left,up))
    refs=read(folder/'reference.jsonl') if (folder/'reference.jsonl').exists() else []
    refs=[r for r in refs if r.get('source') in ('external_metric_pose','visual_inertial_train_scaled')]
    trials=[];skipped=[]
    for e in starts:
        trial_id=key(e)
        end=next((x for x in events if x.get('event')=='trial_end' and key(x)==trial_id),None)
        if end is None:skipped.append(trial_id);continue
        pre=[s for s in samples if e['t']-.6<s['t']<e['t']-.04]
        post=[s for s in samples if end['t']-.35<s['t']<end['t']]
        rows=[s for s in samples if e['t']-.04<=s['t']<=end['t']]
        if min(len(pre),len(post),len(rows))<5:skipped.append(trial_id);continue
        t=np.array([s['t'] for s in rows]);frac=np.clip((t-t[0])/(t[-1]-t[0]),0,1)[:,None]
        b0=np.median([s['gyro_dps'] for s in pre],axis=0);b1=np.median([s['gyro_dps'] for s in post],axis=0)
        yaw=np.radians((np.array([s['gyro_dps'] for s in rows])-b0-(b1-b0)*frac)@up)
        g0=np.median([s['accel_g'] for s in pre],axis=0);g1=np.median([s['accel_g'] for s in post],axis=0)
        accel=(np.array([s['accel_g'] for s in rows])-g0-(g1-g0)*frac)@basis*9.80665
        theta=integral(yaw[:,None],t)[:,0];c,s=np.cos(theta),np.sin(theta)
        world=np.column_stack((c*accel[:,0]-s*accel[:,1],s*accel[:,0]+c*accel[:,1]))
        vel=integral(world,t);vel-=frac*vel[-1]
        velocity=np.column_stack((c*vel[:,0]+s*vel[:,1],-s*vel[:,0]+c*vel[:,1],yaw))
        source='imu_endpoint_constrained_PROVISIONAL'
        usable=[r for r in refs if t[0]<=r['t']<=t[-1]]
        active_times=[s['t'] for s in rows if any(s['wheels'].values())]
        motion_end=(max(active_times)+.25) if active_times else e['t']
        bad_motion_gap=any(r1['t']-r0['t']>.12 and r0['t']<motion_end and r1['t']>e['t'] for r0,r1 in zip(usable[:-1],usable[1:]))
        if len(usable)>=5 and usable[0]['t']<=e['t']+.1 and usable[-1]['t']>=end['t']-.2 and not bad_motion_gap:
            rt=np.array([r['t'] for r in usable]);rv=np.array([r['body_velocity'] for r in usable])
            velocity[:,:2]=np.column_stack([np.interp(t,rt,rv[:,axis]) for axis in (0,1)])
            source=usable[0]['source']
        split=e.get('split', 'heldout' if int(e.get('repeat',0))%3==2 else 'train')
        if split in ('test','validation','holdout'):split='heldout'
        trials.append(dict(id=trial_id,t=t.tolist(),wheels=[[s['wheels'].get(k,0)/4095 for k in WHEEL_ORDER] for s in rows],velocity=velocity.tolist(),split=split,translation_source=source))
    return trials,dict(body_axes_sensor=basis.tolist(),mount_source=mount_source,forward_launch_signal_g=signal if mount_source=='signed_training_launches' else None,skipped_trials=skipped)


def extract_visual_segments(folder):
    """Censor unobserved intervals, retaining original whole-trajectory split."""
    trials,mount=extract(folder);refs=read(folder/'reference.jsonl');segments=[];coverage=[]
    refs=[r for r in refs if r.get('source') in ('external_metric_pose','visual_inertial_train_scaled')]
    for tr in trials:
        tt=np.array(tr['t']);vv=np.array(tr['velocity']);ww=np.array(tr['wheels']);rows=[r for r in refs if tt[0]<=r['t']<=tt[-1]]
        chunks=[];chunk=[]
        for r in rows:
            if chunk and r['t']-chunk[-1]['t']>.12:chunks.append(chunk);chunk=[]
            chunk.append(r)
        if chunk:chunks.append(chunk)
        observed_seconds=0.
        for i,chunk in enumerate(chunks):
            if len(chunk)<4 or chunk[-1]['t']-chunk[0]['t']<.25:continue
            rt=np.array([r['t'] for r in chunk]);rv=np.array([r['body_velocity'] for r in chunk]);selected=(tt>=rt[0])&(tt<=rt[-1]);t=tt[selected]
            if len(t)<10:continue
            velocity=vv[selected].copy();velocity[:,:2]=np.column_stack([np.interp(t,rt,rv[:,axis]) for axis in (0,1)])
            segments.append(dict(id=f"{tr['id']}:{i}",base_trial=tr['id'],split=tr['split'],t=t.tolist(),velocity=velocity.tolist(),wheels=ww[selected].tolist(),translation_source=chunk[0]['source']))
            observed_seconds+=float(t[-1]-t[0])
        coverage.append(dict(trial=tr['id'],split=tr['split'],observed_segment_seconds=observed_seconds,total_trial_seconds=float(tt[-1]-tt[0]),fraction=observed_seconds/float(tt[-1]-tt[0])))
    mount['visual_segment_coverage']=coverage
    return segments,mount


def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--bootstrap',type=int,default=100);p.add_argument('--bayesian',action='store_true');p.add_argument('--visual-segments',action='store_true');p.add_argument('--posterior-draws',type=int,default=80);a=p.parse_args()
    trials,mount=extract_visual_segments(a.folder) if a.visual_segments else extract(a.folder)
    observed=[t for t in trials if t['translation_source'] in ('external_metric_pose','visual_inertial_train_scaled')]
    excluded=[]
    if observed:
        excluded=[t['id'] for t in trials if t not in observed];trials=observed
    profile=fit_bayesian(trials,draws=a.posterior_draws) if a.bayesian else fit(trials,bootstrap=a.bootstrap)
    profile['excluded_unobserved_translation_trials']=excluded
    profile['observation_scope']='contiguous visually observed segments; endpoints are segment endpoints, not full trajectory endpoints' if a.visual_segments else 'whole trajectories'
    verified=all(t['translation_source']=='external_metric_pose' for t in trials)
    profile.update(sensor_mount=mount,translation_status='externally_observed' if verified else ('visual_inertial_train_scaled' if all(t['translation_source']=='visual_inertial_train_scaled' for t in trials) else 'IMU_PROVISIONAL_NOT_VALIDATED'),
                   trial_sources=[dict(id=t['id'],split=t['split'],translation_source=t['translation_source']) for t in trials],
                   limitations=['Empirical PWM response; tire friction, motor torque, mass and slip are not separately identifiable.',
                                'Heldout velocity metrics compare against the declared reference; IMU translation is not independent ground truth.',
                                'Residual covariance models observed trial variation; it does not certify safety or out-of-distribution behavior.',
                                'Deadzone and time constants selected on training trials; bootstrap conditions on these selected values.'])
    out=a.folder/'mecanum_profile.json';out.write_text(json.dumps(profile,indent=2,allow_nan=False));print(out)
    print(json.dumps(profile['evaluation'],indent=2));print(profile['translation_status'])

if __name__=='__main__':main()
