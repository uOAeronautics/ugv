"""Evaluate a frozen Bayesian model on complete guarded camera trajectories.

No fitting. Guard odometry has training-derived metric scale, not surveyed truth.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ugv.mecanum_dynamics import rollout


def read(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def rotate(angle,v):
    c,s=np.cos(angle),np.sin(angle);return np.array([c*v[0]-s*v[1],s*v[0]+c*v[1]])


def extract_guarded_reference(folder,wheel_order=('front-left','rear-left','front-right','rear-right')):
    """Extract complete guarded camera pose/velocity; never integrate IMU translation.

    Yaw uses stationary-bias-corrected gyro and the guard's visual pose shares
    its estimated camera height. Arrays retain the original Pi monotonic clock.
    """
    events=read(folder/'events.jsonl');samples=read(folder/'samples.jsonl');guard=read(folder/'visual_guard.jsonl')
    st=np.array([row['t'] for row in samples]);gyro=np.array([row['gyro_dps'] for row in samples]);accel=np.array([row['accel_g'] for row in samples])
    wheels=np.array([[row['wheels'][key]/4095 for key in wheel_order] for row in samples])
    accepted=[row for row in guard if row['accepted']];gt=np.array([row['frame_t'] for row in accepted]);positions=np.array([row['position_world_m'] for row in accepted]);guard_times=np.array([row['frame_t'] for row in guard]);trials=[]
    for event in events:
        if event.get('event')!='trial_start':continue
        key=event.get('trial',event.get('id'))
        end=next((row for row in events if row.get('event')=='drive_end' and row.get('trial',row.get('id'))==key),None)
        stopped=next((row for row in events if row.get('event')=='trial_end' and row.get('trial',row.get('id'))==key),None)
        if end is None or stopped is None:continue
        start,finish=event['t'],end['t'];pre=(st>start-.8)&(st<start-.1);post=(st>stopped['t']-.5)&(st<stopped['t'])
        if pre.sum()<5 or post.sum()<5:raise ValueError('Guarded trial lacks stationary gyro baseline')
        up=np.median(accel[pre],axis=0);up/=np.linalg.norm(up);b0=np.median(gyro[pre],axis=0);b1=np.median(gyro[post],axis=0)
        fraction=np.clip((st-start)/(stopped['t']-start),0,1)[:,None]
        yawrate=np.radians((gyro-b0-fraction*(b1-b0))@up)
        yaw=np.r_[0,np.cumsum((yawrate[:-1]+yawrate[1:])*.5*np.diff(st))];yaw-=np.interp(start,st,yaw)
        t=np.r_[start,st[(st>start)&(st<finish)],finish]
        indices=np.clip(np.searchsorted(st,t,side='right')-1,0,len(st)-1)
        active=[row for row in guard if start<row['frame_t']<=finish]
        maximum_gap=float(np.max(np.diff([start,*[row['frame_t'] for row in active],finish])))
        complete=bool(active and all(row['accepted'] and row['healthy'] for row in active) and gt[0]<=start and gt[-1]>=finish)
        origin=np.array([np.interp(start,gt,positions[:,axis]) for axis in (0,1)])
        angle=np.interp(start,st,yaw)-np.interp(guard[0]['frame_t'],st,yaw)
        xy=np.array([rotate(-angle,np.array([np.interp(q,gt,positions[:,axis]) for axis in (0,1)])-origin) for q in t])
        pose=np.column_stack((xy,np.interp(t,st,yaw)))
        vt=[];velocity=[]
        for row in accepted:
            index=np.searchsorted(guard_times,row['frame_t']);previous=guard_times[index-1] if index>0 else row['frame_t'];dt=row['frame_t']-previous;mid=(row['frame_t']+previous)/2
            if dt<=0 or not start<=mid<=finish or 'delta_body_m' not in row:continue
            dyaw=np.interp(row['frame_t'],st,yaw)-np.interp(previous,st,yaw)
            vt.append(mid);velocity.append([*rotate(-dyaw/2,np.array(row['delta_body_m'])/dt),float(np.interp(mid,st,yawrate))])
        trials.append(dict(id=key,t=t,wheels=wheels[indices],reference_pose=pose,
            velocity_times=np.array(vt),reference_velocity=np.array(velocity),pwm=event['pwm'],
            recorder_split=event.get('split'),complete_guard_observation=complete,
            maximum_guard_frame_gap_s=maximum_gap,accepted_guard_pairs=sum(row['accepted'] for row in active),guard_pairs=len(active),
            imu_up_sensor=up.tolist(),gyro_bias_before_dps=b0.tolist(),gyro_bias_after_dps=b1.tolist()))
    return trials


def evaluate(folder,profile_path,output,seed=101):
    profile_blob=profile_path.read_bytes();profile=json.loads(profile_blob);events=read(folder/'events.jsonl');samples=read(folder/'samples.jsonl');guard=read(folder/'visual_guard.jsonl');manifest=json.loads((folder/'manifest.json').read_text())
    st=np.array([s['t'] for s in samples]);all_gyro=np.array([s['gyro_dps'] for s in samples]);all_acc=np.array([s['accel_g'] for s in samples]);all_w=np.array([[s['wheels'][k]/4095 for k in profile['wheel_order']] for s in samples]);grows=[g for g in guard if g['accepted']];gt=np.array([g['frame_t'] for g in grows]);gp=np.array([g['position_world_m'] for g in grows]);gv=np.array([g['delta_body_m'] for g in grows]);guard_all_t=np.array([g['frame_t'] for g in guard]);results=[]
    for event in events:
        if event.get('event')!='trial_start':continue
        key=event.get('trial',event.get('id'));end=next((e for e in events if e.get('event')=='drive_end' and e.get('trial',e.get('id'))==key),None);trial_end=next((e for e in events if e.get('event')=='trial_end' and e.get('trial',e.get('id'))==key),None)
        if end is None or trial_end is None:continue
        start,finish=event['t'],end['t'];pre=(st>start-.8)&(st<start-.1);post=(st>trial_end['t']-.5)&(st<trial_end['t']);up=np.median(all_acc[pre],axis=0);up/=np.linalg.norm(up);b0=np.median(all_gyro[pre],axis=0);b1=np.median(all_gyro[post],axis=0);frac=np.clip((st-start)/(trial_end['t']-start),0,1)[:,None];yawrate=np.radians((all_gyro-b0-frac*(b1-b0))@up);yaw=np.r_[0,np.cumsum((yawrate[1:]+yawrate[:-1])*.5*np.diff(st))];yaw-=np.interp(start,st,yaw)
        t=np.r_[start,st[(st>start)&(st<finish)],finish];indices=np.clip(np.searchsorted(st,t,side='right')-1,0,len(st)-1);wheels=all_w[indices]
        active_guard=[g for g in guard if start<g['frame_t']<=finish];complete=bool(active_guard and all(g['accepted'] and g['healthy'] for g in active_guard) and gt[0]<=start and gt[-1]>=finish)
        guard_start=np.array([np.interp(start,gt,gp[:,i]) for i in (0,1)])
        # Guard world axes start at its stationary visual origin, before trial.
        origin_t=guard[0]['frame_t'];start_angle=np.interp(start,st,yaw)-np.interp(origin_t,st,yaw)
        ref_xy=np.array([rotate(-start_angle,np.array([np.interp(q,gt,gp[:,i]) for i in (0,1)])-guard_start) for q in t]);ref_yaw=np.interp(t,st,yaw);reference_pose=np.column_stack((ref_xy,ref_yaw))
        vt=[];vel=[]
        for g in grows:
            i=np.searchsorted(guard_all_t,g['frame_t']);previous_t=guard_all_t[i-1] if i>0 else g['frame_t'];dt=g['frame_t']-previous_t
            if dt<=0:continue
            mid=(g['frame_t']+previous_t)/2
            if not start<=mid<=finish:continue
            dyaw=np.interp(g['frame_t'],st,yaw)-np.interp(previous_t,st,yaw)
            vt.append(mid);vel.append([*rotate(-dyaw/2,np.array(g['delta_body_m'])/dt),float(np.interp(mid,st,yawrate))])
        vt=np.array(vt);vel=np.array(vel)
        velocities=[];poses=[]
        for i,particle in enumerate(profile['posterior_particles']):
            v,p=rollout(wheels,t,particle,initial=[0,0,0],seed=seed+i);velocities.append(v);poses.append(p)
        velocities=np.asarray(velocities);poses=np.asarray(poses);predicted=np.array([np.column_stack([np.interp(vt,t,v[:,i]) for i in range(3)]) for v in velocities]);mean=predicted.mean(axis=0);low,high=np.percentile(predicted,[2.5,97.5],axis=0);pose_mean=poses.mean(axis=0);pose_low,pose_high=np.percentile(poses,[2.5,97.5],axis=0);endpoint_error=pose_mean[-1]-reference_pose[-1];active_pairs=[g['frame_t'] for g in active_guard];maxgap=float(np.max(np.diff([start,*active_pairs,finish])))
        results.append(dict(trial=key,evaluation_split='new_frozen_model_evaluation; recorder split label is ignored',recorder_split=event.get('split'),pwm=event['pwm'],duration_s=finish-start,complete_guard_observation=complete,accepted_guard_pairs=sum(g['accepted'] for g in active_guard),guard_pairs=len(active_guard),maximum_guard_frame_gap_s=maxgap,observed_velocity_samples=len(vt),velocity_rmse=np.sqrt(np.mean((mean-vel)**2,axis=0)).tolist(),velocity_interval_95_coverage=np.mean((vel>=low)&(vel<=high),axis=0).tolist(),pose_time_rmse=np.sqrt(np.mean((pose_mean-reference_pose)**2,axis=0)).tolist(),endpoint_reference=reference_pose[-1].tolist(),endpoint_predicted_mean=pose_mean[-1].tolist(),endpoint_error=endpoint_error.tolist(),endpoint_position_error_norm_m=float(np.linalg.norm(endpoint_error[:2])),endpoint_interval_95_low=pose_low[-1].tolist(),endpoint_interval_95_high=pose_high[-1].tolist(),endpoint_covered=((reference_pose[-1]>=pose_low[-1])&(reference_pose[-1]<=pose_high[-1])).tolist(),maximum_observed_yaw_rate_dps=float(np.degrees(np.max(abs(np.interp(t,st,yawrate))))),imu_up_sensor=up.tolist(),gyro_bias_before_dps=b0.tolist(),gyro_bias_after_dps=b1.tolist()))
    report=dict(schema_version=1,source_run=folder.name,status='FROZEN_MODEL_RELATIVE_VISUAL_INERTIAL_EVALUATION',model_artifact=profile.get('artifact_version'),model_sha256=hashlib.sha256(profile_blob).hexdigest(),model_unchanged=True,model_training_pwm=[1400,2600,4095],model_posterior_particles=len(profile['posterior_particles']),recording_sha256={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ['samples.jsonl','events.jsonl','visual_guard.jsonl']},camera_height_prior_m=manifest.get('visual_guard',{}).get('height_prior_m'),requested_camera_provenance=manifest.get('requested_camera_provenance'),trials=results,limitations=['New trajectories are evaluation only despite recorder split labels; no refit, scale fitting or hyperparameter tuning is performed here.','Guard pose uses training-derived floor height, estimated plane and camera lever arm; no surveyed external metric pose.','Reference yaw and visual rotation compensation share the IMU, so yaw is not an independent optical check.','JPEG receipt times are not hardware exposure times; interpolation and velocity differencing add temporal uncertainty.','Predictive intervals condition on the fitted model and shared visual scale; systematic scale and mounting uncertainty are not fully propagated.'])
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--profile',type=Path,default=Path('simulation/gazebo/mecanum_bayesian_profile.json'));p.add_argument('--output',type=Path,default=Path('simulation/gazebo/mecanum_guarded_ood_validation.json'));a=p.parse_args();r=evaluate(a.folder,a.profile,a.output)
    for t in r['trials']:print(json.dumps({k:t[k] for k in ['trial','pwm','duration_s','complete_guard_observation','velocity_rmse','velocity_interval_95_coverage','endpoint_error','endpoint_position_error_norm_m','endpoint_covered']}))
