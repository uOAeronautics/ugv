"""Replay recorded four-wheel duties through physical Gazebo mecanum mechanics.

No hardware access. References use mecanum_fit's declared IMU/vision extraction;
results retain its scale and sensor limitations. This never labels provisional
translation as independent sim2real validation. Execute with Gazebo control venv.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import xml.etree.ElementTree as E
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
GAZEBO=ROOT/'simulation/gazebo'
sys.path[:0]=[str(GAZEBO),str(ROOT/'src')]
from build_motor_plugin import plugin_environment
from native_motor_parameters import apply_native_parameters, extract_native_parameters
from mecanum_fit import extract, extract_visual_segments


def body_velocity(rows):
    t=np.array([p['t'] for p in rows]);keep=np.r_[True,np.diff(t)>1e-8]
    t=t[keep];x=np.array([p['x'] for p in rows])[keep];y=np.array([p['y'] for p in rows])[keep]
    yaw=np.unwrap(np.array([p['yaw'] for p in rows])[keep])
    if len(t)<5:raise RuntimeError('Insufficient Gazebo poses')
    vx=np.gradient(x,t);vy=np.gradient(y,t);w=np.gradient(yaw,t)
    return t,np.column_stack((np.cos(yaw)*vx+np.sin(yaw)*vy,-np.sin(yaw)*vx+np.cos(yaw)*vy,w))


def observed_endpoints(t, prediction, reference, observed):
    """Local rollouts on observed intervals; never bridge missing images."""
    indices=np.flatnonzero(observed);results=[]
    if not len(indices):return results
    chunks=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    for chunk in chunks:
        if len(chunk)<10 or t[chunk[-1]]-t[chunk[0]]<.25:continue
        local=t[chunk];out=[]
        for values in (prediction[chunk],reference[chunk]):
            heading=np.r_[0.,np.cumsum((values[:-1,2]+values[1:,2])*.5*np.diff(local))]
            fixed=np.column_stack((np.cos(heading)*values[:,0]-np.sin(heading)*values[:,1],
                                   np.sin(heading)*values[:,0]+np.cos(heading)*values[:,1]))
            xy=np.trapezoid(fixed,local,axis=0)
            out.append(np.r_[xy,heading[-1]])
        results.append(dict(start_s=float(local[0]),duration_s=float(local[-1]-local[0]),
                            gazebo=out[0].tolist(),reference=out[1].tolist(),error=(out[0]-out[1]).tolist()))
    return results


def metrics(residuals):
    if not residuals:return None
    r=np.concatenate(residuals)
    return dict(samples=len(r),rmse=np.sqrt(np.mean(r*r,axis=0)).tolist(),
                mae=np.mean(abs(r),axis=0).tolist(),bias=np.mean(r,axis=0).tolist(),
                residual_covariance=np.cov(r.T).tolist())


def enrich_metrics(output,traces):
    """Derive explicit valid-observation metrics from saved native traces."""
    output['observed_translation_active_by_split']={}
    output['primary_metrics_by_split']={}
    for split,whole in output['by_split'].items():
        rows=[];observed_count=0;total_count=0
        for trace in traces:
            if trace['split']!=split:continue
            observed=np.array(trace['translation_observed_mask']);active=np.max(np.abs(trace['recorded_hardware_wheels']),axis=1)>1e-8
            selected=observed & active
            error=np.array(trace['gazebo_body_velocity'])-np.array(trace['reference_body_velocity'])
            if selected.any():rows.append(error[selected])
            observed_count+=int(observed.sum());total_count+=len(observed)
        active_metrics=metrics(rows)
        output['observed_translation_active_by_split'][split]=active_metrics
        observed_metrics=output['observed_translation_by_split'][split]
        output['primary_metrics_by_split'][split]=dict(
            observed_translation_rmse_mps=observed_metrics['rmse'][:2] if observed_metrics else None,
            observed_active_translation_rmse_mps=active_metrics['rmse'][:2] if active_metrics else None,
            all_samples_gyro_yaw_rmse_radps=whole['rmse'][2],
            visual_observation_fraction=observed_count/max(1,total_count))
    return output


def run(args):
    trials,mount=extract(args.folder)
    visual_segments,_=extract_visual_segments(args.folder)
    if args.split!='all':trials=[t for t in trials if t['split']==args.split]
    if args.limit:trials=trials[:args.limit]
    if not trials:raise ValueError('No completed trials selected')
    os.environ['GZ_PARTITION']='ugv-mecanum-replay-'+uuid.uuid4().hex
    from gz.transport import Node
    from gz.msgs.double_v_pb2 import Double_V
    from gz.msgs.pose_v_pb2 import Pose_V
    env=plugin_environment();env['GZ_PARTITION']=os.environ['GZ_PARTITION']
    root=E.parse(args.world)
    drive=root.find('.//model[@name="freenove"]/plugin[@name="ugv::TTMotorSystem"]')
    if drive is None or drive.findtext('mecanum')!='true':raise ValueError('World must use physical mecanum motor plugin')
    params=extract_native_parameters(root)
    profile_label=drive.findtext('calibration_profile') or 'embedded world motor parameters'
    if args.nominal_motors:
        from motor_physics import load_profile
        nominal=load_profile()
        params=apply_native_parameters(root,dict(no_load_speed=nominal['omega0'],stall_torque=nominal['stall_nm'],
            brake_damping=nominal['brake_damping_nm_s_per_rad_assumed'],brake_resistance=nominal['brake_resistance_nm_assumed'],
            wheel_voltage_gain=[1.]*4,wheel_reverse_voltage_gain=[1.]*4,wheel_duty_exponent=[1.]*4,
            wheel_deadzone=[0.]*4,wheel_drive_tau=[0.]*4,wheel_brake_tau=[0.]*4,wheel_gain_stddev=[0.]*4,random_seed=0,
            roller_bearing_damping=.000002,roller_bearing_friction=.000001))
        profile_label='nominal reference motor/contact assumptions'
    elif args.motor_parameters:
        params=apply_native_parameters(root,json.loads(args.motor_parameters.read_text()))
        profile_label=str(args.motor_parameters)

    # Headless mechanics uses the same links, contacts and inertias; camera/IMU
    # render streams are unnecessary for comparison against ideal body poses.
    for world in root.findall('world'):
        for plugin in list(world.findall('plugin')):
            if 'Sensors' in plugin.get('name',''):world.remove(plugin)
    for link in root.findall('.//link'):
        for visual in list(link.findall('visual')):link.remove(visual)
        for sensor in list(link.findall('sensor')):link.remove(sensor)
    node=Node();pub=node.advertise('/model/freenove/wheel_duty',Double_V)
    poses=[];motor_samples=[]
    def receive(message):
        stamp=message.header.stamp;t=stamp.sec+stamp.nsec*1e-9
        for p in message.pose:
            if p.name=='freenove':
                q=p.orientation
                poses.append(dict(t=t,x=p.position.x,y=p.position.y,z=p.position.z,
                    yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))))
    node.subscribe(Pose_V,'/world/freenove_lab/dynamic_pose/info',receive)
    node.subscribe(Double_V,'/model/freenove/motor_state',lambda msg:motor_samples.append(list(msg.data)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=tempfile.TemporaryDirectory(prefix='ugv-mecanum-replay-')
    world=Path(temporary.name)/'world.sdf';root.write(world)
    log=args.output.with_suffix('.log').open('w')
    process=subprocess.Popen(['gz','sim','-s','-r',str(world)],cwd=GAZEBO,env=env,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    def publish(values):
        message=Double_V();message.data.extend(values);pub.publish(message)
    def wait_sim(duration):
        start=poses[-1]['t'];deadline=time.monotonic()+duration*15+10
        while poses[-1]['t']-start<duration:
            publish([0.]*4);time.sleep(.02)
            if time.monotonic()>deadline:raise RuntimeError('Gazebo clock stopped')
    results=[];traces=[];residuals={};active_residuals={}
    try:
        deadline=time.monotonic()+25
        while (not poses or not motor_samples) and time.monotonic()<deadline:time.sleep(.1)
        if not poses or not motor_samples:raise RuntimeError('Gazebo motor system did not initialize')
        wait_sim(.6)
        for trial in trials:
            tt=np.array(trial['t']);tt-=tt[0];duration=float(tt[-1])
            duties=-np.array(trial['wheels'])[:,[0,2,1,3]] # hardware -> physical FL FR RL RR
            start=poses[-1]['t'];index=len(poses)-1;deadline=time.monotonic()+duration*15+10
            while poses[-1]['t']-start<duration:
                elapsed=poses[-1]['t']-start
                i=min(len(tt)-1,max(0,int(np.searchsorted(tt,elapsed,side='right')-1)))
                publish(duties[i]);time.sleep(.012)
                if time.monotonic()>deadline:raise RuntimeError('Gazebo replay exceeded clock deadline')
            publish([0.]*4)
            selected=poses[index:];st,sv=body_velocity(selected);st-=start
            reference=np.array(trial['velocity'])
            # Metric translation is scored only inside contiguous accepted
            # visual segments. Gyro yaw is retained over the whole trajectory.
            observed=np.zeros(len(tt),dtype=bool)
            absolute=np.array(trial['t'])
            for segment in visual_segments:
                if segment['base_trial']!=trial['id']:continue
                segment_t=np.array(segment['t']);mask=(absolute>=segment_t[0])&(absolute<=segment_t[-1])
                observed|=mask
                values=np.array(segment['velocity'])
                for axis in (0,1):reference[mask,axis]=np.interp(absolute[mask],segment_t,values[:,axis])
            prediction=np.column_stack([np.interp(tt,st,sv[:,j]) for j in range(3)])
            error=prediction-reference
            residuals.setdefault(trial['split'],[]).append(error)
            active=np.max(np.abs(duties),axis=1)>1e-8
            if active.any():active_residuals.setdefault(trial['split'],[]).append(error[active])
            result=dict(id=trial['id'],split=trial['split'],duration_s=duration,
                translation_source=trial['translation_source'],**metrics([error]))
            result['active_command_metrics']=metrics([error[active]]) if active.any() else None
            result['observed_translation_fraction']=float(np.mean(observed))
            result['observed_translation_metrics']=metrics([error[observed]]) if observed.any() else None
            result['yaw_all_samples_rmse']=float(np.sqrt(np.mean(error[:,2]**2)))
            result['observed_segment_endpoint_errors']=observed_endpoints(tt,prediction,reference,observed)
            results.append(result)
            traces.append(dict(id=trial['id'],split=trial['split'],time_s=tt.tolist(),
                recorded_hardware_wheels=trial['wheels'],gazebo_body_velocity=prediction.tolist(),
                reference_body_velocity=reference.tolist(),translation_observed_mask=observed.tolist()))
            print(json.dumps(dict(trial=trial['id'],split=trial['split'],rmse=result['rmse'])),flush=True)
            wait_sim(.5)
        independent=all(t['translation_source']=='external_metric_pose' for t in trials)
        output=dict(schema_version=2,engine='Gazebo physical passive mecanum rollers + finite motor torque',
            source_run=str(args.folder.resolve()),world=str(args.world.resolve()),
            world_sha256=hashlib.sha256(args.world.read_bytes()).hexdigest(),native_parameters=params,
            metric_scopes=dict(observed_translation_by_split='Only contiguous accepted visual segments; translation observed but metric scale shares IMU',by_split='Full trajectories; X/Y may include provisional endpoint-constrained IMU integration; do not use for translation acceptance',yaw='Gyro-referenced yaw over all completed samples'),
            motor_parameters=profile_label,
            axes=['forward_mps','left_mps','ccw_radps'],sensor_mount=mount,
            independent_metric_reference=independent,
            acceptance='evaluation_only_requires_external_pose_and_thresholds' if independent else 'PROVISIONAL_NOT_INDEPENDENT_REAL2SIM_VALIDATION',
            limitations=['Visual inertial translation has systematic scale uncertainty; IMU endpoint constraints are not external ground truth.',
                        'Metrics compare heldout trials only when split=heldout is present; probe trials are not heldout.',
                        'Nominal motor and contact parameters are assumptions unless a fitted parameter file is explicitly supplied.'],
            by_split={key:metrics(value) for key,value in residuals.items()},
            by_split_active_command={key:metrics(value) for key,value in active_residuals.items()},
            observed_translation_by_split={key:metrics([np.array(t['gazebo_body_velocity'])[np.array(t['translation_observed_mask'])]-np.array(t['reference_body_velocity'])[np.array(t['translation_observed_mask'])] for t in traces if t['split']==key and any(t['translation_observed_mask'])]) for key in residuals},
            observed_segment_endpoint_by_split={key:metrics([np.array([e['error'] for r in results if r['split']==key for e in r['observed_segment_endpoint_errors']])]) if any(r['observed_segment_endpoint_errors'] for r in results if r['split']==key) else None for key in residuals},
            endpoint_axes=['forward_displacement_m','left_displacement_m','yaw_displacement_rad'],
            endpoint_scope='Local rollouts within observed visual intervals; both headings anchored at each interval start; not full-trajectory endpoints',
            trials=results,
            maximum_torque_nm=max(abs(s[i]) for s in motor_samples if len(s)==16 for i in (2,6,10,14)))
        enrich_metrics(output,traces)
        args.output.write_text(json.dumps(output,indent=2,allow_nan=False))
        args.output.with_suffix('.traces.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in traces))
        print(args.output,flush=True)
        return output
    finally:
        publish([0.]*4)
        os.killpg(process.pid,signal.SIGINT)
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGTERM)
        log.close();temporary.cleanup()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--world',type=Path,default=GAZEBO/'freenove_world.sdf')
    motors=parser.add_mutually_exclusive_group()
    motors.add_argument('--motor-parameters',type=Path)
    motors.add_argument('--nominal-motors',action='store_true',help='Override embedded candidate parameters with nominal references')
    parser.add_argument('--output',type=Path,default=GAZEBO/'mecanum_real2sim_validation.json')
    parser.add_argument('--split',default='all',choices=('all','train','probe','heldout'))
    parser.add_argument('--limit',type=int,default=0)
    run(parser.parse_args())
