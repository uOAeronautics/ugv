"""Bounded full-range mecanum identification through the Pi's normal owner API.

Run on the Pi so events, raw IMU and camera receipt timestamps share one clock.
Each energized command has a server-enforced 250ms lease, renewed at 50ms.
"""
import argparse
import bisect
import asyncio
import json
import math
import os
from pathlib import Path
import statistics
import time
from urllib.request import urlopen
import websockets


def fetch(url):
    with urlopen(url, timeout=2) as response:
        return json.load(response)


def plan(probe=False, trajectories=False):
    vectors = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),
               (1,1,0),(-1,-1,0),(1,-1,0),(-1,1,0),
               (0,0,1),(0,0,-1),(1,0,1),(-1,0,-1),(0,1,1),(0,-1,-1)]
    if trajectories:
        result=[]
        for repeat in range(3):
            for pwm in (600,1000,1400,1800,2600,3400,4095):
                for kind in ('translation','rotation','coupled'):
                    phases=[]
                    for i in range(20):
                        a=i/19*math.pi
                        if kind=='translation':
                            f,s,t=math.cos(a),math.sin(a),0
                        elif kind=='rotation':
                            f,s,t=0,0,math.sin(a)*(.65 if pwm>=3400 else 1)
                        else:
                            f,s,t=.6*math.cos(a),.45*math.sin(2*a),.65*math.sin(a)
                        phases.append(dict(forward=f,strafe=s,turn=t,pwm=pwm,duration=.2))
                    # Time reverse with opposite body twist: a closed path in
                    # ideal SE(2), while exercising inertia/reversal asymmetry.
                    phases += [dict(p,forward=-p['forward'],strafe=-p['strafe'],turn=-p['turn']) for p in reversed(phases)]
                    result.append(dict(forward=phases[0]['forward'],strafe=phases[0]['strafe'],turn=phases[0]['turn'],
                        pwm=pwm,duration=8.,repeat=repeat,split='heldout' if repeat==2 else 'train',
                        trajectory=kind,phases=phases))
        return result
    if probe:
        return [dict(forward=f,strafe=s,turn=t,pwm=p,duration=.16 if p<=1800 else .10,repeat=0,split='probe')
                for p in (700,1400,2200,3200,4095) for f,s,t in vectors[:4]+vectors[8:10]]
    return [dict(forward=f,strafe=s,turn=t,pwm=p,
                 duration=.65 if p<=1400 else (.45 if p<=2600 else .30),
                 repeat=r,split='heldout' if r==2 else 'train')
            for r in range(3) for p in (300,600,1000,1400,1800,2600,3400,4095)
            for f,s,t in vectors]


async def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'frames').mkdir()
    base=args.url.rstrip('/')
    initial=await asyncio.to_thread(fetch,base+'/api/state')
    if initial['state']['armed'] or initial['state']['owner'] or initial['state']['parking']['active']:
        raise RuntimeError('Robot already owned or autonomously active')
    start=await asyncio.to_thread(fetch,base+'/api/dynamics?after=0')
    seq=start['latest_seq']; samples=[]; live=True; state={}; errors=[]; frames=0
    visual_guard=None;visual_state=None;visual_bias=None;visual_up=None;visual_previous=None;visual_yaw=0.;visual_yaw_times=[];visual_yaw_values=[]
    if args.visual_radius:
        import cv2
        import numpy as np
        from mecanum_online_guard import PlaneHomographyGuard
    visual_output=(out/'visual_guard.jsonl').open('w',buffering=1) if args.visual_radius else None
    outputs={n:(out/(n+'.jsonl')).open('w',buffering=1) for n in ('samples','events','frames')}
    trials=plan(args.probe,args.trajectories)
    if args.loop_seconds:
        if not args.trajectories or not .8 <= args.loop_seconds <= 4:
            raise ValueError('Loop period requires trajectories and .8..4 seconds')
        for command in trials:
            if command.get('trajectory') == 'translation':
                command['loop_period_s']=args.loop_seconds
                for i,phase in enumerate(command['phases']):
                    a=math.pi+2*math.pi*i*.2/args.loop_seconds
                    phase.update(forward=math.cos(a),strafe=math.sin(a))
                command.update(forward=-1.,strafe=0.)
            elif command.get('trajectory') == 'rotation':
                command['loop_period_s']=args.loop_seconds
                for i,phase in enumerate(command['phases']):
                    amplitude=.65 if command['pwm']>=2600 else 1.
                    phase['turn']=amplitude*math.sin(2*math.pi*i*.2/args.loop_seconds)
                command.update(turn=0.)
    if args.rotate_degrees:
        trials=[dict(forward=0,strafe=0,turn=1 if args.rotate_degrees>0 else -1,pwm=900,
            duration=6,repeat=0,split='probe',target_yaw_deg=abs(args.rotate_degrees))]
    if args.pilot:
        trials=[c for c in trials if c.get('trajectory')=='translation' and c['pwm'] in (1400,2600,4095)]
    if args.powers:trials=[c for c in trials if c['pwm'] in args.powers]
    if args.kinds:trials=[c for c in trials if c.get('trajectory') in args.kinds]
    if args.repeats:trials=[c for c in trials if c['repeat'] in args.repeats]
    if args.limit:trials=trials[:args.limit]
    if not trials and not args.stationary_seconds:raise ValueError('No trials selected')
    metadata=dict(schema_version=2,clock='Pi monotonic seconds',status='running',
        explicit_authorization='User authorized autonomous clear-floor full-range live identification',
        initial_state=initial,plan=trials,heldout='entire repeat 2, never used for fitting',
        camera_time='Pi receipt timestamp for each JPEG, not hardware exposure timestamp',
        started_utc_untrusted_pi_clock=time.time(),
        visual_guard=dict(enabled=bool(args.visual_radius),radius_m=args.visual_radius,
            height_prior_m=args.visual_height,floor_normal_camera=args.visual_normal,
            initial_origin_only=True,invalid_pair_policy='stop and disarm; no tracking reset between trials',
            scale_source='training estimated camera height, not surveyed') if args.visual_radius else None,
        camera_capture_environment={key:value for key,value in os.environ.items() if key.startswith('UGV_CAMERA_')},
        requested_camera_provenance=dict(exposure_us=args.camera_exposure_us,analogue_gain=args.camera_gain,
            note='Recorder provenance only; camera service controls exposure and gain'))
    def save_manifest():
        (out/'manifest.json').write_text(json.dumps(metadata,indent=2))
    save_manifest()
    def mark(event,**values):
        t=time.monotonic();outputs['events'].write(json.dumps(dict(event=event,t=t,**values))+'\n');return t
    def camera_frame():
        with urlopen(base+'/api/calibration/frame',timeout=2) as response:
            return float(response.headers['X-Pi-Monotonic']),int(response.headers['X-Frame-Sequence']),response.read()
    async with websockets.connect(base.replace('http','ws',1)+'/ws',open_timeout=5,max_size=2**22) as ws:
        async def receive():
            async for raw in ws:
                message=json.loads(raw)
                if message.get('type')=='error':errors.append(message['error'])
                if message.get('type')=='state':state.clear();state.update(message['state'])
        async def collect():
            nonlocal seq,visual_previous,visual_yaw
            while live:
                message=await asyncio.to_thread(fetch,base+f'/api/dynamics?after={seq}')
                if message['latest_seq']<seq or message['dropped']:
                    raise RuntimeError('Telemetry overflow/restart')
                for sample in message['samples']:
                    outputs['samples'].write(json.dumps(sample,separators=(',',':'))+'\n')
                    samples.append(sample)
                    if visual_bias is not None:
                        if visual_previous is not None:
                            dt=sample['t']-visual_previous['t']
                            if not 0<dt<.2:raise RuntimeError('Visual guard gyro integration gap')
                            visual_yaw+=math.radians(sum((sample['gyro_dps'][i]+visual_previous['gyro_dps'][i]-2*visual_bias[i])*.5*visual_up[i] for i in range(3)))*dt
                        visual_previous=sample
                        visual_yaw_times.append(sample['t']);visual_yaw_values.append(visual_yaw)
                seq=message['latest_seq']
                await asyncio.sleep(.035)
        async def images():
            nonlocal frames,visual_state
            previous=-1
            while live:
                t,sequence,jpeg=await asyncio.to_thread(camera_frame)
                if sequence!=previous:
                    relative=f'frames/{sequence:09d}.jpg'
                    (out/relative).write_bytes(jpeg)
                    outputs['frames'].write(json.dumps(dict(t=t,seq=sequence,file=relative))+'\n')
                    previous=sequence;frames+=1
                    if visual_guard is not None:
                        # CPU work runs off the lease-renewal event loop. Receipt
                        # timestamp and gyro horizon mismatch are checked below.
                        if visual_previous is None or abs(t-visual_previous['t'])>.15:
                            raise RuntimeError('Visual guard camera/gyro timestamps out of sync')
                        image=cv2.imdecode(np.frombuffer(jpeg,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
                        index=bisect.bisect_right(visual_yaw_times,t)
                        if index==0:
                            # A cached JPEG may predate the newly established
                            # stationary origin; wait for a newer frame.
                            await asyncio.sleep(.04)
                            continue
                        if index<len(visual_yaw_times):
                            fraction=(t-visual_yaw_times[index-1])/(visual_yaw_times[index]-visual_yaw_times[index-1])
                            frame_yaw=visual_yaw_values[index-1]+fraction*(visual_yaw_values[index]-visual_yaw_values[index-1])
                        else:
                            rate=math.radians(sum((visual_previous['gyro_dps'][i]-visual_bias[i])*visual_up[i] for i in range(3)))
                            frame_yaw=visual_yaw+rate*(t-visual_previous['t'])
                        result=await asyncio.to_thread(visual_guard.update,image,t,frame_yaw)
                        visual_state=dict(result,frame_t=t,checked_at=time.monotonic())
                        visual_output.write(json.dumps(visual_state)+'\n')
                await asyncio.sleep(.005 if visual_guard is not None else .04)
        tasks=[asyncio.create_task(fn()) for fn in (receive,collect,images)]
        async def send(message):await ws.send(json.dumps(message))
        def check(owned=True):
            for task in tasks:
                if task.done():task.result();raise RuntimeError('Capture task ended')
            if errors:raise RuntimeError(errors[-1])
            if visual_guard is not None:
                if visual_state is not None and not visual_state['healthy']:
                    raise RuntimeError('Visual workspace tracking lost: '+visual_state['reason'])
                if visual_state is not None and visual_state['radius_m']>args.visual_radius:
                    raise RuntimeError('Visual workspace radius exceeded')
                if owned and (visual_state is None or not visual_state['fresh'] or time.monotonic()-visual_state['frame_t']>.25):
                    raise RuntimeError('Visual workspace estimate stale before motion')
            if not samples or time.monotonic()-samples[-1]['t']>.20:raise RuntimeError('Stale IMU')
            if owned and (not state.get('armed') or not state.get('you_are_owner')):raise RuntimeError('Lost ownership')
            if max(abs(v) for v in samples[-1]['gyro_dps'])>850:raise RuntimeError('Angular rate exceeded campaign bound')
            window=[s for s in samples if s['t']>samples[-1]['t']-.12]
            recent=[statistics.median(s['accel_g'][i] for s in window) for i in range(3)]
            norm=math.sqrt(sum(v*v for v in recent))
            if norm<.4 or norm>3:raise RuntimeError('Impact/tilt acceleration exceeded bound')
            if math.sqrt(sum(v*v for v in samples[-1]['accel_g']))>3:raise RuntimeError('Impact/tilt acceleration exceeded bound')
        async def rest(seconds,owned=True):
            end=time.monotonic()+seconds
            while time.monotonic()<end:
                await send(dict(type='heartbeat'));await asyncio.sleep(.07);check(owned)
        def quiet():
            recent=[s for s in samples if s['t']>time.monotonic()-.6]
            return len(recent)>=35 and all(statistics.pstdev(s['gyro_dps'][i] for s in recent)<2
                and statistics.pstdev(s['accel_g'][i] for s in recent)<.045 for i in range(3))
        try:
            await asyncio.sleep(.5)
            mark('baseline_start');await rest(args.stationary_seconds or 10,False);mark('baseline_end')
            if not quiet():raise RuntimeError('Unstable initial rest')
            if args.stationary_seconds:
                metadata['status']='complete';return
            if args.visual_radius:
                baseline=samples[-100:]
                gravity=[statistics.median(s['accel_g'][i] for s in baseline) for i in range(3)]
                norm=math.sqrt(sum(v*v for v in gravity));visual_up=[v/norm for v in gravity]
                visual_bias=[statistics.median(s['gyro_dps'][i] for s in baseline) for i in range(3)]
                visual_yaw=0.;visual_previous=samples[-1]
                visual_yaw_times.append(visual_previous['t']);visual_yaw_values.append(0.)
                visual_guard=PlaneHomographyGuard(height_m=args.visual_height,
                    floor_normal_camera=args.visual_normal,max_radius_m=args.visual_radius,width=480,max_gap_s=.25)
                metadata['visual_guard'].update(gyro_bias_dps=visual_bias,gyro_up_sensor=visual_up)
                metadata['visual_guard']['processing']=visual_guard.processing_configuration
                mark('visual_origin_set',origin='initial stationary baseline only')
                # Initializing is allowed only while disarmed. Any failed pair
                # latches unhealthy and prevents arming.
                deadline=time.monotonic()+3
                while time.monotonic()<deadline:
                    await asyncio.sleep(.05);check(False)
                    if visual_state and visual_state['fresh']:break
                if visual_state is None or not visual_state['fresh']:raise RuntimeError('Visual workspace failed to initialize')
            await send(dict(type='arm'));await asyncio.sleep(.2);check()
            await send(dict(type='aruco',enabled=False,follow=False))
            await asyncio.sleep(.1)
            for trial,command in enumerate(trials):
                await rest(.7)
                deadline=time.monotonic()+8
                while not quiet() and time.monotonic()<deadline:await rest(.3)
                if not quiet():raise RuntimeError('No stable pretrial rest')
                mark('trial_start',trial=trial,**command)
                baseline=samples[-50:]
                gravity=[statistics.median(s['accel_g'][i] for s in baseline) for i in range(3)]
                norm=math.sqrt(sum(v*v for v in gravity));up=[v/norm for v in gravity]
                bias=[statistics.median(s['gyro_dps'][i] for s in baseline) for i in range(3)]
                previous=samples[-1];angle=0
                for phase_index,phase in enumerate(command.get('phases',[command])):
                    if 'phases' in command:mark('phase_start',trial=trial,phase=phase_index,**phase)
                    end=time.monotonic()+phase['duration']
                    while time.monotonic()<end:
                        check()
                        if command.get('target_yaw_deg'):
                            for sample in samples:
                                if sample['seq']>previous['seq']:
                                    dt=sample['t']-previous['t']
                                    angle+=sum((sample['gyro_dps'][i]+previous['gyro_dps'][i]-2*bias[i])*.5*up[i] for i in range(3))*dt
                                    previous=sample
                            if abs(angle)>=command['target_yaw_deg']:break
                        await send(dict(type='calibration-step',forward=phase['forward'],strafe=phase['strafe'],
                            turn=phase['turn'],speed_limit=phase['pwm'],duration=.20 if 'phases' in command else min(.20,max(.01,end-time.monotonic()))))
                        await asyncio.sleep(min(.05,max(0,end-time.monotonic())))
                await send(dict(type='drive',forward=0,strafe=0,turn=0,speed_limit=500))
                mark('drive_end',trial=trial)
                await rest(1.1)
                mark('trial_end',trial=trial)
                print(f"{trial+1}/{len(trials)} f/s/t={command['forward']}/{command['strafe']}/{command['turn']} PWM{command['pwm']} {command['duration']}s stopped",flush=True)
            mark('final_baseline_start');await rest(5);mark('final_baseline_end')
            metadata['status']='complete'
        except BaseException as error:
            metadata.update(status='aborted',error=repr(error));raise
        finally:
            try:await send(dict(type='stop',reason='disarm'))
            except Exception:pass
            live=False
            for task in tasks:task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            for output in outputs.values():output.close()
            if visual_output:visual_output.close()
            metadata.update(sample_count=len(samples),frame_count=frames,ended_utc_untrusted_pi_clock=time.time())
            save_manifest()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8765')
    parser.add_argument('--output',required=True)
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--trajectories',action='store_true')
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--stationary-seconds',type=float,default=0)
    parser.add_argument('--rotate-degrees',type=float,default=0)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--powers',type=int,nargs='+',help='Select PWM limits from the standard plan')
    parser.add_argument('--kinds',choices=['translation','rotation','coupled'],nargs='+')
    parser.add_argument('--repeats',type=int,nargs='+',help='Select repeat indexes; repeat 2 remains held out')
    parser.add_argument('--camera-exposure-us',type=int,help='Record deployed camera exposure; does not change service')
    parser.add_argument('--camera-gain',type=float,help='Record deployed camera analogue gain; does not change service')
    parser.add_argument('--visual-radius',type=float,default=0,help='Fail closed outside tracked radius in meters; 0 disables')
    parser.add_argument('--loop-seconds',type=float,default=0,help='Continuous translation circle period; total duration remains eight seconds')
    parser.add_argument('--visual-height',type=float,default=.05992,help='Training estimated camera height in meters')
    parser.add_argument('--visual-normal',type=float,nargs=3,default=[-.03495,.99926,-.00158],help='Training floor normal in camera coordinates')
    asyncio.run(run(parser.parse_args()))
