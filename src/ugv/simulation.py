"""Deterministic, hardware-free stochastic parking bench.

Plant truth is used only by sensors, the renderer and the outcome evaluator.
The controller receives delayed/noisy measurements through its public inputs.
"""
from __future__ import annotations
import argparse, heapq, json, math
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from .continuous import ContinuousDocking, MotorCommand, PoseFusion, VisionPose
from .parking import Pose2, angle_error, gear_obstacles, swept_clear


@dataclass(frozen=True)
class Scenario:
    name: str
    drop_probability: float
    position_noise: float
    outlier_probability: float
    latency: float
    delay_jitter: float
    steering_noise: float
    sticky_probability: float = 0.
    blackout: tuple[float,float] | None = None


SCENARIOS={
    'nominal':Scenario('Nominal',.05,.005,.005,.08,.025,.08),
    'noisy':Scenario('Uneven steering / transient tags',.25,.009,.06,.12,.05,.30,.025),
    'rough':Scenario('High noise / late frames',.45,.014,.10,.20,.09,.45,.06),
    'blackout':Scenario('Three-second camera blackout',.25,.009,.06,.12,.05,.30,.025,(9.,12.)),
}
VEHICLE={'width_m':.19,'length_m':.26}
GEAR={'clear_width_m':.254,'bar_length_m':.3048,'bar_radius_m':.03,'inward_slope':0.}


class NoisyPlant:
    def __init__(self,seed,initial,scenario):
        self.rng=np.random.default_rng(seed+300_000);self.pose=initial;self.v=self.w=0.
        self.scenario=scenario;self.speed_gain=self.rng.uniform(.82,1.18)
        self.left_gain=self.rng.uniform(.75,1.3);self.right_gain=self.rng.uniform(.75,1.3)
        self.accel_tau=self.rng.uniform(.055,.11);self.brake_tau=self.rng.uniform(.025,.06)
        self.turn_tau=self.rng.uniform(.04,.075);self.bias=math.radians(self.rng.uniform(-3,3))
        self.grip=0.;self.sticky_until=0.;self.queue=[MotorCommand(),MotorCommand()]
        self.gyro_bias=math.radians(self.rng.uniform(-.35,.35))

    def step(self,command,now,dt):
        self.queue.append(command);applied=self.queue.pop(0)
        self.grip+=(-self.grip*dt/.35+self.rng.normal(0,self.scenario.steering_noise)*math.sqrt(dt/.35))
        if applied.turn and self.rng.random()<self.scenario.sticky_probability:self.sticky_until=now+self.rng.uniform(.04,.14)
        # Small proportional motor commands do not overcome tyre friction.
        turn_power=math.copysign(max(0.,abs(applied.turn)-.78)/.22,applied.turn)
        full_rate=math.radians(155*self.right_gain if applied.turn<0 else 110*self.left_gain)
        target_w=-turn_power*full_rate*max(.35,min(1.65,1+self.grip))
        if now<self.sticky_until:target_w*=.15
        forward_power=math.copysign(max(0.,abs(applied.forward)-.65)/.35,applied.forward)
        target_v=forward_power*(.9 if applied.forward>=0 else .7)*self.speed_gain
        old_w=self.w
        self.v+=(target_v-self.v)*(1-math.exp(-dt/(self.accel_tau if applied.forward else self.brake_tau)))
        self.w+=(target_w+self.bias*abs(self.v)/.25-self.w)*(1-math.exp(-dt/(self.turn_tau if applied.turn else self.brake_tau)))
        yaw=angle_error(self.pose.yaw+(old_w+self.w)/2*dt,0)
        self.pose=Pose2(self.pose.x+math.sin(yaw)*self.v*dt,self.pose.z+math.cos(yaw)*self.v*dt,yaw)
        gyro=(old_w+self.w)/2+self.gyro_bias+self.rng.normal(0,math.radians(.6))
        return gyro


class NoisyVision:
    def __init__(self,seed,scenario):
        self.rng=np.random.default_rng(seed+600_000);self.scenario=scenario
        self.next_capture=.08;self.queue=[];self.serial=0;self.sent=0;self.dropped=0
        self.last_capture_outcome='waiting';self.scene_since=None

    def capture(self,now,pose):
        if now<self.next_capture:return
        self.next_capture=now+self.rng.uniform(.065,.105);s=self.scenario
        blackout=s.blackout is not None and s.blackout[0]<=now<s.blackout[1]
        burst=(s.name!='Nominal' and (7.<=now<7.55 or 18.<=now<18.6))
        if blackout or burst or self.rng.random()<s.drop_probability:
            self.dropped+=1;self.last_capture_outcome='blackout' if blackout else 'missed';return
        source='scene' if pose.z>-.10 else 'aruco'
        if source=='scene' and self.scene_since is None:self.scene_since=now
        drift=min(.010,(now-(self.scene_since or now))*.001) if source=='scene' else 0.
        sigma=s.position_noise*(1.25 if source=='scene' else 1.)
        x,z=self.rng.normal([pose.x+drift,pose.z],sigma)
        yaw=pose.yaw+self.rng.normal(0,math.radians(2.))
        tag=(9,12,13,14)[self.sent%4] if source=='aruco' else None
        outcome='scene' if source=='scene' else 'tag'
        if self.rng.random()<s.outlier_probability:
            x+=self.rng.choice((-1,1))*self.rng.uniform(.18,.55)
            z+=self.rng.uniform(-.25,.25);yaw+=math.radians(self.rng.uniform(30,100))
            if source=='aruco' and self.rng.random()<.5:tag=17
            outcome='bad pose'
        m=VisionPose(now,Pose2(float(x),float(z),angle_error(yaw,0)),sigma,math.radians(2.),tag,1.,source)
        self.serial+=1;self.sent+=1;self.last_capture_outcome=outcome
        delay=max(.02,self.rng.normal(s.latency,s.delay_jitter))
        heapq.heappush(self.queue,(now+delay,self.serial,m))

    def deliver(self,now):
        while self.queue and self.queue[0][0]<=now:yield heapq.heappop(self.queue)[2]


def run_episode(seed=0,scenario='noisy',max_seconds=65.,trace=False,initial=None):
    profile=SCENARIOS[scenario];rng=np.random.default_rng(seed)
    initial=initial or Pose2(float(rng.uniform(-.38,.38)),float(rng.uniform(-1.05,-.70)),math.radians(float(rng.uniform(-65,65))))
    plant=NoisyPlant(seed,initial,profile);vision=NoisyVision(seed,profile)
    first=Pose2(initial.x+rng.normal(0,profile.position_noise),initial.z+rng.normal(0,profile.position_noise),initial.yaw+rng.normal(0,math.radians(2)))
    fusion=PoseFusion(first);obstacles=gear_obstacles(GEAR)
    control=ContinuousDocking(VEHICLE,obstacles);command=MotorCommand();frames=[]
    outcome='timeout';dt=.02;max_est_error=0.;hold_time=0.;moving_time=0.;settling_until=None
    for tick in range(1,int(max_seconds/dt)+1):
        now=tick*dt;before=plant.pose;gyro=plant.step(command,now,dt)
        # Swept ground-truth collision evaluation, independent of estimated pose.
        if max(before.z,plant.pose.z)>-.40 and not swept_clear(before,plant.pose,VEHICLE['width_m'],VEHICLE['length_m'],obstacles,0.):
            outcome='collision';break
        fusion.advance(now,command,gyro);vision.capture(now,plant.pose)
        for measurement in vision.deliver(now):fusion.observe(measurement)
        estimate=fusion.estimate();command=control.step(estimate,gyro,now)
        max_est_error=max(max_est_error,math.hypot(estimate.pose.x-plant.pose.x,estimate.pose.z-plant.pose.z))
        if command.reason in ('waiting for vision','footprint blocked'):hold_time+=dt
        if abs(plant.v)>.01 or abs(plant.w)>math.radians(2):moving_time+=dt
        if trace and tick%4==0:
            frames.append({'t':round(now,2),'truth':[round(plant.pose.x,4),round(plant.pose.z,4),round(plant.pose.yaw,4)],
                           'estimate':[round(estimate.pose.x,4),round(estimate.pose.z,4),round(estimate.pose.yaw,4)],
                           'sigma':round(estimate.position_std,4),'vision_age':round(estimate.vision_age,3),
                           'vision':vision.last_capture_outcome,'v':round(plant.v,3),'w':round(math.degrees(plant.w),2),
                           'f':command.forward,'r':command.turn,'phase':command.reason,'accepted':fusion.accepted,'rejected':fusion.rejected})
        if control.parked:
            if settling_until is None:settling_until=now+.5
            if now>=settling_until:
                error=math.hypot(plant.pose.x,plant.pose.z)
                outcome='parked' if error<=.03 and abs(plant.pose.yaw)<=math.radians(5) else 'false parked'
                break
    result={'seed':seed,'scenario':scenario,'outcome':outcome,'time_s':round(now,2),
            'position_error_m':round(math.hypot(plant.pose.x,plant.pose.z),4),'yaw_error_deg':round(abs(math.degrees(plant.pose.yaw)),2),
            'accepted_poses':fusion.accepted,'rejected_poses':fusion.rejected,'dropped_frames':vision.dropped,
            'hold_time_s':round(hold_time,2),'max_estimate_error_m':round(max_est_error,4),'moving_time_s':round(moving_time,2)}
    if trace:result['frames']=frames
    return result


def benchmark(seeds,scenarios=tuple(SCENARIOS)):
    results=[run_episode(seed,s) for s in scenarios for seed in seeds]
    summaries={}
    for s in scenarios:
        cases=[r for r in results if r['scenario']==s]
        summaries[s]={'runs':len(cases),'parked':sum(r['outcome']=='parked' for r in cases),
                      'collisions':sum(r['outcome']=='collision' for r in cases),
                      'false_parked':sum(r['outcome']=='false parked' for r in cases),
                      'timeouts':sum(r['outcome']=='timeout' for r in cases),
                      'median_seconds':round(float(np.median([r['time_s'] for r in cases])),2),
                      'p95_position_error_m':round(float(np.percentile([r['position_error_m'] for r in cases],95)),4)}
    return {'summaries':summaries,'runs':results}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=0);parser.add_argument('--scenario',choices=SCENARIOS,default='noisy')
    parser.add_argument('--runs',type=int,default=1);parser.add_argument('--all',action='store_true');parser.add_argument('--trace',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=benchmark(range(args.seed,args.seed+args.runs),tuple(SCENARIOS) if args.all else (args.scenario,)) if args.runs>1 or args.all else run_episode(args.seed,args.scenario,trace=args.trace)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result.get('summaries',{k:v for k,v in result.items() if k!='frames'}),indent=2))

if __name__=='__main__':main()
