"""Hardware-free continuous docking core: delayed vision + gyro + motor history.

Angles are clockwise-positive radians, x right, z forward. Motor turn retains
manual Q/E convention (positive = Q/left). Call at50Hz; commands have a finite
lease. This module never imports a bus, service, socket or hardware controller.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from .parking import Pose2, angle_error, swept_clear


@dataclass(frozen=True)
class VisionPose:
    captured_at: float
    pose: Pose2
    position_std: float = .008
    yaw_std: float = math.radians(2)
    tag_id: int | None = 14
    confidence: float = 1.
    source: str = 'aruco'


@dataclass(frozen=True)
class MotorCommand:
    forward: float = 0.
    turn: float = 0.
    lease_seconds: float = .08
    reason: str = 'holding'


@dataclass
class Estimate:
    pose: Pose2
    speed: float
    position_std: float
    vision_age: float
    last_vision_at: float


@dataclass
class _Node:
    t: float
    prior: np.ndarray
    prior_cov: np.ndarray
    state: np.ndarray
    cov: np.ndarray
    dt: float = 0.
    command: MotorCommand = field(default_factory=MotorCommand)
    gyro: float = 0.
    measurements: list[VisionPose] = field(default_factory=list)


class PoseFusion:
    """Fixed-lag EKF. Late frames correct their capture-time state, then replay.

    The filter only sees supplied observations, IMU rates and issued commands.
    It never has access to the simulated true pose/velocity or random generator.
    """
    TAGS=frozenset((9,12,13,14))
    def __init__(self, initial: Pose2, now=0., history_seconds=2., command_delay=.04):
        x=np.array([initial.x,initial.z,initial.yaw,0.],float)
        p=np.diag([.012**2,.012**2,math.radians(4)**2,.06**2])
        self.nodes=[_Node(now,x.copy(),p.copy(),x,p)]
        self.history_seconds=history_seconds;self.last_vision_at=now
        self.accepted=0;self.rejected=0;self.expired=0
        self.command_delay=command_delay;self.pending_commands=[(now,MotorCommand())]

    @staticmethod
    def _predict(x,p,dt,command,gyro):
        target=command.forward*(.9 if command.forward>=0 else .7)
        tau=.08 if command.forward else .04
        a=math.exp(-dt/tau)
        v=a*x[3]+(1-a)*target
        yaw=x[2]+gyro*dt
        out=np.array([x[0]+math.sin(yaw)*v*dt,x[1]+math.cos(yaw)*v*dt,
                      angle_error(yaw,0),v])
        f=np.eye(4);f[0,2]=math.cos(yaw)*v*dt;f[1,2]=-math.sin(yaw)*v*dt
        f[0,3]=math.sin(yaw)*a*dt;f[1,3]=math.cos(yaw)*a*dt;f[3,3]=a
        q=np.diag([.002**2*dt,.002**2*dt,math.radians(.8)**2*dt,.16**2*dt])
        return out,f@p@f.T+q

    @staticmethod
    def _correct(x,p,m,gate=True):
        h=np.eye(3,4);r=np.diag([m.position_std**2,m.position_std**2,m.yaw_std**2])
        innovation=np.array([m.pose.x-x[0],m.pose.z-x[1],angle_error(m.pose.yaw,x[2])])
        s=h@p@h.T+r
        if gate and float(innovation@np.linalg.solve(s,innovation))>16.3:return None
        k=np.linalg.solve(s,h@p).T
        state=x+k@innovation;state[2]=angle_error(state[2],0)
        a=np.eye(4)-k@h
        cov=a@p@a.T+k@r@k.T
        return state,(cov+cov.T)/2

    def advance(self, now, command, gyro_rate):
        previous=self.nodes[-1];dt=now-previous.t
        if not math.isfinite(gyro_rate) or dt<=0 or dt>.25:raise ValueError('invalid control/IMU interval')
        self.pending_commands.append((now,command))
        cutoff=now-self.command_delay
        while len(self.pending_commands)>1 and self.pending_commands[1][0]<=cutoff+1e-9:self.pending_commands.pop(0)
        applied=self.pending_commands[0][1] if self.pending_commands[0][0]<=cutoff+1e-9 else MotorCommand()
        x,p=self._predict(previous.state,previous.cov,dt,applied,gyro_rate)
        self.nodes.append(_Node(now,x.copy(),p.copy(),x,p,dt,applied,gyro_rate))
        while len(self.nodes)>2 and self.nodes[1].t<now-self.history_seconds:self.nodes.pop(0)

    def observe(self, m):
        now=self.nodes[-1].t
        values=[m.captured_at,m.pose.x,m.pose.z,m.pose.yaw,m.position_std,m.yaw_std,m.confidence]
        if (not all(math.isfinite(v) for v in values) or m.position_std<=0 or m.yaw_std<=0
            or m.confidence<.85 or m.source not in ('aruco','scene')
            or (m.source=='aruco' and m.tag_id not in self.TAGS)):
            self.rejected+=1;return False
        if m.captured_at<self.nodes[0].t-.011 or m.captured_at>now+.011:
            self.expired+=1;return False
        index=min(range(len(self.nodes)),key=lambda i:abs(self.nodes[i].t-m.captured_at))
        node=self.nodes[index]
        # Duplicate decoded frames must not manufacture extra confidence.
        if any(v.source==m.source and v.tag_id==m.tag_id and v.captured_at==m.captured_at for v in node.measurements):return False
        corrected=self._correct(node.state,node.cov,m)
        if corrected is None:self.rejected+=1;return False
        node.measurements.append(m);node.state,node.cov=corrected
        for i in range(index+1,len(self.nodes)):
            child=self.nodes[i];parent=self.nodes[i-1]
            x,p=self._predict(parent.state,parent.cov,child.dt,child.command,child.gyro)
            child.prior=x.copy();child.prior_cov=p.copy()
            for measurement in child.measurements:x,p=self._correct(x,p,measurement,gate=False)
            child.state,child.cov=x,p
        self.last_vision_at=max(self.last_vision_at,m.captured_at);self.accepted+=1
        return True

    def estimate(self):
        node=self.nodes[-1];x,p=node.state,node.cov
        return Estimate(Pose2(float(x[0]),float(x[1]),float(x[2])),float(x[3]),
                        math.sqrt(max(float(p[0,0]),float(p[1,1]))),
                        node.t-self.last_vision_at,self.last_vision_at)


class TorqueAllocator:
    """Pulse density provides small average movement above motor breakaway.

    Axes get full-strength20ms slots; feedback continues through every slot.
    Gyro rate closes the asymmetrical steering response. No rigid settle gaps.
    """
    def __init__(self):
        self.forward_debt=0.;self.turn_debt=0.;self.forward_sign=0;self.turn_sign=0

    def reset(self):
        self.forward_debt=self.turn_debt=0.;self.forward_sign=self.turn_sign=0

    def step(self,speed,yaw_rate,gyro_rate,reason):
        fsign=1 if speed>0 else -1 if speed<0 else 0
        corrected=yaw_rate+.4*(yaw_rate-gyro_rate)
        rsign=1 if corrected>0 else -1 if corrected<0 else 0
        if abs(speed)<.002:fsign=0
        if abs(yaw_rate)<math.radians(.6) and abs(gyro_rate)<math.radians(2):rsign=0
        if fsign!=self.forward_sign:self.forward_debt=0.
        if rsign!=self.turn_sign:self.turn_debt=0.
        self.forward_sign=fsign;self.turn_sign=rsign
        f=abs(speed)/(.45 if fsign>0 else .35) if fsign else 0.
        r=abs(corrected)/math.radians(95 if rsign>0 else 70) if rsign else 0.
        scale=max(1.,(f+r)/.90);f/=scale;r/=scale
        self.forward_debt+=f;self.turn_debt+=r
        if self.forward_debt>=1 and self.forward_debt>=self.turn_debt:
            self.forward_debt-=1;return MotorCommand(float(fsign),0.,reason=reason)
        if self.turn_debt>=1:
            self.turn_debt-=1;return MotorCommand(0.,float(-rsign),reason=reason)
        return MotorCommand(reason=reason)


class ContinuousDocking:
    def __init__(self, vehicle, obstacles, goal=Pose2(0,0,0)):
        self.vehicle=vehicle;self.obstacles=obstacles;self.goal=goal
        self.phase='approach';self.allocator=TorqueAllocator();self.aligned_since=None
        self.settled_since=None;self.parked=False;self.reason='approach';self.holds=0

    def hold(self,reason):
        self.allocator.reset();self.settled_since=None;self.reason=reason;self.holds+=1
        return MotorCommand(reason=reason)

    def _clear(self,pose,speed,rate,inflation):
        horizon=.16
        # Include the rectangle's full yaw sweep, rather than the centre line.
        mid=pose.yaw+rate*horizon/2
        predicted=Pose2(pose.x+math.sin(mid)*speed*horizon,pose.z+math.cos(mid)*speed*horizon,
                        pose.yaw+rate*horizon)
        return swept_clear(pose,predicted,self.vehicle['width_m'],self.vehicle['length_m'],self.obstacles,inflation)

    def step(self,estimate,gyro_rate,now):
        if self.parked:return MotorCommand(reason='parked')
        if not all(math.isfinite(v) for v in (estimate.pose.x,estimate.pose.z,estimate.pose.yaw,estimate.speed,estimate.position_std,estimate.vision_age,gyro_rate,now)):
            return self.hold('invalid feedback')
        p=estimate.pose
        if estimate.vision_age>1.2 or estimate.position_std>.06:return self.hold('waiting for vision')
        stage_z=self.goal.z-.50
        speed=rate=0.
        if self.phase=='approach':
            dx,dz=self.goal.x-p.x,stage_z-p.z;distance=math.hypot(dx,dz)
            if distance<.024:self.phase='align';self.aligned_since=None
            else:
                error=angle_error(math.atan2(dx,dz),p.yaw)
                rate=max(-math.radians(70),min(math.radians(70),2.5*error))
                speed=.16*math.tanh(distance*4)*max(0.,min(1.,(math.cos(error)-.78)/.22))
        if self.phase=='align':
            error=angle_error(self.goal.yaw,p.yaw);rate=max(-math.radians(60),min(math.radians(60),2.6*error))
            if abs(error)<math.radians(2):
                if self.aligned_since is None:self.aligned_since=now
                if now-self.aligned_since>.25:
                    if abs(p.x-self.goal.x)>.018:self.phase='approach'
                    else:self.phase='lane'
            else:self.aligned_since=None
        if self.phase=='lane':
            target=self.goal.yaw+max(-math.radians(12),min(math.radians(12),math.atan2(self.goal.x-p.x,.08)))
            rate=max(-math.radians(30),min(math.radians(30),3.5*angle_error(target,p.yaw)))
            speed=.05*max(0.,math.cos(angle_error(target,p.yaw)))
            if p.z>=self.goal.z-.38:self.phase='straighten';self.aligned_since=None
        if self.phase=='straighten':
            speed=0.;error=angle_error(self.goal.yaw,p.yaw)
            rate=max(-math.radians(25),min(math.radians(25),3.*error))
            if abs(error)<math.radians(2):
                if self.aligned_since is None:self.aligned_since=now
                if now-self.aligned_since>.25:
                    if abs(p.x-self.goal.x)>.012:self.phase='recover'
                    else:self.phase='enter'
            else:self.aligned_since=None
        if self.phase=='recover':
            if p.z<stage_z+.02:self.phase='approach'
            else:speed=-.04;rate=0.
        if self.phase=='enter':
            dx,dz=self.goal.x-p.x,self.goal.z-p.z
            heading_limit=math.radians(3)
            target=self.goal.yaw+max(-heading_limit,min(heading_limit,math.atan2(dx,.17)))
            error=angle_error(target,p.yaw)
            rate=max(-math.radians(25),min(math.radians(25),3.*error))
            speed=max(-.045,min(.06,.65*dz))
            if abs(error)>math.radians(6):speed*=.2
            settled=(math.hypot(dx,dz)<.014 and abs(angle_error(self.goal.yaw,p.yaw))<math.radians(3)
                     and abs(estimate.speed)<.018 and estimate.vision_age<.35)
            if settled:
                speed=rate=0.
                if self.settled_since is None:self.settled_since=now
                if now-self.settled_since>1. and estimate.last_vision_at>self.settled_since+.7:
                    self.parked=True;self.allocator.reset();return MotorCommand(reason='parked')
            else:self.settled_since=None
        if estimate.vision_age>.50:speed*=.4
        inflation=.004+min(.006,.35*estimate.position_std)
        if not self._clear(p,speed,rate,inflation):
            # Auto recovery backs out along the same opening before realigning.
            straighten=max(-math.radians(10),min(math.radians(10),2.5*angle_error(self.goal.yaw,p.yaw)))
            if self.phase=='enter' and abs(angle_error(self.goal.yaw,p.yaw))>math.radians(1) and self._clear(p,0.,straighten,.001):
                speed=0.;rate=straighten
            elif self.phase=='enter' and estimate.vision_age<.35 and self._clear(p,-.04,0.,.003):
                self.phase='recover';speed=-.04;rate=0.
            elif self.phase=='recover' and self._clear(p,-.04,0.,.003):speed=-.04;rate=0.
            else:return self.hold('footprint blocked')
        self.reason=self.phase
        return self.allocator.step(speed,rate,gyro_rate,self.phase)
