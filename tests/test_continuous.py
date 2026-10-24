import math
import numpy as np
import pytest
from ugv.continuous import (ContinuousDocking,Estimate,MotorCommand,PoseFusion,TorqueAllocator,VisionPose)
from ugv.parking import Pose2,gear_obstacles
from ugv.simulation import GEAR,VEHICLE,run_episode


def test_late_frame_replays_motion_and_later_accepted_frames():
    timely=PoseFusion(Pose2(0,-1),command_delay=0)
    delayed=PoseFusion(Pose2(0,-1),command_delay=0)
    first=second=None
    for tick in range(1,31):
        now=tick*.02
        command=MotorCommand(1 if tick%3==0 else 0,0)
        for fusion in (timely,delayed):fusion.advance(now,command,.08)
        if tick==10:
            p=timely.estimate().pose
            first=VisionPose(now,Pose2(p.x+.005,p.z+.012,p.yaw+.015))
            assert timely.observe(first)
        if tick==17:
            p=timely.estimate().pose
            second=VisionPose(now,Pose2(p.x-.004,p.z-.005,p.yaw-.01))
            assert timely.observe(second) and delayed.observe(second)
        if tick==25:assert delayed.observe(first)
    assert np.allclose(timely.nodes[-1].state,delayed.nodes[-1].state,atol=1e-10)
    assert np.allclose(timely.nodes[-1].cov,delayed.nodes[-1].cov,atol=1e-10)


def test_yaw_wrap_wrong_tag_bad_pose_duplicate_and_expiry():
    f=PoseFusion(Pose2(0,-.5,math.radians(179)))
    f.advance(.02,MotorCommand(),0)
    m=VisionPose(.02,Pose2(0,-.5,math.radians(-179)))
    assert f.observe(m)
    assert not f.observe(m) and f.accepted==1
    assert not f.observe(VisionPose(.02,Pose2(.3,-.3),tag_id=17))
    assert not f.observe(VisionPose(.02,Pose2(1,1,1),tag_id=12))
    assert not f.observe(VisionPose(10,Pose2(0,-.5)))
    assert not f.observe(VisionPose(.02,Pose2(float('nan'),-.5)))
    assert f.rejected==3 and f.expired==1


def test_command_deadtime_does_not_predict_instant_movement():
    f=PoseFusion(Pose2(0,0),command_delay=.04)
    f.advance(.02,MotorCommand(1),0)
    f.advance(.04,MotorCommand(1),0)
    assert f.estimate().pose.z==0
    f.advance(.06,MotorCommand(1),0)
    assert f.estimate().pose.z>0


def test_short_detection_gap_preserves_control_long_gap_holds_then_resumes():
    c=ContinuousDocking(VEHICLE,gear_obstacles(GEAR))
    estimate=Estimate(Pose2(0,-.9),0,.01,.30,0.)
    commands=[c.step(estimate,0,i*.02) for i in range(30)]
    assert any(m.forward for m in commands) and all(m.reason!='waiting for vision' for m in commands)
    late=Estimate(estimate.pose,0,.03,1.3,0.)
    assert c.step(late,0,1.3).reason=='waiting for vision'
    fresh=Estimate(estimate.pose,0,.01,.05,1.4)
    assert any(c.step(fresh,0,1.5+i*.02).forward for i in range(30))
    assert not c.parked


def test_allocator_crosses_breakaway_and_emits_only_one_axis_per_slot():
    allocator=TorqueAllocator()
    commands=[allocator.step(.06,math.radians(10),0,'enter') for _ in range(100)]
    assert any(m.forward==1 for m in commands) and any(m.turn==-1 for m in commands)
    assert all(abs(m.forward)+abs(m.turn)<=1 and m.lease_seconds<=.1 for m in commands)
    allocator.reset()
    assert all(allocator.step(0,0,0,'stop').forward==0 for _ in range(20))


@pytest.mark.parametrize('scenario',['nominal','noisy','blackout'])
def test_continuous_docking_with_asymmetry_delay_and_transient_poses(scenario):
    result=run_episode(3,scenario)
    assert result['outcome']=='parked',result
    assert result['position_error_m']<=.03 and result['yaw_error_deg']<=5
    assert result['accepted_poses']>20
    if scenario!='nominal':assert result['dropped_frames']>10 and result['rejected_poses']>0


def test_blackout_produces_a_hold_and_returns_to_closed_loop():
    result=run_episode(0,'blackout',trace=True)
    during=[f for f in result['frames'] if 10.5<f['t']<12]
    assert during and all(f['f']==f['r']==0 for f in during)
    assert result['outcome']=='parked' and result['hold_time_s']>1


def test_true_collision_evaluator_is_independent_of_controller_claim():
    result=run_episode(0,'nominal',initial=Pose2(.157,0,0))
    assert result['outcome']=='collision'
