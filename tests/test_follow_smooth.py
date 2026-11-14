"""Holonomic smooth-follow policy: lead, damping, simultaneity, CoastSearch."""


from ugv.aruco import TARGET_DISTANCE_M
from ugv.follow import AlphaBetaLead, HolonomicFollowPolicy, smooth_deadzone


def test_lead_predicts_ahead_with_velocity():
    lead = AlphaBetaLead(predict_s=0.2)
    lead.update(0.10, 0.60, 0.0)
    lead.update(0.14, 0.58, 0.1)
    st = lead.state
    assert st.vx != 0.0  # motion observed
    x_lead, z_lead, sigma = lead.lead(0.1)
    assert x_lead == st.x + st.vx * 0.2
    assert z_lead == st.z + st.vz * 0.2
    assert sigma >= lead.base_sigma_m


def test_sigma_grows_with_age():
    lead = AlphaBetaLead(predict_s=0.15)
    lead.update(0.0, 0.5, 0.0)
    _, _, fresh = lead.lead(0.0)
    _, _, stale = lead.lead(1.0)
    assert stale > fresh


def test_yaw_rate_damping_reduces_command():
    policy = HolonomicFollowPolicy()
    undamped = policy.step(0.3, 0.6, 0.0, yaw_rate_dps=0.0)
    policy.reset()
    # A yaw rate already carrying the corrective turn (negative for +error)
    # backs the command off via -kd * yaw_rate_dps.
    damped = policy.step(0.3, 0.6, 0.0, yaw_rate_dps=-60.0)
    assert abs(damped.turn) < abs(undamped.turn)


def test_damped_plant_overshoots_less_than_undamped():
    # Laggy second-order yaw axis: the undamped proportional loop coasts
    # past zero (negative min error); gyro damping removes the overshoot.
    def run(kd):
        policy = HolonomicFollowPolicy(kd_yaw=kd)
        error, velocity = 0.5, 0.0
        min_error = 1.0
        for tick in range(600):
            now = tick * 0.02
            # Gyro measures body rate in the command frame (same sign as
            # turn velocity), so -kd * yaw_rate_dps is negative feedback.
            cmd = policy.step(error, TARGET_DISTANCE_M + 0.3, now, yaw_rate_dps=velocity * 30.0)
            velocity += (cmd.turn * 3.0 - velocity) * 0.02 / 0.35
            error += velocity * 0.02
            min_error = min(min_error, error)
        return min_error

    undamped_crossing = run(0.0)
    damped_crossing = run(0.006)
    assert undamped_crossing < 0.0  # control case genuinely overshoots
    assert damped_crossing > undamped_crossing  # damping removes it


def test_simultaneous_forward_strafe_turn():
    policy = HolonomicFollowPolicy()
    cmd = policy.step(0.3, 0.6, 0.0)
    assert cmd.forward > 0
    assert cmd.strafe != 0.0
    assert cmd.turn != 0.0
    assert cmd.reason == "smooth-track"


def test_smooth_deadzone_is_continuous():
    assert smooth_deadzone(0.02) == 0.0
    assert smooth_deadzone(0.04) == 0.0
    just_out = smooth_deadzone(0.05)
    assert 0.0 < just_out < 0.05
    assert smooth_deadzone(-0.5) < 0.0


def test_coast_search_stop_transitions():
    policy = HolonomicFollowPolicy()
    cmd = policy.step(0.3, 0.6, 0.0)
    assert policy.mode == "TRACK"
    assert cmd.reason == "smooth-track"
    coast = policy.step(None, None, 0.2, visible=False)
    assert policy.mode == "COAST"
    assert coast.reason == "smooth-coast"
    assert abs(coast.turn) <= abs(cmd.turn)  # exponential decay along lead
    search = policy.step(None, None, 0.9, visible=False)
    assert policy.mode == "SEARCH"
    assert search.reason == "smooth-search"
    assert abs(search.turn) <= policy.search_scan_amp + 1e-9
    assert search.forward == 0.0 and search.strafe == 0.0
    stop = policy.step(None, None, 2.0, visible=False)
    assert policy.mode == "STOP"
    assert stop.reason == "smooth-stop"
    assert (stop.forward, stop.strafe, stop.turn) == (0.0, 0.0, 0.0)
    # reacquire returns to TRACK
    reacquire = policy.step(0.1, 0.6, 2.1)
    assert policy.mode == "TRACK"
    assert reacquire.reason == "smooth-track"


def test_target_reached_brakes():
    policy = HolonomicFollowPolicy()
    cmd = policy.step(0.01, TARGET_DISTANCE_M, 0.0)
    assert cmd.reason == "smooth-target-reached"
    assert (cmd.forward, cmd.strafe, cmd.turn) == (0.0, 0.0, 0.0)
