import asyncio
import math
from unittest.mock import patch
import pytest
from ugv.web import RobotController


def controller():
    c=RobotController();c.state.owner='test';c.state.armed=True
    c._write=lambda wheels:None
    return c


def test_full_pwm_is_bounded_and_deadline_brakes():
    c=controller()
    with patch.object(c,'_imu_ready',return_value=True):
        asyncio.run(c.calibration_step('test',0,1,0,4095,.1))
        assert max(abs(v) for v in c.state.wheels.values())==4095
        assert c.state.strafe==1
        c._guard_check(c.drive_deadline+.001)
    assert c.state.stopped and not any(c.state.wheels.values())


@pytest.mark.parametrize('pwm,duration',[(4096,.1),(1000,.251),(-1,.1),(math.nan,.1)])
def test_invalid_calibration_pulse_never_moves(pwm,duration):
    c=controller()
    with patch.object(c,'_imu_ready',return_value=True),pytest.raises(ValueError):
        asyncio.run(c.calibration_step('test',1,0,0,pwm,duration))
    assert c.state.stopped


def test_sensor_loss_and_foreign_owner_cannot_energize():
    c=controller()
    with patch.object(c,'_imu_ready',return_value=False),pytest.raises(PermissionError):
        asyncio.run(c.calibration_step('test',1,0,0,1800,.1))
    with pytest.raises(PermissionError):
        asyncio.run(c.calibration_step('other',1,0,0,1800,.1))
    assert c.state.stopped
