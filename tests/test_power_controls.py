import asyncio
from unittest.mock import AsyncMock, patch
import pytest
from ugv import web as m


def test_pause_clears_stale_camera_and_resume_requires_new_frame():
    camera = m.CameraStream()
    camera.frame = b'old'; camera.received_at = m.time.monotonic()
    camera.frame_times.append(camera.received_at)
    camera.set_enabled(False)
    assert camera.latest_sample() == (None, 0, 0.0)
    assert camera.metrics()['camera_enabled'] is False
    assert camera.metrics()['camera_fps'] == 0
    camera.set_enabled(True)
    assert camera.latest_sample() == (None, 0, 0.0)


def test_camera_pause_cancels_autonomy_before_capture_and_rejects_observer():
    async def check():
        controller = m.RobotController()
        controller.cancel_parking = AsyncMock()
        controller.broadcast = AsyncMock()
        with patch.object(m, 'camera') as camera, patch.object(m, 'aruco') as aruco:
            aruco.configure = AsyncMock()
            controller.state.owner = 'driver'
            with pytest.raises(PermissionError):
                await controller.camera_capture('observer', False)
            camera.set_enabled.assert_not_called()
            await controller.camera_capture('driver', False)
            controller.cancel_parking.assert_awaited_once_with('camera-paused')
            aruco.configure.assert_awaited_once_with('driver', False, False)
            camera.set_enabled.assert_called_once_with(False)
            aruco.configure.reset_mock()
            await controller.camera_capture('driver', True)
            aruco.configure.assert_not_awaited()
    asyncio.run(check())
