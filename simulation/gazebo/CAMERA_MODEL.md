# Current camera stand and Arducam simulation

The user confirmed the upright stand is **40 × 40 mm**, centered laterally and
**40 mm back from the main PCB's front edge**. The optical axis passes through
the stand's centre, 20 mm above the board. With the existing 190.5 mm PCB length
and 44.45 mm board height:

- Stand centre plane: body X = 55.25 mm.
- Lens centre height: body Z = 64.45 mm.
- Optical origin: body [71.05, 0, 64.45] mm, including an estimated 15.8 mm
  forward lens projection from the stand plane.
- Camera faces body +X, with no assumed pitch or yaw offset. The vertical
  white frame, green board, screws, black lens and orange ribbon loop replace
  the old camera below and ahead of the board.

Frame thickness, lens projection, component details and servo placement behind
the fixed stand are photo-informed estimates. The camera does not move with
servo commands. Camera/mount mass remains an estimated 8 g; the component mass
centres are updated while retaining the last measured 577.4 g total pending a
new weighing. Camera plus stand now sit within the tire envelope.

## Observed hardware and matching settings

Read-only inspection of the running Pi on 2026-10-17 found:

- Driver name `arducam-pivariety`, 1600 × 1300, 10-bit monochrome.
- Advertised modes 1280 × 720, 1600 × 1080 and 1600 × 1300, each at 30 fps.
- Running `rpicam-vid`: **1280 × 720, 30 fps**, monochrome MJPEG, quality 75,
  shutter 8000 µs, gain 1.5 and `cdn_off` denoise.

Historical notes identify the board as UC-788 Rev.B and label the sensor OV9281.
That sensor label conflicts with the live 1600 × 1300 modes: Arducam's
[OV9281 datasheet](https://www.arducam.com/downloads/modules/RaspberryPi_camera/OV9281_MIPI_Camera_Module_DS_v2.pdf)
lists a 1280 × 800 sensor. The generic Pivariety name is insufficient to identify
the installed sensor and lens. We therefore match the observed stream without
claiming a verified sensor SKU or borrowing FOV from a different Arducam lens.

`camera_profile.json` is the shared camera/mount configuration. The native
Gazebo camera publishes **1280 × 720 L8 grayscale at a configured 30 Hz**, with
instantaneous full-frame projection. It uses the same **nominal** camera matrix
as the real-car vision code: fx = fy = 968 px, cx = 640 px, cy = 360 px. This
implies about 67° horizontal and 41° vertical field of view. Distortion is
provisionally zero. The real intrinsics remain **unconfirmed**; existing parking
calibration flags are not promoted or replaced by this simulation update.

The native camera uses `gz::sim::systems::Sensors` with Ogre2; the ideal/noisy IMU
pipeline remains separate. Native topics are:

```sh
gz topic -e -t /model/freenove/camera/image
gz topic -e -t /model/freenove/camera/camera_info
```

The browser console's Front camera panel shows a bandwidth-limited PNG preview
at up to 5 fps; that preview does not change the native sensor's 30 Hz setting.
`/camera.png` serves the newest frame without accumulating history. Native
image messages and the PNG preview are grayscale; no color camera is converted
or substituted. Gazebo's [CameraSensor implementation](https://github.com/gazebosim/gz-sensors/blob/main/src/CameraSensor.cc)
supports the L8 format and camera intrinsics used here.

## Verification

```sh
python3 simulation/gazebo/generate.py
gz sdf -k simulation/gazebo/freenove_world.sdf
python3 -m unittest discover -s simulation/gazebo -p 'test_*.py'
simulation/gazebo/.control-venv/bin/python simulation/gazebo/check_camera.py
node simulation/gazebo/check_browser_camera.cjs
```

The isolated camera check creates temporary black/white targets at known body
coordinates. It verifies native image size and format, frame timing, CameraInfo
K and projected target centroids. `camera_validation.json` records approximately
one-pixel projection errors. These verify the configured simulator optics; they
are **not a calibration of the real lens**. `--output PATH` saves the target
image outside the default local log directory.

The browser check verifies the live preview, grayscale PNG header/dimensions,
keyboard pivot and simultaneous IMU updates. It needs the local console and
Gazebo running, plus Playwright/Chromium (`PLAYWRIGHT_MODULE`, `CHROME_PATH`;
optional `CAMERA_SCREENSHOT` output path). It never connects to the real car.

## Remaining differences

The renderer does not yet integrate an actual 8 ms exposure, reproduce analog
gain, motion blur, lens focus/PSF, spectral/infrared response, calibrated
photometric noise or measured distortion. No unsupported camera noise constants
are invented. Matching those requires the exact module/lens identification,
checkerboard intrinsics and controlled image captures. The simulated optical
pose also remains an estimate beyond the user-confirmed stand measurements.
