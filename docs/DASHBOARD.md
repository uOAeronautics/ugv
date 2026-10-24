# DASHBOARD

```text
url := http://192.168.8.170:8765
transport := browser ↔ WebSocket(/ws) ↔ Pi-authoritative controller ↔ PCA9685
keys := W/S forward/reverse; A/D strafe-left/right; Q/E rotate-left/right; joystick=continuous forward/back/strafe; arrows=pan/tilt; Space STOP
input := keyboard-only movement; UI keycaps=status indicators, not controls
touch := directional pad + STOP
slider := speed_limit 500..1800; default=1800
video := GET /stream.mjpg; MJPEG 640x480@20; rot180; latest-frame-only
vision := optional browser-only Ultralytics YOLO11n@320 ONNX + ORT-Web/WASM; Pi inference∅
boxes := canvas overlay; confidence=.38; class-aware NMS=.45; cadence≈3.3Hz
stabilize := optional browser-only 64x48 translational registration; max shift=24x18px
arm := explicit/session-local; reconnect=>disarmed
```

## state

```text
server := owner+vector+speed+wheels+reason+revision
one-controller; observers=N
manual := direct mecanum wheel vectors; no gyro requirement or yaw-rate loop
W/S := forward/reverse; A/D := left/right strafe; Q/E := left/right rotation
mix := left=-forward+turn; right=-forward-turn; normalize max magnitude to selected speed 500..1800
watchdog := brake after 600ms without fresh drive; remain armed for same connected controller
keyup|blur|hidden|STOP := disable follow + brake without disarming; DISARM button explicitly brakes+releases
owner-disconnect|shutdown := brake+release session; reconnect requires ARM
camera := pan ch8 + tilt ch9; center=1500us; step=25us; bounds=1000..2000us
camera.pan polarity := ArrowLeft=>+25us; ArrowRight=>-25us [physical correction]
```

## fail-safe

```text
drive-refresh=150ms while held; watchdog=600ms independent hardware thread
heartbeat=200ms state-only; heartbeat does NOT renew motion; arm-required
keyup|blur|hidden|disconnect|shutdown|STOP => wheel0..7 brake=4095; STOP preserves arm, explicit DISARM releases
client raw-PWM∅; server clamps vector±1 + speed[500,1800]
camera worker ∥ motor guard; camera stall/failure cannot stall braking
LAN-only; auth∅ => trusted-private-network only
LED := unsupported PCB-v3; UI∅
heading := game compass, relative to 3s stationary calibration pose; clockwise gyro integration
heading-rest := controls neutral + quiet IMU 500ms; bias adaptation only at rest; gaps>150ms => hold heading
heading-limit := no magnetometer => no north; gyro yaw drifts; IMU cannot recover translation from wheel skid
```

## Follow centering · 2026-10-13

`FOLLOW := center with Q/E-style full selected-speed pivots > approach straight > stop near tag; pivot impulse40..180ms; settle>=120ms; fresh post-pivot image required; Speed slider applies live; STOP disables follow while preserving ARM.`

## Temporary power controls

- CAMERA ON/OFF stops/restarts Pi capture, rather than only hiding the image. OFF clears the last frame, stops video transmission and disables ArUco/follow and parking. ON resumes capture only; enable ArUco separately if needed.
- ARUCO ON/OFF independently pauses Pi tag detection and parking observation while leaving video available.
- These settings apply to all viewers, remain until changed or the service restarts, and cannot be changed through the camera switch by an observer while another controller owns the robot.
- DETECT and STABILIZE run on the browser device. Their savings primarily affect that device. IMU, manual controls and braking watchdog remain available when capture is paused.

## Small drone markers in dim light

The detector keeps the normal 4x4-50 dictionary pass, adds bounded local-contrast
and small-window passes, and retries tiny candidates at higher resolution.
Additional passes require exact dictionary bits. New low-light identities need
repeat reads with consistent position and size before entering live tracking.
The supplied dark screenshot resolves underbody marker 12; live frames also
resolve marker 13. Only visible markers can be decoded; hidden tags are not inferred.

Camera settings support `UGV_CAMERA_SHUTTER_US`, `UGV_CAMERA_GAIN`, and
`UGV_CAMERA_JPEG_QUALITY` (defaults 8000, 1.5, 95). The under-drone setup uses
16000 microseconds and gain 3 to improve dark marker detail. This longer shutter
can blur during fast movement. Detection does not establish a surveyed tag size,
accurate parking pose, or permission to start following.

Live acquisition cycles one detector alternative per fresh frame rather than
stacking every retry into a single result. Acquired small markers use enlarged
local regions for faster repeat decoding; only the previously acquired ID can
be accepted there (with at most the dictionary's one-bit correction). Initial
small-marker acquisition requires repeat reads. Stale results and lost tracks
still expire; the overlay is not kept alive indefinitely.

For the close, bent landing-gear marker (ID 13), live acquisition also tries
contrast-enhanced border rectification on up to four large rejected contours.
It still requires an exact dictionary match and repeat acquisition. Bent paper
is not planar: the overlay identifies the marker, while its planar pose/distance
estimate can be biased and is not a precise docking measurement.
