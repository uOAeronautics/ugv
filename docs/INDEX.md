# INDEX

`purpose:=minimal-token context; audience:=LLM; prose:=min; truth:=typed`

## load

```text
L0 README.md
L1 STATE.md + SAFETY.md
L2 HARDWARE.md + NETWORK.md + COMMANDS.md + DASHBOARD.md
L3 PROTOCOL.md
```

## epistemics

```text
✓ observed/tested
~ inferred
? unknown/unverified
× absent/disabled/fail
! hazard/blocker
∆ changed
@ locator
# reference
```

`conflict := newest-observation > inference > vendor-doc > assumption`

`mutation := declare intent -> safety gate -> smallest effect -> verify -> STATE∆`

`never-store := passwords|tokens|private-keys|router-secrets`

`ARUCO.md := 50mm IDs0..4 + pose/follow equations + autonomous failsafes`

`PARKING.md := provisional tag/IMU/scene map + swept-footprint planner + calibrated entry gates`

`RL_PARKING.md := goal + parked definition + L0-L7 ladder + obs/act contracts + train-on-3090 / deploy-on-Pi4 pipeline`

`DYNAMICS.md := raw 100Hz IMU/PWM capture + drift-bracketed real-car identification + Gazebo effort fit`
