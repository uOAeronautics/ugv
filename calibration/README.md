# Calibration recordings

`runs/` contains timestamped IMU samples, commanded wheel PWM, motion events,
camera frame manifests, session settings, and fitted/evaluation results. Aborted
sessions are retained and marked in their manifests; they are not automatically
valid complete trials. Pi monotonic timestamps align each session's sensor data.

`dataset_catalog.json` records SHA-256 hashes of the versioned logs and results.
The October 23 raw sessions were copied from `/home/advanced/ugv/calibration/runs/`
on the Pi into this laptop's `calibration/runs/` directory.

Camera JPEGs are retained on both machines alongside their frame manifests.
They are excluded from Git to avoid adding roughly 783 MB of images to ordinary
repository history. Consequently a GitHub clone contains the collection logs,
but image analysis also needs the corresponding raw image backup. Never delete
those backups merely because the logs have been pushed.

The frozen Bayesian profile, physical Gazebo profile, and validation reports are
under `simulation/gazebo/`. See `docs/MECANUM_REAL2SIM.md` for uncertainty,
coordinate conventions, training/heldout splits, and remaining validation gaps.
