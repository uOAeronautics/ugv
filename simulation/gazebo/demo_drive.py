"""Drive the Gazebo-only car in a slow circle, then send a stop command."""

import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from drive_guard import require_drivable_car  # noqa: E402


def _live_topics(gz: str) -> list[str]:
    result = subprocess.run([gz, "topic", "-l"], capture_output=True, text=True, timeout=5)
    return result.stdout.splitlines() if result.returncode == 0 else []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--world", type=Path, default=None)
    parser.add_argument("--car", choices=("freenove", "payload"), default=None)
    args = parser.parse_args(argv)
    if not 0 < args.seconds <= 300:
        parser.error("--seconds must be between 0 and 300")
    sample_record = require_drivable_car(world=args.world, car=args.car)
    gz = shutil.which("gz")
    if not gz:
        parser.error("gz is not installed or on PATH")
    require_drivable_car(world=args.world, car=args.car, topics=_live_topics(gz))
    model_name = "payload_car" if sample_record is not None else "freenove"

    def publish(speed, turn):
        subprocess.run(
            [
                gz,
                "topic",
                "-t",
                f"/model/{model_name}/cmd_vel",
                "-m",
                "gz.msgs.Twist",
                "-p",
                f"linear: {{x: {speed}}}, angular: {{z: {turn}}}",
            ],
            check=True,
            timeout=5,
        )

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    print(f"Simulation demo: slow circle for {args.seconds:g} seconds", flush=True)
    try:
        publish(0.06, 0.45)
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            time.sleep(min(0.2, max(0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        pass
    finally:
        publish(0, 0)
        print("Sent simulation STOP", flush=True)


if __name__ == "__main__":
    main()
