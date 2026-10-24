"""Bounded keyboard commands and an expiring client heartbeat, independent of Gazebo."""
import math
import threading
import time

KEYS = {'KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE'}

class ControlState:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.RLock()
        self.deadline = 0
        self.linear = self.lateral = self.angular = 0.0
        self.pwm = None
        self.left_duty = self.right_duty = 0.0
        self.wheel_duties = (0.,0.,0.,0.)
        self.owner = None
        self.sequences = {}

    def drive(self, client, sequence, keys, speed, pwm=None):
        if not isinstance(client, str) or not 1 <= len(client) <= 80:
            raise ValueError('Invalid controller ID')
        if type(sequence) is not int or sequence < 0:
            raise ValueError('Invalid sequence')
        if not isinstance(keys, list) or len(keys) > 6 or any(k not in KEYS for k in keys):
            raise ValueError('Invalid driving keys')
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed):
            raise ValueError('Invalid speed')
        if pwm is not None and (type(pwm) is not int or not 0 <= pwm <= 4095):
            raise ValueError("PWM must be an integer from 0 to 4095")
        speed = max(.03, min(.3, speed))
        with self.lock:
            if sequence <= self.sequences.get(client, -1):
                return False
            if len(self.sequences) >= 32 and client not in self.sequences:
                raise ValueError('Too many controller sessions; restart the panel server')
            # An idle tab may send a release when the simulation pauses or
            # reconnects. It must not claim ownership and block a driving tab.
            if not keys:
                self.sequences[client] = sequence
                if self.owner in (None, client):
                    self.linear = self.lateral = self.angular = 0.0
                    self.left_duty = self.right_duty = 0.0
                    self.wheel_duties=(0.,0.,0.,0.)
                    self.deadline = 0
                    self.owner = None
                return True
            if self.owner not in (None, client) and self.clock() < self.deadline:
                raise ValueError('Another panel is controlling the car')
            self.sequences[client] = sequence
            self.owner = client
            forward = int('KeyW' in keys) - int('KeyS' in keys)
            pivot = int('KeyQ' in keys) - int('KeyE' in keys)
            steering = int('KeyA' in keys) - int('KeyD' in keys)
            self.pwm = pwm
            # Body +Y is left; Q is counterclockwise rotation.
            raw=(forward-steering-pivot,forward+steering+pivot,
                 forward+steering-pivot,forward-steering+pivot)
            scale=max(1,*(abs(value) for value in raw))
            self.wheel_duties=tuple(value/scale*(pwm or 0)/4095 for value in raw)
            self.left_duty,self.right_duty=self.wheel_duties[:2]
            self.linear = forward * speed
            self.lateral = steering * speed
            self.angular = pivot * min(1, speed * 5)
            self.deadline = self.clock() + .4
            return True

    def command(self, enabled=True):
        with self.lock:
            if not enabled or self.clock() >= self.deadline:
                return 0.0, 0.0
            return self.linear, self.angular

    def duty_command(self, enabled=True):
        with self.lock:
            if not enabled or self.clock() >= self.deadline:
                return (0.,0.,0.,0.)
            return self.wheel_duties

    def lateral_command(self, enabled=True):
        with self.lock:
            return self.lateral if enabled and self.clock()<self.deadline else 0.

    def stop(self):
        with self.lock:
            self.linear = self.lateral = self.angular = 0.0
            self.left_duty = self.right_duty = 0.0
            self.wheel_duties=(0.,0.,0.,0.)
            self.deadline = 0
