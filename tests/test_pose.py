"""Synthetic pose pipeline: mesh, skeleton, renderer and PnP agree with each other."""

from __future__ import annotations

import numpy as np

from ugv.pose.mesh import PARTS, keypoints, load_airframe
from ugv.pose.render import (
    CAR_VIEWS,
    Camera,
    look_at,
    part_albedo,
    procedural_texture,
    random_part_albedo,
    render,
)
from ugv.pose.solve import pose_error, solve_pnp


def test_airframe_loads_in_model_frame() -> None:
    frame = load_airframe()
    assert set(frame.part.tolist()) == set(range(len(PARTS)))
    span = np.ptp(frame.tris.reshape(-1, 3)[:, 1])
    assert 1.1 < span < 1.3, "wingspan is about 1.2 m; units or frame changed"


def test_skeleton_has_twelve_distinct_points() -> None:
    kp = keypoints(load_airframe())
    assert len(kp) == 12
    pts = np.array(list(kp.values()))
    assert np.min(np.linalg.norm(pts[:, None] - pts[None], axis=-1) + np.eye(len(pts))) > 0.02
    assert kp["wingtip_left"][1] > 0 > kp["wingtip_right"][1], "y must point to the left wing"


def test_render_and_pnp_round_trip() -> None:
    frame = load_airframe()
    kp = keypoints(frame)
    cam = Camera()
    R, t = look_at(np.array([-1.5, 0.6, 0.4]), np.zeros(3), 0.0)
    view = render(
        frame, kp, R, t, cam, np.random.default_rng(0), procedural_texture(frame.centroids)
    )
    assert view.image.shape == (cam.height, cam.width, 3)
    assert (view.ids >= 0).mean() > 0.02, "airframe should cover part of the image"
    names = [n for n in kp if view.kp_visible[n]]
    sol = solve_pnp(
        np.array([kp[n] for n in names]), np.array([view.kp2d[n] for n in names]), cam.K
    )
    assert sol is not None
    rot, pos = pose_error(sol[0], sol[1], R, t)
    assert rot < 0.1 and pos < 0.005


def test_car_views_look_up_at_the_belly() -> None:
    assert CAR_VIEWS.eye_z_m is not None and max(CAR_VIEWS.eye_z_m) < 0


def test_part_albedo_dark_rotors_and_light_body() -> None:
    frame = load_airframe()
    sdf = part_albedo(frame)
    body_mask = frame.part == PARTS.index("body")
    prop_mask = frame.part == PARTS.index("prop_left")
    assert sdf[body_mask].mean() > 0.8 and sdf[prop_mask].mean() < 0.2
    rng = np.random.default_rng(7)
    for _ in range(10):
        rand = random_part_albedo(frame, rng)
        assert rand.shape == sdf.shape
        assert 0.0 <= rand.min() and rand.max() <= 1.0
        assert rand[body_mask].mean() > rand[prop_mask].mean() + 0.15
