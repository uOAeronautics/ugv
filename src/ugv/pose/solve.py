"""Pose from 2D-3D correspondences, pose error, and the SIFT feature-matching baseline.

The SIFT baseline is the supervisor's suggestion: the airframe never changes, so a bank of
reference views with known pose and a triangle-id map turns every matched SIFT keypoint into a
2D-3D pair, and PnP + RANSAC gives the pose. It needs no training and no tags.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ugv.pose.mesh import Airframe
from ugv.pose.render import View


def solve_pnp(
    pts3d: np.ndarray, pts2d: np.ndarray, K: np.ndarray, reproj_px: float = 4.0
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """RANSAC EPnP refined by LM. Returns (R, t, inliers) or None with fewer than 6 points."""
    if len(pts3d) < 6:
        return None
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        pts3d.astype(np.float64),
        pts2d.astype(np.float64),
        K,
        None,
        reprojectionError=reproj_px,
        iterationsCount=500,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inl is None or len(inl) < 6:
        return None
    idx = inl[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(pts3d[idx], pts2d[idx], K, None, rvec, tvec)
    return cv2.Rodrigues(rvec)[0], tvec[:, 0], len(idx)


def pose_error(
    R: np.ndarray, t: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray
) -> tuple[float, float]:
    """(rotation error in degrees, camera-position error in metres)."""
    cos = (np.trace(R.T @ R_gt) - 1) / 2
    rot = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    pos = float(np.linalg.norm(-R.T @ t + R_gt.T @ t_gt))
    return rot, pos


@dataclass
class Reference:
    """Descriptors of one reference view and the 3D model point under each keypoint."""

    desc: np.ndarray
    pts3d: np.ndarray


def sift_reference(view: View, frame: Airframe, sift: cv2.SIFT) -> Reference:
    gray = cv2.cvtColor(view.image, cv2.COLOR_BGR2GRAY)
    kps, desc = sift.detectAndCompute(gray, (view.ids >= 0).astype(np.uint8) * 255)
    if desc is None:
        return Reference(np.zeros((0, 128), np.float32), np.zeros((0, 3)))
    uv = np.array([k.pt for k in kps])
    tri = view.ids[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    keep = tri >= 0
    return Reference(desc[keep], frame.centroids[tri[keep]])


def sift_pose(
    image: np.ndarray, bank: list[Reference], K: np.ndarray, sift: cv2.SIFT, ratio: float = 0.8
):
    """Match a query image against every reference (Lowe ratio test), pool 2D-3D pairs, run PnP."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kps, desc = sift.detectAndCompute(gray, None)
    if desc is None:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    p2, p3 = [], []
    for ref in bank:
        if len(ref.desc) < 2:
            continue
        for pair in matcher.knnMatch(desc, ref.desc, k=2):
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                p2.append(kps[pair[0].queryIdx].pt)
                p3.append(ref.pts3d[pair[0].trainIdx])
    if len(p2) < 6:
        return None
    return solve_pnp(np.array(p3), np.array(p2), K)
