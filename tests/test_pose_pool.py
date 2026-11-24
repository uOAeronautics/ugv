import time

import numpy as np

from ugv.pose.dataset import RollingPool


def test_rolling_pool_fills_and_refreshes() -> None:
    pool = RollingPool(64, crop=False, bay=0.2, workers=2, seed0=7_000_000_000, shape=(180, 320))
    try:
        assert pool.images.any(axis=(1, 2)).all(), "every slot filled before training starts"
        assert pool.kp.shape == (64, 15, 2) and pool.vis.shape == (64, 15)
        before = pool.images.copy()
        pool.start()
        t0 = time.time()
        while pool.replaced < 64 and time.time() - t0 < 60:
            time.sleep(0.2)
        assert pool.replaced >= 64
        assert (pool.images != before).any(), "the pool changed while running"
    finally:
        pool.close()
    assert np.isfinite(pool.kp).all()
