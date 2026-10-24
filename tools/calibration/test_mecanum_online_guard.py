import unittest
import numpy as np
from mecanum_online_guard import FloorMotionGuard,rotation

class FloorGuardTests(unittest.TestCase):
    def project(self,ground,height=.06635):
        return np.column_stack((320-ground[:,1]/ground[:,0]*484,180+height/ground[:,0]*484)).astype(np.float32)
    def test_translation_yaw_and_lever_arm(self):
        guard=FloorMotionGuard()
        pixels=np.array([(u,v) for u in np.linspace(60,580,14) for v in np.linspace(220,335,10)],np.float32)
        old=guard._ground(pixels,(360,640))
        for angle in [-.03,0,.03]:
            for motion in [np.array([.004,0]),np.array([0,.004]),np.array([-.004,-.003])]:
                camera_motion=motion+(rotation(angle)-np.eye(2))@guard.offset
                new=(old-camera_motion)@rotation(angle)
                estimate,quality=guard.estimate_pair(pixels,self.project(new),(360,640),angle)
                self.assertIsNotNone(estimate,quality)
                np.testing.assert_allclose(estimate,motion,atol=1e-6)
    def test_wrong_plane_rejected(self):
        guard=FloorMotionGuard()
        pixels=np.array([(u,v) for u in np.linspace(60,580,14) for v in np.linspace(220,335,10)],np.float32)
        # A vertical object's pixel translation does not match constant floor translation.
        estimate,quality=guard.estimate_pair(pixels,pixels+np.array([20,0],np.float32),(360,640),0)
        self.assertIsNone(estimate,quality)
    def test_loss_latches_and_reset_recovers(self):
        guard=FloorMotionGuard();blank=np.zeros((360,640),np.uint8)
        self.assertFalse(guard.update(blank,0,0)['fresh'])
        lost=guard.update(blank,.05,0)
        self.assertFalse(lost['healthy']);self.assertFalse(lost['within_radius'])
        guard.reset();self.assertTrue(guard.update(blank,1,0)['healthy'])
    def test_long_gap_rejected(self):
        guard=FloorMotionGuard();blank=np.zeros((360,640),np.uint8)
        guard.update(blank,0,0)
        self.assertEqual(guard.update(blank,1,0)['reason'],'frame_gap')

if __name__=='__main__':unittest.main()

class HomographyGuardTests(unittest.TestCase):
    def test_exact_pitch_roll_and_translation(self):
        from mecanum_online_guard import PlaneHomographyGuard
        pixels=np.array([(u,v) for u in np.linspace(60,580,14) for v in np.linspace(220,335,10)],np.float32)
        K=np.array([[484.,0,320],[0,484.,180],[0,0,1]])
        for rotvec in ([.006,.02,.005],[-.008,-.03,.007],[.01,0,-.01]):
            guard=PlaneHomographyGuard(camera_offset_body=(0,0))
            import cv2
            R=cv2.Rodrigues(np.array(rotvec,float))[0]
            d=np.array([.003,0,.005]);n=np.array([0,1.,0])
            H=K@(R-np.outer(R@d,n)/guard.height)@np.linalg.inv(K)
            new=cv2.perspectiveTransform(pixels[:,None],H)[:,0]
            dyaw=np.arctan2(R[0,2],R[2,2])
            estimate,quality=guard.estimate_pair(pixels,new,(360,640),dyaw)
            self.assertIsNotNone(estimate,quality)
            np.testing.assert_allclose(estimate,[.005,-.003],atol=1e-6)
            np.testing.assert_allclose(guard.normal,R@n,atol=1e-6)
