"""Conservative relative floor odometry for calibration workspace monitoring.

Scale uses a camera height prior, not surveyed ground truth. A lost pair latches
unhealthy because motion during the gap cannot safely be inferred. No actuation.
"""
import math
import cv2
import numpy as np


def rotation(angle):
    c,s=math.cos(angle),math.sin(angle)
    return np.array([[c,-s],[s,c]])


class FloorMotionGuard:
    def __init__(self,height_m=.06635,focal_px_1280=968.,camera_offset_body=(.10,0),
                 max_radius_m=.8,width=640,max_gap_s=.3,floor_normal_camera=(0,1,0),
                 denoise_sigma=.7,feature_quality=.02):
        self.height=float(height_m);self.focal=float(focal_px_1280)
        self.denoise_sigma=float(denoise_sigma);self.feature_quality=float(feature_quality)
        self.offset=np.asarray(camera_offset_body,dtype=float)
        self.radius=float(max_radius_m);self.width=width;self.max_gap=max_gap_s
        self.fixed_normal=np.asarray(floor_normal_camera,dtype=float)
        self.fixed_normal/=np.linalg.norm(self.fixed_normal)
        self.processing_configuration=dict(width_px=self.width,denoise_kernel_px=[3,3],
            denoise_sigma=self.denoise_sigma,denoise_before_resize=True,
            corner_quality=self.feature_quality,corner_min_distance_px=6,
            homography_ransac_px=1.5,minimum_inlier_fraction=.7)
        self.reset()

    def reset(self):
        self.previous=None;self.position=np.zeros(2);self.origin_yaw=None
        self.last_good=None;self.lost=False

    def _ground(self,p,shape):
        h,w=shape;f=self.focal*w/1280.
        rays=np.column_stack(((p[:,0]-w/2)/f,(p[:,1]-h/2)/f,np.ones(len(p))))
        normal=self.fixed_normal
        points=rays*(self.height/(rays@normal))[:,None]
        forward=np.array([0.,0.,1.]);forward-=normal*(forward@normal);forward/=np.linalg.norm(forward)
        left=np.cross(forward,normal);left/=np.linalg.norm(left)
        return np.column_stack((points@forward,points@left))

    def estimate_pair(self,a,b,shape,dyaw):
        """Pixel correspondences -> camera displacement in previous body axes."""
        h,w=shape
        valid=(a[:,1]>h*.55)&(b[:,1]>h*.55)&(b[:,1]<h*.98)&(b[:,0]>0)&(b[:,0]<w)
        a,b=a[valid],b[valid]
        if len(a)<24:return None,dict(reason='too_few_floor_tracks',inliers=len(a))
        hom,mask=cv2.findHomography(a,b,cv2.RANSAC,1.5)
        if hom is None:return None,dict(reason='homography_failed')
        keep=mask[:,0]>0
        if keep.sum()<24 or keep.mean()<.7:return None,dict(reason='nonplanar_flow',inliers=int(keep.sum()))
        a,b=a[keep],b[keep]
        old=self._ground(a,shape);new=self._ground(b,shape)
        shifts=old-new@rotation(dyaw).T
        delta=np.median(shifts,axis=0)
        errors=np.linalg.norm(shifts-delta,axis=1)
        gate=max(.0025,3*float(np.median(errors)))
        inliers=errors<min(gate,.012)
        if inliers.sum()<24 or inliers.mean()<.75:return None,dict(reason='inconsistent_floor_translation',inliers=int(inliers.sum()))
        spread=np.ptp(a[inliers],axis=0)/np.array([w,h])
        residual=float(np.sqrt(np.mean(errors[inliers]**2)))
        if residual>.004:return None,dict(reason='floor_model_residual',residual_m=residual)
        if spread[0]<.25 or spread[1]<.12:return None,dict(reason='insufficient_floor_coverage')
        delta=np.median(shifts[inliers],axis=0)
        # Camera world displacement includes the rotating camera lever arm.
        delta_body=delta-(rotation(dyaw)-np.eye(2))@self.offset
        return delta_body,dict(reason='ok',inliers=int(inliers.sum()),residual_m=residual,
                               inlier_fraction=float(inliers.mean()),coverage=spread.tolist())

    def update(self,gray,t,yaw_ccw):
        if gray is None:return self._result(t,False,'missing_frame')
        if gray.ndim==3:gray=cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
        # Mild sensor-noise suppression before resizing prevents tracking gain
        # noise as floor texture. Stronger corners preserve real floor marks.
        if self.denoise_sigma>0:gray=cv2.GaussianBlur(gray,(3,3),self.denoise_sigma)
        h,w=gray.shape
        if w!=self.width:gray=cv2.resize(gray,(self.width,round(h*self.width/w)))
        if self.origin_yaw is None:self.origin_yaw=float(yaw_ccw)
        old=self.previous;self.previous=(gray,float(t),float(yaw_ccw))
        if old is None:return self._result(t,False,'initializing',latch=False)
        prev,pt,pyaw=old;dt=t-pt
        if prev.shape!=gray.shape or not 0<dt<=self.max_gap:return self._result(t,False,'frame_gap')
        mask=np.zeros_like(prev);h,w=gray.shape;mask[int(h*.56):int(h*.96),int(w*.06):int(w*.94)]=255
        p=cv2.goodFeaturesToTrack(prev,500,self.feature_quality,6,mask=mask)
        if p is None or len(p)<24:return self._result(t,False,'too_few_features')
        q,status,_=cv2.calcOpticalFlowPyrLK(prev,gray,p,None,winSize=(25,25),maxLevel=3)
        if q is None:return self._result(t,False,'flow_failed')
        back,bs,_=cv2.calcOpticalFlowPyrLK(gray,prev,q,None,winSize=(25,25),maxLevel=3)
        if back is None:return self._result(t,False,'backflow_failed')
        ok=(status[:,0]>0)&(bs[:,0]>0)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<.8)
        dyaw=math.atan2(math.sin(yaw_ccw-pyaw),math.cos(yaw_ccw-pyaw))
        delta,quality=self.estimate_pair(p[ok,0],q[ok,0],gray.shape,dyaw)
        if delta is None:return self._result(t,False,quality.pop('reason'),**quality)
        if np.linalg.norm(delta)/dt>1.5 or abs(dyaw)/dt>5:return self._result(t,False,'implausible_speed')
        self.position+=rotation(pyaw-self.origin_yaw)@delta
        self.last_good=t
        return self._result(t,True,'ok',delta_body_m=delta.tolist(),**{k:v for k,v in quality.items() if k!='reason'})

    def _result(self,t,accepted,reason,latch=True,**quality):
        if not accepted and latch:self.lost=True
        age=float('inf') if self.last_good is None else max(0.,float(t-self.last_good))
        return dict(accepted=accepted,reason=reason,position_world_m=self.position.tolist(),
                    fresh=bool(accepted and not self.lost),healthy=not self.lost,age_s=age,
                    within_radius=bool(not self.lost and np.linalg.norm(self.position)<=self.radius),
                    radius_m=float(np.linalg.norm(self.position)),scale_source='estimated_camera_height',**quality)


class PlaneHomographyGuard(FloorMotionGuard):
    """Known initial floor normal removes planar decomposition sign ambiguity.

    R is scene rotation old-camera to new-camera; translation is recovered from
    the normal column after the tangent columns determine homography scale.
    Camera attitude is propagated only for accepted floor homographies.
    """
    def __init__(self,*args,floor_normal_camera=(0,1,0),**kwargs):
        self.initial_normal=np.asarray(floor_normal_camera,dtype=float)
        self.initial_normal/=np.linalg.norm(self.initial_normal)
        super().__init__(*args,**kwargs)
    def reset(self):
        super().reset();self.normal=self.initial_normal.copy()
    def estimate_pair(self,a,b,shape,dyaw):
        h,w=shape;valid=(a[:,1]>h*.55)&(b[:,1]>h*.52)&(b[:,1]<h*.99)&(b[:,0]>0)&(b[:,0]<w)
        a,b=a[valid],b[valid]
        if len(a)<24:return None,dict(reason='too_few_floor_tracks',inliers=len(a))
        H,mask=cv2.findHomography(a,b,cv2.RANSAC,1.5)
        if H is None:return None,dict(reason='homography_failed')
        keep=mask[:,0]>0
        if keep.sum()<24 or keep.mean()<.7:return None,dict(reason='nonplanar_flow',inliers=int(keep.sum()))
        spread=np.ptp(a[keep],axis=0)/np.array([w,h])
        if spread[0]<.25 or spread[1]<.12:return None,dict(reason='insufficient_floor_coverage')
        f=self.focal*w/1280.;K=np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]])
        M=np.linalg.solve(K,H@K);n=self.normal
        seed=np.array([1.,0,0]);b0=seed-n*(seed@n);b0/=np.linalg.norm(b0)
        b1=np.cross(n,b0);B=np.column_stack((b0,b1,n))
        tangent=M@B[:,:2];scale=1/np.sqrt(np.mean(np.sum(tangent*tangent,axis=0)))
        if np.linalg.det(M)<0:scale=-scale
        u,_,vt=np.linalg.svd(scale*tangent,full_matrices=False);Q=u@vt
        R=np.column_stack((Q[:,0],Q[:,1],np.cross(Q[:,0],Q[:,1])))@B.T
        tn=(scale*M-R)@n
        displacement=-R.T@tn*self.height
        next_normal=R@n;next_normal/=np.linalg.norm(next_normal)
        # The tangent columns are independent of translation; their deviation
        # from orthonormal rotation detects a non-floor or ill-conditioned fit.
        tangent_error=float(np.linalg.norm(scale*tangent-Q))
        yaw_visual=float(np.arctan2(R[0,2],R[2,2]))
        yaw_error=float(np.arctan2(np.sin(yaw_visual-dyaw),np.cos(yaw_visual-dyaw)))
        projected=cv2.perspectiveTransform(a[keep,None,:],H)[:,0]
        reprojection=float(np.sqrt(np.mean(np.sum((projected-b[keep])**2,axis=1))))
        quality=dict(inliers=int(keep.sum()),inlier_fraction=float(keep.mean()),coverage=spread.tolist(),
                     reprojection_rmse_px=reprojection,tangent_error=tangent_error,
                     visual_yaw_rad=yaw_visual,gyro_yaw_error_rad=yaw_error,floor_normal_camera=next_normal.tolist())
        if tangent_error>.035:return None,dict(reason='plane_rotation_inconsistent',**quality)
        if abs(yaw_error)>.025+.3*abs(dyaw):return None,dict(reason='gyro_rotation_inconsistent',**quality)
        if next_normal[1]<.7 or abs(displacement@n)>.012:return None,dict(reason='implausible_floor_attitude',**quality)
        if np.linalg.norm(displacement)>.12:return None,dict(reason='implausible_pair_translation',**quality)
        forward=np.array([0.,0,1.]);forward-=n*(forward@n);forward/=np.linalg.norm(forward)
        left=np.cross(forward,n);left/=np.linalg.norm(left)
        delta=np.array([displacement@forward,displacement@left])-(rotation(dyaw)-np.eye(2))@self.offset
        self.normal=next_normal
        return delta,dict(reason='ok',residual_m=reprojection*self.height/f,**quality)
