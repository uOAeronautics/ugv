"""Floor-plane visual velocity with explicit monocular scale assumptions.

This is independent of accelerometer integration, but not a measured metric
reference unless camera height and orientation are surveyed. Homography may
select furniture or an ambiguous plane; all quality gates are recorded.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def analyze(folder,height=.065,focal=968.,surveyed=False,stride=1):
    frames=[json.loads(l) for l in (folder/'frames.jsonl').read_text().splitlines() if l.strip()]
    frames=frames[::stride];results=[];reference=[];previous=None
    for frame in frames:
        path=folder/frame['file']
        if not path.exists():path=folder/'frames'/Path(frame['file']).name
        gray=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if gray is None:continue
        h,w=gray.shape;scale=w/1280.;k=np.array([[focal*scale,0,w/2],[0,focal*scale,h/2],[0,0,1.]])
        if previous is None:previous=(frame,gray);continue
        old,prev=previous;previous=(frame,gray);dt=frame['t']-old['t']
        if not .015<=dt<=.6:continue
        mask=np.zeros_like(prev);mask[int(h*.58):int(h*.96),int(w*.08):int(w*.92)]=255
        pts=cv2.goodFeaturesToTrack(prev,maxCorners=600,qualityLevel=.008,minDistance=8,mask=mask)
        row=dict(t=(frame['t']+old['t'])/2,dt=dt,accepted=False)
        if pts is None or len(pts)<24:row['reason']='too_few_floor_features';results.append(row);continue
        nxt,status,_=cv2.calcOpticalFlowPyrLK(prev,gray,pts,None,winSize=(25,25),maxLevel=3)
        back,backstatus,_=cv2.calcOpticalFlowPyrLK(gray,prev,nxt,None,winSize=(25,25),maxLevel=3)
        ok=(status[:,0]>0)&(backstatus[:,0]>0)&(np.linalg.norm(back[:,0]-pts[:,0],axis=1)<1.)
        a,b=pts[ok,0],nxt[ok,0]
        if len(a)<24:row['reason']='flow_tracking_failed';results.append(row);continue
        hom,inliers=cv2.findHomography(a,b,cv2.RANSAC,2.)
        if hom is None:row['reason']='homography_failed';results.append(row);continue
        keep=inliers[:,0]>0;ratio=float(keep.mean());row.update(inliers=int(keep.sum()),inlier_fraction=ratio)
        if keep.sum()<20 or ratio<.65:row['reason']='nonplanar_or_bad_flow';results.append(row);continue
        projected=cv2.perspectiveTransform(a[keep,None,:],hom)[:,0];row['reprojection_rmse_px']=float(np.sqrt(np.mean((projected-b[keep])**2)))
        count,rotations,translations,normals=cv2.decomposeHomographyMat(hom,k)
        candidates=[]
        for r,t,n in zip(rotations,translations,normals):
            n=n[:,0];translation=t[:,0]
            # Floor normal points camera-down; reject vertical furniture planes.
            if n[1]<.5 or abs(n[0])>.4:continue
            physical=-r.T@translation*height
            angle=float(np.arccos(np.clip((np.trace(r)-1)/2,-1,1)))
            if angle/dt>5 or np.linalg.norm(physical)/dt>3:continue
            # Floor clearance is invariant for a ground robot; prefer tiny vertical motion.
            score=abs(float(physical@n))+angle*.01
            candidates.append((score,physical,r,n))
        if not candidates and float(np.median(np.linalg.norm(b[keep]-a[keep],axis=1)))<.2:
            candidates=[(0.,np.zeros(3),np.eye(3),np.array([0.,1.,0.]))]
        if not candidates:row['reason']='no_plausible_floor_decomposition';results.append(row);continue
        _,motion,r,n=min(candidates,key=lambda v:v[0])
        floor_forward=np.array([0.,0.,1.]);floor_forward-=n*float(floor_forward@n);floor_forward/=np.linalg.norm(floor_forward)
        floor_left=np.cross(floor_forward,n);floor_left/=np.linalg.norm(floor_left)
        velocity=[float(motion@floor_forward/dt),float(motion@floor_left/dt),float(np.arctan2(r[0,2],r[2,2])/dt)]
        row.update(accepted=True,body_velocity=velocity,floor_normal_camera=n.tolist(),camera_displacement_m=motion.tolist(),
                   source='external_metric_pose' if surveyed else 'visual_planar_height_prior',height_m=height,
                   relative_scale_systematic_minimum=.05 if surveyed else .35)
        results.append(row);reference.append(row)
    (folder/'vision_quality.json').write_text(json.dumps(dict(schema_version=1,method='KLT forward/backward tracking; floor ROI; RANSAC homography; planar decomposition',camera_height_m=height,focal_px_at_1280=focal,surveyed_extrinsics=surveyed,accepted=len(reference),attempted=len(results),limitations=['Unknown pitch and plane selection can bias translation.','Scale is proportional to camera height and focal calibration.','No IMU acceleration is used; gyro comparison remains a separate cross-check.','Robot body forward assumed aligned with optical +Z; left optical -X.'],pairs=results),indent=2))
    (folder/'visual_reference.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reference))
    print(json.dumps(dict(accepted=len(reference),attempted=len(results),source='external_metric_pose' if surveyed else 'visual_planar_height_prior')))
    return reference

def analyze_gyro_floor(folder,height=.065,focal=968.):
    """Batch gyro-assisted floor tracking; reuse the independently tested guard."""
    from mecanum_online_guard import FloorMotionGuard
    from mecanum_fit import extract,read
    trials,mount=extract(folder);samples=read(folder/'samples.jsonl');frames=read(folder/'frames.jsonl')
    st=np.array([r['t'] for r in samples]);up=np.array(mount['body_axes_sensor'])[:,2]
    gyro=np.array([r['gyro_dps'] for r in samples]);bias=np.median(gyro[:min(300,len(gyro))],axis=0)
    yawrate=np.radians((gyro-bias)@up)
    for tr in trials:
        selected=(st>=tr['t'][0])&(st<=tr['t'][-1]);yawrate[selected]=np.interp(st[selected],tr['t'],np.asarray(tr['velocity'])[:,2])
    angles=np.r_[0.,np.cumsum((yawrate[1:]+yawrate[:-1])*.5*np.diff(st))]
    guard=FloorMotionGuard(height_m=height,focal_px_1280=focal,camera_offset_body=(0,0),max_radius_m=999.,width=640)
    rows=[];accepted=[];last=None
    for frame in frames:
        path=folder/frame['file'];gray=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if gray is None:continue
        t=float(frame['t']);yaw=float(np.interp(t,st,angles));r=guard.update(gray,t,yaw)
        if not np.isfinite(r.get('age_s',0)):r['age_s']=None
        if last is not None:
            previous_t,previous_yaw=last;dt=t-previous_t;r.update(t=(t+previous_t)/2,dt=dt)
            if r['accepted']:
                delta=np.asarray(r['delta_body_m']);r.update(body_velocity=[float(delta[0]/dt),float(delta[1]/dt),float((yaw-previous_yaw)/dt)],source='visual_planar_height_prior',height_m=height,rotation_source='gyro_rest_bias_corrected',relative_scale_systematic_minimum=.15)
                accepted.append(r)
            rows.append(r)
        last=t,yaw
    prior=folder/'visual_reference.jsonl'
    if prior.exists() and not (folder/'visual_reference_homography.jsonl').exists():(folder/'visual_reference_homography.jsonl').write_text(prior.read_text())
    prior.write_text(''.join(json.dumps(r)+'\n' for r in accepted))
    quality=dict(method='Gyro-assisted horizontal floor ray projection, KLT+RANSAC and robust translation; camera lever arm deferred to scale alignment',accepted=len(accepted),attempted=len(rows),camera_height_m=height,pairs=rows,limitations=['Pitch and roll of camera are assumed small; camera height remains a scale prior until training visual-inertial alignment.','Gyro is shared with velocity reference yaw; visual translation is independent of accelerometer integration but not gyro compensation.'])
    (folder/'vision_quality_gyro.json').write_text(json.dumps(quality,indent=2))
    print(json.dumps(dict(accepted=len(accepted),attempted=len(rows),method=quality['method'])))
    return accepted


def align_visual_inertial(folder, height_prior=.065, bootstrap=200, imu_offset=(-.051302,.002974), camera_offset=(.10,0.)):
    """Estimate metric scale on TRAIN excitation, reserve whole heldout trials.

    Uses integrated specific acceleration against visual velocity increments plus
    rotating-frame transport. This estimates a shared visual/inertial scale,
    not independent surveyed truth. IMU/camera lever arms are configurable.
    """
    from mecanum_fit import extract, read
    trials,mount=extract(folder);basis=np.array(mount['body_axes_sensor'])
    samples=read(folder/'samples.jsonl');visual=read(folder/'visual_reference.jsonl');events=read(folder/'events.jsonl')
    equations=[];diagnostics=[]
    for tr in trials:
        t=np.asarray(tr['t']);raw=[s for s in samples if t[0]<=s['t']<=t[-1]]
        if len(raw)<10:continue
        st=np.array([s['t'] for s in raw]);a=np.array([s['accel_g'] for s in raw])
        event=next(e for e in events if e.get('event')=='trial_start' and e.get('trial',e.get('id'))==tr['id'])
        before=[s['accel_g'] for s in samples if event['t']-.6<s['t']<event['t']-.04]
        after=[s['accel_g'] for s in samples if st[-1]-.35<s['t']<=st[-1]]
        if len(before)<5 or len(after)<5:continue
        g0=np.median(before,axis=0);g1=np.median(after,axis=0)
        f=((st-st[0])/(st[-1]-st[0]))[:,None];acc=(a-g0-f*(g1-g0))@basis*9.80665
        yaw=np.interp(st,t,np.array(tr['velocity'])[:,2]);alpha=np.gradient(yaw,st)
        rx,ry=imu_offset;acc[:,0]+=alpha*ry+yaw*yaw*rx;acc[:,1]+=-alpha*rx+yaw*yaw*ry
        rows=[r for r in visual if st[0]<=r['t']<=st[-1]]
        if len(rows)<5:continue
        vt=np.array([r['t'] for r in rows]);q=np.array([r['body_velocity'][:2] for r in rows])/height_prior
        w=np.interp(vt,st,yaw);cx,cy=camera_offset
        # Camera velocity = COM velocity + omega cross camera lever arm.
        # Offset is metric; remove it from target transport as a known term.
        camera_correction=np.column_stack((-w*cy,w*cx))
        for j in range(len(rows)-2):
            k=j+2;duration=vt[k]-vt[j]
            if not .08<=duration<=.45:continue
            ts=np.unique(np.r_[vt[j],st[(st>vt[j])&(st<vt[k])],vt[k]])
            if len(ts)<4:continue
            aa=np.column_stack([np.interp(ts,st,acc[:,axis]) for axis in (0,1)])
            # Covariant body velocity increment: delta v + integral omega J v.
            transport=np.column_stack((-w[j:k+1]*q[j:k+1,1],w[j:k+1]*q[j:k+1,0]))
            excitation=q[k]-q[j]+np.trapezoid(transport,vt[j:k+1],axis=0)
            correction=camera_correction[k]-camera_correction[j]+np.trapezoid(np.column_stack((-w[j:k+1]*camera_correction[j:k+1,1],w[j:k+1]*camera_correction[j:k+1,0])),vt[j:k+1],axis=0)
            target=np.trapezoid(aa,ts,axis=0)+correction
            if np.linalg.norm(excitation)*height_prior<.015:continue
            equations.append((tr['id'],tr['split'],complex(*excitation),complex(*target)))
    train=[e for e in equations if e[1]!='heldout'];heldout=[e for e in equations if e[1]=='heldout']
    if len(train)<12 or len({e[0] for e in train})<3:
        result=dict(valid=False,reason='insufficient independent training excitation',training_equations=len(train))
        (folder/'visual_inertial_alignment.json').write_text(json.dumps(result,indent=2));return result
    def solve(rows):
        x=np.array([r[2] for r in rows]);y=np.array([r[3] for r in rows]);weights=np.ones(len(x));c=complex(height_prior)
        for _ in range(15):
            c=np.sum(weights*np.conj(x)*y)/max(float(np.sum(weights*abs(x)**2)),1e-12)
            error=abs(y-c*x);scale=max(float(np.median(error))*1.4826,.01);weights=np.minimum(1,1.5*scale/np.maximum(error,1e-9))
        return c
    coefficient=solve(train);height=float(abs(coefficient));angle=float(np.angle(coefficient));ids=list(dict.fromkeys(e[0] for e in train));rng=np.random.default_rng(23);draws=[]
    for _ in range(bootstrap):
        selected=rng.choice(ids,len(ids),replace=True);rows=[e for trial_id in selected for e in train if e[0]==trial_id];draws.append(abs(solve(rows)))
    interval=np.percentile(draws,[2.5,97.5]).tolist()
    def score(rows):
        if not rows:return None
        residual=np.array([e[3]-coefficient*e[2] for e in rows]);return dict(equations=len(rows),trials=len({e[0] for e in rows}),integrated_acceleration_rmse_mps=float(np.sqrt(np.mean(abs(residual)**2))))
    valid=bool(.04<=height<=.10 and abs(angle)<np.radians(25) and interval[1]-interval[0]<height*.8)
    result=dict(valid=valid,height_m=height,height_bootstrap_95_m=interval,camera_to_body_yaw_rad=angle,training=score(train),heldout=score(heldout),source='visual_inertial_train_scaled',scale_fit_trial_ids=ids,imu_offset_m=list(imu_offset),camera_offset_m=list(camera_offset),limitations=['Metric scale shares accelerometer calibration; heldout images are independent of scale fitting but not an external surveyed reference.','Bootstrap resamples whole training trials and excludes systematic camera, IMU and mounting errors.','Floor homography ambiguity, vibration and tilt remain systematic errors; invalid alignment is not promoted.'])
    (folder/'visual_inertial_alignment.json').write_text(json.dumps(result,indent=2))
    if valid:
        output=[]
        for r in visual:
            z=coefficient*complex(*r['body_velocity'][:2])/height_prior;w=r['body_velocity'][2]
            r=dict(r);r.update(body_velocity=[z.real+w*camera_offset[1],z.imag-w*camera_offset[0],w],source='visual_inertial_train_scaled',height_m=height,scale_bootstrap_95_m=interval);output.append(r)
        (folder/'reference.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in output))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--height',type=float,default=.065);p.add_argument('--focal',type=float,default=968.);p.add_argument('--surveyed-extrinsics',action='store_true');p.add_argument('--stride',type=int,default=1);p.add_argument('--align-imu',action='store_true');p.add_argument('--gyro-floor',action='store_true');a=p.parse_args()
    if a.gyro_floor:analyze_gyro_floor(a.folder,a.height,a.focal)
    else:analyze(a.folder,a.height,a.focal,a.surveyed_extrinsics,a.stride)

    if a.align_imu:print(json.dumps(align_visual_inertial(a.folder,a.height),indent=2))
