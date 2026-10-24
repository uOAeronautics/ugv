"""Empirical mecanum dynamics, independent of hardware.

Coordinates: forward m/s, left m/s, CCW rad/s. Wheel inputs are signed
actual PWM / 4095 in the recorded hardware convention (not ideal wheel rpm).
Parameters are identified responses, not separately identifiable tire/motor constants.
"""
from __future__ import annotations
import numpy as np

WHEEL_ORDER = ('front-left', 'rear-left', 'front-right', 'rear-right')


def features(wheels, deadzone=0.):
    """Direction-asymmetric wheel response, saturation and coupled loading."""
    u = np.asarray(wheels, dtype=float)
    q = np.sign(u) * np.maximum(np.abs(u)-deadzone, 0) / (1-deadzone)
    p, n = np.maximum(q, 0), np.minimum(q, 0)
    # No intercept: a stopped robot has zero equilibrium velocity.
    return np.concatenate((p, n, q*np.abs(q), q*np.mean(np.abs(q), axis=-1, keepdims=True)), axis=-1)


def step(state, wheels, dt, profile, rng=None):
    state = np.asarray(state, dtype=float)
    target = features(wheels, profile['deadzone']) @ np.asarray(profile['coefficients'])
    braking = np.max(np.abs(wheels)) < 1e-8
    tau = np.asarray(profile['brake_tau_s'] if braking else profile['drive_tau_s'])
    decay = np.exp(-max(0., dt)/tau)
    result = decay*state+(1-decay)*target
    if rng is not None:
        # OU-like correlated velocity error: stationary covariance independent of dt.
        cov = np.asarray(profile['innovation_covariance'])
        # The fitted covariance is conditional transition noise, not PWM-scaled.
        # PWM scaling would double-count the input and understate low-power uncertainty.
        activity = 1.
        result += activity*rng.multivariate_normal(np.zeros(3), cov)*np.sqrt(1-np.exp(-2*dt/float(np.mean(tau))))
    return result


def rollout(wheels, times, profile, initial=None, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else None
    v = np.zeros(3) if initial is None else np.array(initial, dtype=float)
    velocities, poses = [v.copy()], [np.zeros(3)]
    for i in range(1, len(times)):
        dt = float(times[i]-times[i-1]); previous = v.copy()
        v = step(v, wheels[i-1], dt, profile, rng)
        mid = (previous+v)/2; theta = poses[-1][2]+mid[2]*dt/2
        c, s = np.cos(theta), np.sin(theta)
        poses.append(poses[-1]+dt*np.array([c*mid[0]-s*mid[1],s*mid[0]+c*mid[1],mid[2]]))
        velocities.append(v.copy())
    return np.asarray(velocities), np.asarray(poses)


def fit(trials, bootstrap=100, seed=17):
    """Fit on complete training trials; choose regularization/deadzone by train loss.

    Each trial has t[N], wheels[N,4], velocity[N,3], id, split. Velocity must
    use the documented body coordinate convention. Heldout trials never tune fit.
    """
    train = [t for t in trials if t.get('split') != 'heldout']
    heldout = [t for t in trials if t.get('split') == 'heldout']
    if len(train)<2: raise ValueError('Need at least two complete training trials')
    def solve(rows, deadzone, tau, brake_tau, ridge):
        x, y = [], []
        for tr in rows:
            dt = np.diff(tr['t']); v = np.asarray(tr['velocity']); w = np.asarray(tr['wheels'])[:-1]
            moving = np.max(np.abs(w),axis=1)>1e-8
            decay = np.exp(-dt/np.where(moving,tau,brake_tau))
            x.append(features(w,deadzone)*(1-decay[:,None]))
            y.append(v[1:]-decay[:,None]*v[:-1])
        x,y=np.concatenate(x),np.concatenate(y)
        coef=np.linalg.solve(x.T@x+ridge*np.eye(x.shape[1]),x.T@y)
        return coef, y-x@coef
    best=None
    # Fit axes independently: lateral scrub and yaw have different relaxation.
    for dz in (0., .05, .10, .16, .23):
        coefficients=np.zeros((16,3)); taus=[]; brakes=[]; losses=[]
        for axis in range(3):
            candidate=None
            for tau in (.04,.08,.15,.25,.4,.65,1.):
                for brake in (.025,.06,.12,.25,.5):
                    coef,residual=solve(train,dz,tau,brake,.000001)
                    loss=float(np.mean(residual[:,axis]**2))
                    if candidate is None or loss<candidate[0]: candidate=(loss,tau,brake,coef[:,axis])
            loss,tau,brake,coef=candidate;coefficients[:,axis]=coef;taus.append(tau);brakes.append(brake);losses.append(loss)
        # Normalize axes so angular units do not dominate deadzone choice.
        scale=np.std(np.concatenate([tr['velocity'] for tr in train]),axis=0)+1e-4
        score=float(np.sum(np.array(losses)/scale**2))
        if best is None or score<best[0]:best=(score,dz,coefficients,taus,brakes)
    _,dz,coef,taus,brakes=best
    profile=dict(schema_version=1,coordinates=['forward_mps','left_mps','ccw_radps'],wheel_order=list(WHEEL_ORDER),
                 deadzone=dz,coefficients=coef.tolist(),drive_tau_s=taus,brake_tau_s=brakes,
                 innovation_covariance=np.zeros((3,3)).tolist())
    def evaluate(rows):
        errors=[];individual=[]
        for tr in rows:
            predicted,pose=rollout(tr['wheels'],tr['t'],profile,initial=tr['velocity'][0])
            error=predicted-np.asarray(tr['velocity']);errors.append(error)
            # Integrate the observed twist using the same SE(2) midpoint rule.
            observed=np.asarray(tr['velocity']);truth=np.zeros(3)
            for j in range(1,len(tr['t'])):
                dt=tr['t'][j]-tr['t'][j-1];mid=(observed[j]+observed[j-1])/2
                heading=truth[2]+mid[2]*dt/2;c,s=np.cos(heading),np.sin(heading)
                truth+=dt*np.array([c*mid[0]-s*mid[1],s*mid[0]+c*mid[1],mid[2]])
            individual.append(dict(id=tr['id'],velocity_rmse=np.sqrt(np.mean(error**2,axis=0)).tolist(),
                                   integrated_twist_endpoint_error= (pose[-1]-truth).tolist()))
        if not errors:return dict(trial_count=0,velocity_rmse=None,trials=[])
        e=np.concatenate(errors)
        return dict(trial_count=len(rows),velocity_rmse=np.sqrt(np.mean(e**2,axis=0)).tolist(),
                    velocity_bias=np.mean(e,axis=0).tolist(),trials=individual)
    training=evaluate(train); testing=evaluate(heldout)
    residuals=[]
    for tr in train:
        v,_=rollout(tr['wheels'],tr['t'],profile,initial=tr['velocity'][0]);residuals.append(v-np.asarray(tr['velocity']))
    profile['innovation_covariance']=np.cov(np.concatenate(residuals).T).tolist()
    rng=np.random.default_rng(seed);draws=[]
    # Cluster bootstrap preserves correlated samples within each trial.
    for _ in range(bootstrap):
        rows=[train[i] for i in rng.integers(0,len(train),len(train))]
        boot=np.zeros_like(coef)
        for axis in range(3):boot[:,axis]=solve(rows,dz,taus[axis],brakes[axis],.000001)[0][:,axis]
        draws.append(boot)
    if draws:
        profile['coefficient_interval_95']={'low':np.percentile(draws,2.5,axis=0).tolist(),'high':np.percentile(draws,97.5,axis=0).tolist()}
    profile['evaluation']={'training':training,'heldout':testing,'split_unit':'whole trial','bootstrap_clusters':len(train),'bootstrap_draws':bootstrap}
    return profile


def _inverse_wishart(rng, scale, degrees):
    """Bartlett draw; inverse-Wishart covariance retains cross-axis coupling."""
    dimension=len(scale);a=np.zeros((dimension,dimension))
    for i in range(dimension):
        a[i,i]=np.sqrt(rng.chisquare(degrees-i))
        a[i,:i]=rng.normal(size=i)
    factor=np.linalg.cholesky(np.linalg.inv(scale))@a
    return np.linalg.inv(factor@factor.T)


def fit_bayesian(trials, draws=120, seed=29):
    """Matrix-normal inverse-Wishart regression with a discrete lag/deadzone prior.

    Likelihood: conditional OU transition errors, thinned to >=80ms within each
    training trial. Equal grid prior; no heldout data selects parameters. Shared
    scalar relaxation permits conjugate cross-axis covariance inference. The
    posterior is conditional on observed reference velocities (not sensor truth).
    """
    import math
    train=[tr for tr in trials if tr.get('split')!='heldout']
    heldout=[tr for tr in trials if tr.get('split')=='heldout']
    if len(train)<3:raise ValueError('Bayesian fit needs at least three training trials')
    transitions=[]
    for tr in train:
        t=np.asarray(tr['t']);v=np.asarray(tr['velocity']);w=np.asarray(tr['wheels'])
        index=[0]
        for j in range(1,len(t)):
            if t[j]-t[index[-1]]>=.08:index.append(j)
        for i,j in zip(index[:-1],index[1:]):
            # Do not average across a discontinuous control change; motor inputs
            # are time varying and transitions use their observed duration mean.
            dt=t[j]-t[i];mean_w=np.trapezoid(w[i:j+1],t[i:j+1],axis=0)/dt
            transitions.append((dt,v[i],v[j],mean_w,t[i:j+1],w[i:j+1]))
    n=len(transitions);p=16;d=3
    if n<20:raise ValueError('Too few observed transitions for Bayesian dynamics')
    dt=np.array([r[0] for r in transitions]);v0=np.array([r[1] for r in transitions]);v1=np.array([r[2] for r in transitions]);w=np.array([r[3] for r in transitions])
    precision=np.eye(p)*.01;scale0=np.diag([.02**2,.02**2,.08**2]);nu0=5.
    def multigamma(a):return d*(d-1)/4*math.log(math.pi)+sum(math.lgamma(a+(1-j)/2) for j in range(1,d+1))
    candidates=[]
    for deadzone in (0.,.05,.10,.16,.23):
        f=np.array([np.trapezoid(features(r[5],deadzone),r[4],axis=0)/r[0] for r in transitions])
        for tau in (.04,.08,.15,.25,.4,.65,1.):
            for brake in (.025,.06,.12,.25,.5):
                active=np.array([np.max(abs(r[5]))>1e-8 for r in transitions])
                decay=np.exp(-dt/np.where(active,tau,brake));noise=np.sqrt(np.maximum(1-decay**2,1e-8))
                x=f*((1-decay)/noise)[:,None];y=(v1-decay[:,None]*v0)/noise[:,None]
                kn=precision+x.T@x;mean=np.linalg.solve(kn,x.T@y)
                sn=scale0+y.T@y-mean.T@kn@mean;sn=(sn+sn.T)/2
                sign,logdet=np.linalg.slogdet(sn)
                if sign<=0:continue
                nun=nu0+n
                evidence= d/2*(np.linalg.slogdet(precision)[1]-np.linalg.slogdet(kn)[1])+nu0/2*np.linalg.slogdet(scale0)[1]-nun/2*logdet+multigamma(nun/2)-multigamma(nu0/2)-n*d/2*math.log(math.pi)-d*np.log(noise).sum()
                candidates.append(dict(deadzone=deadzone,tau=tau,brake=brake,mean=mean,precision=kn,scale=sn,df=nun,log_evidence=float(evidence)))
    logp=np.array([c['log_evidence'] for c in candidates]);weights=np.exp(logp-logp.max());weights/=weights.sum()
    rng=np.random.default_rng(seed);particles=[]
    for index in rng.choice(len(candidates),size=draws,p=weights):
        c=candidates[index];cov=_inverse_wishart(rng,c['scale'],c['df']);row_factor=np.linalg.cholesky(np.linalg.inv(c['precision']))
        coefficients=c['mean']+row_factor@rng.normal(size=(p,d))@np.linalg.cholesky(cov).T
        particles.append(dict(deadzone=c['deadzone'],coefficients=coefficients.tolist(),drive_tau_s=[c['tau']]*3,brake_tau_s=[c['brake']]*3,innovation_covariance=cov.tolist()))
    winner=candidates[int(np.argmax(weights))]
    result=dict(schema_version=2,coordinates=['forward_mps','left_mps','ccw_radps'],wheel_order=list(WHEEL_ORDER),deadzone=winner['deadzone'],coefficients=winner['mean'].tolist(),drive_tau_s=[winner['tau']]*3,brake_tau_s=[winner['brake']]*3,innovation_covariance=(winner['scale']/(winner['df']-d-1)).tolist(),posterior_particles=particles)
    def endpoint(velocity,times):
        pose=np.zeros(3)
        for j in range(1,len(times)):
            delta=times[j]-times[j-1];mid=(velocity[j]+velocity[j-1])/2;angle=pose[2]+mid[2]*delta/2;c,s=np.cos(angle),np.sin(angle)
            pose+=delta*np.array([c*mid[0]-s*mid[1],s*mid[0]+c*mid[1],mid[2]])
        return pose
    def evaluate(rows):
        metrics=[];errors=[];coverage=[]
        for tr in rows:
            observed=np.asarray(tr['velocity']);predicted=[];ends=[]
            for i,particle in enumerate(particles):
                v,poses=rollout(tr['wheels'],tr['t'],particle,initial=observed[0],seed=seed+i);predicted.append(v);ends.append(poses[-1])
            predicted=np.asarray(predicted);lo,hi=np.percentile(predicted,[2.5,97.5],axis=0);mean=predicted.mean(axis=0)
            e=mean-observed;errors.append(e);covered=(observed>=lo)&(observed<=hi);coverage.append(covered)
            truth=endpoint(observed,np.asarray(tr['t']));endlo,endhi=np.percentile(ends,[2.5,97.5],axis=0)
            metrics.append(dict(id=tr['id'],velocity_rmse=np.sqrt(np.mean(e*e,axis=0)).tolist(),velocity_interval_95_coverage=covered.mean(axis=0).tolist(),integrated_twist_endpoint_error=(np.mean(ends,axis=0)-truth).tolist(),endpoint_interval_95_low=endlo.tolist(),endpoint_interval_95_high=endhi.tolist(),endpoint_covered=((truth>=endlo)&(truth<=endhi)).tolist()))
        return dict(trial_count=len(rows),velocity_rmse=np.sqrt(np.mean(np.concatenate(errors)**2,axis=0)).tolist() if errors else None,velocity_interval_95_coverage=np.mean(np.concatenate(coverage),axis=0).tolist() if coverage else None,trials=metrics)
    result['bayesian']=dict(method='Matrix-normal inverse-Wishart conditional OU transitions; equal categorical prior over deadzone/drive/brake grid',training_transitions=n,transition_spacing_min_s=.08,coefficient_prior_row_precision=.01,covariance_prior_scale=scale0.tolist(),covariance_prior_df=nu0,posterior_draws=draws,grid_posterior=[dict(deadzone=c['deadzone'],drive_tau_s=c['tau'],brake_tau_s=c['brake'],probability=float(prob)) for c,prob in zip(candidates,weights) if prob>1e-6],limitations=['Conditional likelihood treats thinned velocity residuals as Markov; reference and calibration systematic errors are not independent observations.','Common relaxation time preserves conjugacy; axis-dependent residual covariance captures coupled disturbances.','Posterior uncertainty is conditional on selected model family and training environment; coverage on heldout trials must be checked.'])
    result['evaluation']={'training':evaluate(train),'heldout':evaluate(heldout),'split_unit':'whole trajectory'}
    coefficients=np.array([p['coefficients'] for p in particles]);result['coefficient_interval_95']={'low':np.percentile(coefficients,2.5,axis=0).tolist(),'high':np.percentile(coefficients,97.5,axis=0).tolist()}
    return result


def posterior_rollout(wheels, times, profile, initial=None, seed=0):
    """Draw one correlated parameter particle per trajectory, then process noise."""
    rng=np.random.default_rng(seed);particles=profile.get('posterior_particles')
    if not particles:raise ValueError('Profile has no Bayesian posterior particles')
    particle=particles[int(rng.integers(len(particles)))]
    return rollout(wheels,times,particle,initial=initial,seed=int(rng.integers(2**31)))
