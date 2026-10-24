"""Physical and statistical invariants for the identified response model."""
import numpy as np
from ugv.mecanum_dynamics import features, rollout, step


def profile():
    coef=np.zeros((16,3));coef[:4,0]=.25;coef[4:8,0]=.25
    return dict(deadzone=.1,coefficients=coef.tolist(),drive_tau_s=[.2]*3,
                brake_tau_s=[.05]*3,innovation_covariance=np.diag([.01,.01,.001]).tolist())


def test_deadzone_and_zero_equilibrium():
    assert np.all(features([.05,-.05,0,0],.1)==0)
    np.testing.assert_allclose(step([0,0,0],[0]*4,.1,profile()),[0]*3)


def test_exact_lag_is_invariant_to_time_subdivision():
    p=profile();one=step([0]*3,[1]*4,.2,p)
    two=step(step([0]*3,[1]*4,.1,p),[1]*4,.1,p)
    np.testing.assert_allclose(one,two)
    assert step([1,0,0],[0]*4,.2,p)[0] < .02


def test_probabilistic_rollout_reproducible_and_covariance_nonzero():
    p=profile();times=np.linspace(0,1,101);wheels=np.ones((101,4))
    a,_=rollout(wheels,times,p,seed=3);b,_=rollout(wheels,times,p,seed=3)
    np.testing.assert_array_equal(a,b)
    endpoints=np.array([rollout(wheels,times,p,seed=i)[0][-1] for i in range(100)])
    assert np.all(np.diag(np.cov(endpoints.T))>0)


def test_identification_recovers_unseen_trial_dynamics():
    from ugv.mecanum_dynamics import fit
    p=profile();p['deadzone']=.1;p['drive_tau_s']=[.15]*3;p['brake_tau_s']=[.06]*3
    coef=np.zeros((16,3));coef[:4]=[[.2,-.2,.5],[.2,.2,.5],[.2,.2,-.5],[.2,-.2,-.5]];coef[4:8]=coef[:4]
    p['coefficients']=coef.tolist();rng=np.random.default_rng(2);trials=[]
    for i in range(24):
        times=np.arange(0,1.2,.025);wheels=np.tile(rng.uniform(-1,1,4),(len(times),1));wheels[times>.7]=0
        velocity,_=rollout(wheels,times,p)
        trials.append(dict(id=i,t=times,wheels=wheels,velocity=velocity,split='heldout' if i%3==2 else 'train'))
    fitted=fit(trials,bootstrap=8)
    assert fitted['evaluation']['heldout']['trial_count']==8
    assert max(fitted['evaluation']['heldout']['velocity_rmse']) < .001


def test_bayesian_posterior_retains_parameter_and_process_uncertainty():
    from ugv.mecanum_dynamics import fit_bayesian, posterior_rollout
    p=profile();rng=np.random.default_rng(21);trials=[]
    for i in range(9):
        times=np.arange(0,1.6,.025);wheels=np.vstack([np.tile(rng.uniform(-1,1,4),(32,1)) for _ in range(2)])
        velocity,_=rollout(wheels,times,p)
        trials.append(dict(id=i,t=times,wheels=wheels,velocity=velocity,split='heldout' if i%3==2 else 'train'))
    fitted=fit_bayesian(trials,draws=12)
    assert fitted['evaluation']['heldout']['trial_count']==3
    assert abs(sum(g['probability'] for g in fitted['bayesian']['grid_posterior'])-1)<1e-4
    coefficients=np.asarray([particle['coefficients'] for particle in fitted['posterior_particles']])
    assert np.max(np.std(coefficients,axis=0))>0
    a,_=posterior_rollout(wheels,times,fitted,seed=3);b,_=posterior_rollout(wheels,times,fitted,seed=4)
    assert not np.allclose(a,b)
