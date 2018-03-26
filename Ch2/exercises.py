import numpy as np
import pymc3 as pm
import scipy.stats as stats
import theano.tensor as tt

# Set some constants
N = 100
X = 35

with pm.Model() as model:
    p = pm.Uniform("freq_cheating", 0, 1)
    true_answers = pm.Bernoulli("truths", p, shape=N, testval=np.random.binomial(1, 0.5, N))
    first_coin_flips = pm.Bernoulli("first_flips", 0.5, shape=N, testval=np.random.binomial(1, 0.5, N))
    second_coin_flips = pm.Bernoulli("second_flips", 0.5, shape=N, testval=np.random.binomial(1, 0.5, N))

    val = first_coin_flips*true_answers + (1-first_coin_flips)*second_coin_flips
    observed_proportion = pm.Deterministic("observed_proportion", tt.sum(val)/float(N))

    observations = pm.Binomial("obs", N, observed_proportion, observed=X)

    step = pm.Metropolis()
    trace = pm.sample(40000, step=step)
    burned_trace = trace[15000:]

