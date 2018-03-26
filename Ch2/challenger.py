import pymc3 as pm
import numpy as np
import theano.tensor as tt

challenger_data = np.genfromtxt("data/challenger_data.csv", skip_header=1,
                                usecols=[1, 2], missing_values="NA",
                                delimiter=",")
challenger_data = challenger_data[~np.isnan(challenger_data[:,1])]

temperature = challenger_data[:, 0]
D = challenger_data[:, 1] # defect or not?

with pm.Model() as model:
    beta = pm.Normal("beta", mu=0, tau=1, testval=0)
    alpha = pm.Normal("alpha", mu=0, tau=1, testval=0)
    p = pm.Deterministic("p", 1.0/(1.0 + tt.exp(beta*temperature + alpha)))

    observed = pm.Bernoulli("bernoulli_obs", p, observed=D)
    simulated_data = pm.Beroulli("simulated_obs", p)

    start = pm.find_MAP()
    step = pm.Metropolis()
    trace = pm.sample(120000, step=step, start=start)
    burned_trace = trace[100000::2]

alpha_samples = burned_trace['alpha'][:, None]
beta_samples = burned_trace['beta'][:, None]

print(alpha_samples[0])
print(beta_samples[0])

t = np.linspace(temperature.min()-5, temperature.max()+5, 50)[:, None]
p_t = logistic(t.T, beta_samples, alpha_samples)
