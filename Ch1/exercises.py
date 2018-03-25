import pymc3 as pm
import numpy as np
import theano.tensor as tf

count_data = np.loadtxt("data/txtdata.csv")
n_count_data = len(count_data)

with pm.Model() as model:
    alpha = 1.0/count_data.mean()

    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)

    tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data-1)

    idx = np.arange(n_count_data)
    lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)

    observation = pm.Poisson("obs", lambda_, observed=count_data)

    step = pm.Metropolis()
    trace = pm.sample(10000, tune=5000, step=step)

lambda_1_samples = trace['lambda_1']
print(type(lambda_1_samples))
lambda_2_samples = trace['lambda_2']
print(type(lambda_1_samples))
tau_samples = trace['tau']
print(type(tau_samples))

print(lambda_1_samples.shape)
print(lambda_2_samples.shape)
print(tau_samples.shape)

print("let's get some information about things:\n")
print("lambda_1: {}".format(lambda_1_samples[:5]))
print("lambda_2: {}".format(lambda_2_samples[:5]))
print("tau: {}".format(tau_samples[:5]))

N = tau_samples.shape[0]
expected_texts_per_day = np.zeros(n_count_data)
for day in range(0, n_count_data):
    # ix is now an boolean array of size tau.shape[0], where True if
    # ix < tau_samples, otherwise False
    ix = day < tau_samples

    # Each posterior corresponds to a value of tau. If the day we're
    # estimating is before tau, lambda_1_sapmle. Otherwise, lambda_2_sample.
    # these sample values seem to be some kind of MC search... we're just
    # doign some averages over samples right now.
    expected_texts_per_day[day] = (lambda_1_samples[ix].sum()
                                   + lambda_2_samples[~ix].sum()) / N

# Exercise 1: compute the mean of posteriors lambda_1 and lambda_2
mean_lambda1 = lambda_1_samples.mean()
mean_lambda2 = lambda_2_samples.mean()

print("lambda_1_samples mean: {}".format(mean_lambda1))
print("lambda_2_samples mean: {}".format(mean_lambda2))

# Exercise 2: What is the expected percentage increase in text-messages?
expected_increase = (lambda_1_samples/lambda_2_samples).mean()
diff_in_means = mean_lambda1/mean_lambda2

print("% Expected Increase: {}".format(expected_increase))
print("Difference in means: {}".format(diff_in_means))

# What is the mean of lambda1 given that we know tau is less than 45?
x = lambda_1_samples[tau_samples < 45].mean()

print("Mean of lambda1 give that we know tau is less than 45: {}".format(x))


