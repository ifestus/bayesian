import numpy as np
import scipy.stats as stats
import pymc3 as pm
import theano.tensor as T

data = np.loadtxt("data/mixture_data.csv", delimiter=",")

with pm.Model() as model:
    p1 = pm.Uniform("p", 0, 1)
    p2 = 1 - p1
    p = T.stack([p1, p2])
    assignment = pm.Categorical("assignment", p,
                             shape=data.shape[0],
                             testval=np.random.randint(0, 2, data.shape[0]))

    print("assignment type: {}".format(type(assignment)))
    print("assignment: {}".format(assignment.tag.test_value[:4]))

    sds = pm.Uniform("sds", 0, 100, shape=2)
    centers = pm.Normal("centers", mu=np.array([120, 190]),
                                    sd=np.array([10, 10]),
                                    shape=2)
    center_i = pm.Deterministic('center_i', centers[assignment])
    sd_i = pm.Deterministic('sd_i', sds[assignment])

    print("center_i: {}".format(center_i.tag.test_value[:4]))
    print("sd_i: {}".format(sd_i.tag.test_value[:4]))

    start = pm.find_MAP()
    step1 = pm.Metropolis(vars=[p, sds, centers])
    step2 = pm.ElemwiseCategorical(vars=[assignment])
    trace = pm.sample(25000, step=[step1, step2]) # start=start
    # Do some investigation

    trace = pm.sample(50000, step=[step1, step2], trace=trace)
    assignment_trace = trace["assignment"][25000:]

    center_trace = trace["centers"][25000:]
    prev_center_trace = trace["centers"][:25000]

    std_trace = trace["sds"][25000:]
    prev_std_trace = trace["sds"][:25000]

    p_trace = trace["p"][25000:]
    prev_p_trace = trace["p"][:25000]

norm_pdf = stats.norm.pdf
x = 175
v = p_trace * norm_pdf(x, loc=center_trace[:, 0], scale=std_trace[:, 0]) > \
        (1 - p_trace) * norm_pdf(x, loc=center_trace[:, 1], scale=std_trace[:, 1])

print("Probability of belonging to cluster 1: {}".format(v.mean()))
