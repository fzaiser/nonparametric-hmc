# A very simple example for which the answer is known.

import sys

from infer import np_dhmc
from ppl import ProbCtx, run_prob_prog

from typing import Any, Callable, Iterator, List, Tuple
import torch
import time
from torch.distributions import Normal

import numpy as np

mu0 = 0.0
sigma0 = 1.0
sigma = 1.0
z = 4.0

def analytic_example(ctx: ProbCtx) -> float:
    """An analytic example"""
    mu = ctx.sample(Normal(mu0, sigma0), is_cont=True)
    ctx.observe(torch.tensor(z, requires_grad=True), Normal(mu, sigma))
    return mu

def run(sampler: Callable) -> dict:
    torch.manual_seed(42)
    start = time.time()
    results = sampler()
    stop = time.time()
    elapsed = stop - start
    return {
        "time": elapsed,
        "samples": results,
    }

samples = {}
samples["hmc"] = run( lambda: np_dhmc(lambda trace: run_prob_prog(analytic_example, trace=trace), count=1100, eps=0.1, leapfrog_steps=10, burnin=100,), )

result = [tensor.item() for tensor in samples['hmc']['samples']]

estimated_mean = np.mean(result)
estimated_sigma = np.sqrt(np.var(result))

true_mean = z * sigma0**2 / (sigma**2 + sigma0**2) + mu0 * sigma**2 / (sigma**2 + sigma0**2)
true_sigma = np.sqrt(1 / (1 / sigma0**2 + 1 / sigma**2))

