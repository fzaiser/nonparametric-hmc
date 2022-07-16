import math
import sys

import torch
from torch.distributions import Normal, Poisson, Uniform

from infer import run_inference, run_inference_icml2022
from ppl import ProbCtx, run_prob_prog


def loglikelihoods(means, data):
    (K, d) = means.size()
    N = len(data)
    liks = Normal(means.view(1, K, d).expand(N, K, d), std).log_prob(
        data.view(N, 1, d).expand(N, K, d)
    )
    liks = torch.sum(liks, dim=2)
    return torch.logsumexp(liks, dim=1) - torch.log(torch.tensor(float(K)))


def loglikelihood(means, data):
    return torch.sum(loglikelihoods(means, data))


# Data for the following more complex GMM example
torch.manual_seed(0)
num_data = 200
num_mixtures = 9
dims = 3
data_means = Uniform(0.0, 100.0).sample((num_mixtures, dims))
"""This random draw gives this result on torch 1.6.0:

data_means = torch.tensor(
    [
        [49.6256599426, 76.8221817017, 8.8477430344],
        [13.2030487061, 30.7422809601, 63.4078674316],
        [49.0093421936, 89.6444702148, 45.5627975464],
        [63.2306289673, 34.8893470764, 40.1717300415],
        [2.2325754166, 16.8858947754, 29.3888454437],
        [51.8521804810, 69.7667617798, 80.0011367798],
        [16.1029453278, 28.2268581390, 68.1608581543],
        [91.5193939209, 39.7099914551, 87.4155883789],
        [41.9408340454, 55.2907066345, 95.2738113403],
    ]
)
"""
std = 10.0


def sample_prior(num_samples):
    return torch.stack(
        [
            Normal(
                data_means[math.floor(torch.rand(()).item() * num_mixtures)],
                torch.tensor(std),
            ).sample(())
            for n in range(num_samples)
        ]
    )


training_data = sample_prior(num_data)
num_test_data = 50
test_data = sample_prior(num_test_data)
standard_normal = Normal(0, 1)


def gmm(ctx: ProbCtx):
    """Gaussian Mixture Model from the paper"""
    poisson = ctx.sample(dist=Poisson(10), is_cont=False)
    ctx.constrain(poisson, geq=0)
    K = math.floor(poisson.item()) + 1
    # The following statement samples from Uniform(0,1) in a continuous way.
    # Sampling directly from Uniform(0,1) would lead to discontinuities at 0 and 1.
    # Instead, we sample from the standard normal and apply its CDF.
    # This is a standard reparametrization.
    means = standard_normal.cdf(ctx.sample_n(K * dims, standard_normal, is_cont=True))
    means = means.reshape(K, dims) * 100
    lik = loglikelihood(means, training_data)
    ctx.score_log(lik)
    return means.tolist()


if __name__ == "__main__":
    print(f"True training LPPD: {loglikelihood(data_means, training_data)}")
    print(f"True test LPPD: {loglikelihood(data_means, test_data)}")
    repetitions = 10
    count = 1_000
    if len(sys.argv) > 1 and sys.argv[1] == "icml2022":
        configs = [
            (L, alpha, K, eps)
            for L in [25]
            for eps in [0.05]
            for alpha in [1.0, 0.5]
            for K in [0, 1, 2]
        ]
        for rep in range(repetitions):
            for L, alpha, K, eps in configs:
                print(
                    f"REPETITION {rep+1}/{repetitions} ({eps=}, {L=}, {alpha=}, {K=})"
                )
                run_inference_icml2022(
                    lambda trace: run_prob_prog(gmm, trace=trace),
                    name=f"gmm_{rep}",
                    count=count,
                    burnin=0,  # 100,
                    eps=eps,
                    L=L,
                    K=K,
                    alpha=alpha,
                    seed=rep,
                )
    else:
        for rep in range(repetitions):
            print(f"REPETITION {rep+1}/{repetitions}")
            run_inference(
                lambda trace: run_prob_prog(gmm, trace=trace),
                count=count,
                burnin=100,
                eps=0.05,
                leapfrog_steps=50,
                name=f"gmm_{rep}",
                seed=rep,
            )
