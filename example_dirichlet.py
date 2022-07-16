import sys

import torch
from torch.distributions import Beta, Normal

import example_gmm
from infer import run_inference, run_inference_icml2022
from ppl import ProbCtx, run_prob_prog


def dp(ctx: ProbCtx, alpha, dims):
    """Stick-breaking method for Dirichlet processes"""
    stick = 1.0
    beta = 0.0
    cumprod = 1.0
    weights = []
    means = []
    while stick > 0.01:
        cumprod *= 1 - beta
        beta_sample = ctx.sample(Beta(1, alpha), is_cont=False)
        ctx.constrain(beta_sample, 0.0, 1.0)
        beta = beta_sample.item()
        theta = ctx.sample_n(dims, standard_normal, is_cont=True)
        weights.append(beta * cumprod)
        means.append(theta)
        stick -= beta * cumprod
    weights_tensor = torch.tensor(weights)
    # Means should be sampled from [0,100].
    # As in the GMM example we use the reparametrization with the standard normal
    # to avoid discontinuities.
    means_tensor = 100 * standard_normal.cdf(torch.stack(means))
    return weights_tensor, means_tensor


def loglikelihoods(weights, means, data):
    (K, d) = means.size()
    N = len(data)
    liks = Normal(means.view(1, K, d).expand(N, K, d), std).log_prob(
        data.view(N, 1, d).expand(N, K, d)
    )
    liks = torch.sum(liks, dim=2) + torch.log(weights).view(1, K).expand(N, K)
    return torch.logsumexp(liks, dim=1)


def loglikelihood(weights, means, data):
    return torch.sum(loglikelihoods(weights, means, data))


standard_normal = Normal(0, 1)
dims = 3
num_training_data = 200
num_test_data = 50
true_alpha = 5.0
true_weights = torch.ones((example_gmm.num_mixtures)) / example_gmm.num_mixtures
true_means = example_gmm.data_means
std = example_gmm.std

training_data = example_gmm.training_data
test_data = example_gmm.test_data


def dp_mixture(ctx: ProbCtx):
    """Dirichlet Process Mixture Model from the paper"""
    weights, means = dp(ctx, true_alpha, dims)
    lik = loglikelihood(weights, means, training_data)
    ctx.score_log(lik)
    return weights.tolist(), means.tolist()


if __name__ == "__main__":
    print(
        f"True training LPPD: {loglikelihood(true_weights, true_means, training_data)}"
    )
    print(f"True test LPPD: {loglikelihood(true_weights, true_means, test_data)}")
    repetitions = 10
    if len(sys.argv) > 1 and sys.argv[1] == "icml2022":
        configs = [
            (L, alpha, K, eps)
            for L in [20]
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
                    lambda trace: run_prob_prog(dp_mixture, trace=trace),
                    name=f"dp_mixture_gmm_{rep}",
                    count=150,
                    burnin=0,  # 50,
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
                lambda trace: run_prob_prog(dp_mixture, trace=trace),
                name=f"dp_mixture_gmm_{rep}",
                count=100,
                burnin=50,
                eps=0.05,
                leapfrog_steps=20,
                seed=rep,
            )
