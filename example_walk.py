import pickle
import sys
import torch

import pyro
import pyro.infer.mcmc as pyromcmc  # type: ignore
from torch.distributions import Normal, Uniform

from infer import run_inference, importance_resample
from ppl import ProbCtx, run_prob_prog

distance_limit = 10


def walk_model(ctx: ProbCtx) -> float:
    """Random walk model.

    Mak et al. (2020): Densities of almost-surely terminating probabilistic programs are differentiable almost everywhere.
    """
    distance = torch.tensor(0.0, requires_grad=True)
    start = ctx.sample(Uniform(0, 3), is_cont=False)
    position = start
    while position > 0 and distance < distance_limit:
        step = ctx.sample(Uniform(-1, 1), is_cont=False)
        distance = distance + torch.abs(step)
        position = position + step
    ctx.observe(distance, Normal(1.1, 0.1))
    return start.item()


def pyro_walk_model() -> float:
    """The same model written in Pyro."""
    start = pyro.sample("start", pyro.distributions.Uniform(0, 3))
    t = 0
    position = start
    distance = torch.tensor(0.0)
    while position > 0 and position < distance_limit:
        step = pyro.sample(f"step_{t}", pyro.distributions.Uniform(-1, 1))
        distance = distance + torch.abs(step)
        position = position + step
        t = t + 1
    pyro.sample("obs", pyro.distributions.Normal(1.1, 1.0), obs=distance)
    return start.item()


def run_pyro(use_nuts, rep, count, eps, num_steps):
    """Runs Pyro HMC and NUTS."""
    if use_nuts:
        info = f"nuts{rep}_count{count}"
        kernel = pyromcmc.NUTS(pyro_walk_model)
    else:
        info = f"hmc{rep}_count{count}_eps{eps}_steps{num_steps}"
        kernel = pyromcmc.HMC(
            pyro_walk_model,
            step_size=eps,
            num_steps=num_steps,
            adapt_step_size=False,
        )
    mcmc = pyromcmc.MCMC(kernel, num_samples=count, warmup_steps=count // 10)
    mcmc.run()
    samples = mcmc.get_samples()
    raw_samples = [value.item() for value in samples["start"]]
    with open(f"samples_produced/walk_model{rep}_pyro_{info}.pickle", "wb") as f:
        pickle.dump(raw_samples, f)
    mcmc.summary()


if __name__ == "__main__":
    count = 1_000
    repetitions = 10
    eps = 0.1
    num_steps = 50
    if len(sys.argv) > 1 and sys.argv[1] == "pyro":
        for use_nuts in [False, True]:
            if use_nuts:
                print("Running Pyro NUTS...")
            else:
                print("Running Pyro HMC...")
            for rep in range(repetitions):
                print(f"REPETITION {rep+1}/{repetitions}")
                run_pyro(use_nuts, rep, count, eps, num_steps)
        sys.exit()
    for rep in range(repetitions):
        print(f"REPETITION {rep+1}/{repetitions}")
        run_inference(
            lambda trace: run_prob_prog(walk_model, trace=trace),
            name=f"walk_model{rep}",
            count=count,
            burnin=100,
            eps=eps,
            leapfrog_steps=num_steps,
            seed=rep,
        )
    print("Generating importance samples as ground truth...")
    ground_truth_count = 1000 * count * repetitions
    torch.manual_seed(0)
    weighted, samples = importance_resample(
        (lambda trace: run_prob_prog(walk_model, trace)),
        count=ground_truth_count,
    )
    with open(f"samples_produced/walk_is_{ground_truth_count}.pickle", "wb") as f:
        pickle.dump((weighted, samples), f)
