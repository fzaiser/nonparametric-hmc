import math
import pickle
import time
from typing import Any, Callable, Iterator, List, Tuple

import torch
from torch.distributions import Uniform, Laplace, Normal
from tqdm import tqdm

from ppl import ProbRun, T

torch.manual_seed(0)  # makes executions deterministic
torch.set_printoptions(precision=10)  # more precise printing for debugging


class State:
    """Describes a state in phase space (position q, momentum p) for NP-DHMC

    The field `is_cont` stores which variables are continuous.
    """

    def __init__(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        is_cont: torch.Tensor,
    ) -> None:
        self.q = q
        """position"""
        self.p = p
        """momentum"""
        self.is_cont = is_cont
        """is_cont[i] == True if the density function is continuous in coordinate i.

        If a branch (if-statement) in a program depends on self.q[i], it is discontinuous and is_cont[i] == False."""

    def kinetic_energy(self) -> torch.Tensor:
        """Computes the kinetic energy of the particle.

        In discontinuous HMC, discontinuous coordinates use Laplace momentum, not Gaussian momentum."""
        gaussian = self.p * self.is_cont
        laplace = self.p * ~self.is_cont
        return gaussian.dot(gaussian) / 2 + torch.sum(torch.abs(laplace))


def importance_sample(
    run_prog: Callable[[torch.Tensor], ProbRun[T]],
    count: int = 10_000,
) -> Iterator[Tuple[float, T]]:
    """Samples from a probabilistic program using importance sampling.

    The resulting samples are weighted.

    Note: This is not needed to reproduce the results, but hopefully makes the code easier to understand.

    Args:
        run_prog (Callable[[torch.Tensor], ProbRun[T]]): runs the probabilistic program on a trace.
        count (int, optional): the desired number of samples. Defaults to 10_000.

    Yields:
        Iterator[Tuple[torch.Tensor, T]]: samples of the form (log_score, value)
    """
    for _ in tqdm(range(count)):
        result = run_prog(torch.tensor([]))
        yield result.log_score.item(), result.value


def importance_resample(
    run_prog: Callable[[torch.Tensor], ProbRun[T]],
    count: int = 10_000,
) -> Tuple[List[Tuple[float, T]], List[T]]:
    """Samples from a probabilistic program using importance sampling and systematic resampling.

    It uses systematic resampling on the weighted importance samples to obtain unweighted samples.

    Note: This is not needed to reproduce the results, but hopefully makes the code easier to understand.

    Args:
        run_prog (Callable[[torch.Tensor], ProbRun[T]]): runs the probabilistic program on a trace.
        count (int, optional): the desired number of samples. Defaults to 10_000.

    Returns:
        Tuple[List[Tuple[float, T]], List[T]]: weighted samples, resamples
    """
    weighted_samples = list(importance_sample(run_prog, count))
    count = len(weighted_samples)
    mx = max(log_weight for (log_weight, _) in weighted_samples)
    weight_sum = sum(math.exp(log_weight - mx) for (log_weight, _) in weighted_samples)
    # systematic resampling:
    u_n = Uniform(0, 1).sample().item()
    sum_acc = 0.0
    resamples: List[T] = []
    for (log_weight, value) in weighted_samples:
        weight = math.exp(log_weight - mx) * count / weight_sum
        sum_acc += weight
        while u_n < sum_acc:
            u_n += 1
            resamples.append(value)
    return weighted_samples, resamples


def coord_integrator(
    run_prog: Callable[[torch.Tensor], ProbRun[T]],
    i: int,
    t: float,
    eps: float,
    state: State,
    state_0: State,
    result: ProbRun,
) -> ProbRun[T]:
    """Coordinate integrator adapted from discontinuous HMC.

    For NP-DHMC, it also has to deal with possible changes in dimension."""
    U = -result.log_weight
    q = state.q.clone().detach()
    q[i] += eps * torch.sign(state.p[i])
    new_result = run_prog(q)
    new_U = -new_result.log_weight.item()
    delta_U = new_U - U
    if not math.isfinite(new_U) or torch.abs(state.p[i]) <= delta_U:
        state.p[i] = -state.p[i]
    else:
        state.p[i] -= torch.sign(state.p[i]) * delta_U
        N2 = new_result.len
        N = result.len
        result = new_result
        if N2 > N:
            # extend everything to the higher dimension
            state.q = result.samples.clone().detach()
            is_cont = result.is_cont.clone().detach()
            # pad the momentum vector:
            gauss = Normal(0, 1).sample([N2 - N])
            laplace = Laplace(0, 1).sample([N2 - N])
            p_padding = gauss * is_cont[N:N2] + laplace * ~is_cont[N:N2]
            state_0.p = torch.cat((state_0.p, p_padding))
            state_0.is_cont = torch.cat((state_0.is_cont, is_cont[N:N2]))
            state.p = torch.cat((state.p, p_padding))
            state.is_cont = is_cont
            # adjust the position vector:
            q0_padding = (
                state.q[N:N2].clone().detach()
                - t * state.p[N:N2] * is_cont[N:N2]
                - t * torch.sign(state.p[N:N2]) * ~is_cont[N:N2]
            )
            state_0.q = torch.cat((state_0.q, q0_padding))
        else:
            # truncate everything to the lower dimension
            state.q = result.samples[:N2].clone().detach()
            state.p = state.p[:N2]
            state.is_cont = result.is_cont[:N2]
            state_0.q = state_0.q[:N2]
            state_0.p = state_0.p[:N2]
            state_0.is_cont = state_0.is_cont[:N2]
        assert len(state.p) == len(state_0.p)
        assert len(state.p) == len(state.q)
        assert len(state.is_cont) == len(state.p)
        assert len(state_0.is_cont) == len(state_0.p)
        assert len(state_0.p) == len(state_0.q)
    return result


def integrator_step(
    run_prog: Callable[[torch.Tensor], ProbRun[T]],
    t: float,
    eps: float,
    state: State,
    state_0: State,
) -> ProbRun[T]:
    """Performs one integrator step (called "leapfrog step" in standard HMC)."""
    result = run_prog(state.q)
    # first half of leapfrog step for continuous variables:
    state.p = state.p - eps / 2 * result.gradU() * state.is_cont
    state.q = state.q + eps / 2 * state.p * state.is_cont
    result = run_prog(state.q)
    # Integrate the discontinuous coordinates in a random order:
    disc_indices = torch.flatten(torch.nonzero(~state.is_cont, as_tuple=False))
    perm = torch.randperm(len(disc_indices))
    disc_indices_permuted = disc_indices[perm]
    for j in disc_indices_permuted:
        if j >= len(state.q):
            continue  # out-of-bounds can happen if q changes length during the loop
        result = coord_integrator(
            run_prog, int(j.item()), t, eps, state, state_0, result
        )
    # second half of leapfrog step for continuous variables
    state.q = state.q + eps / 2 * state.p * state.is_cont
    result = run_prog(state.q)
    state.p = state.p - eps / 2 * result.gradU() * state.is_cont
    return result


def np_dhmc(
    run_prog: Callable[[torch.Tensor], ProbRun[T]],
    count: int,
    leapfrog_steps: int,
    eps: float,
    burnin: int = None,
) -> List[T]:
    """Samples from a probabilistic program using NP-DHMC.

    Args:
        run_prog (Callable[[torch.Tensor], ProbRun[T]]): runs the probabilistic program on a trace.
        count (int, optional): the desired number of samples. Defaults to 10_000.
        burnin (int): number of samples to discard at the start. Defaults to `count // 10`.
        leapfrog_steps (int): number of leapfrog steps the integrator performs.
        eps (float): the step size of the leapfrog steps.

    Returns:
        List[T]: list of samples
    """
    if burnin is None:
        burnin = count // 10
    final_samples = []
    result = run_prog(torch.tensor([]))
    U = -result.log_weight
    q = result.samples.clone().detach()
    is_cont = result.is_cont.clone().detach()
    count += burnin
    accept_count = 0
    for _ in tqdm(range(count)):
        N = len(q)
        dt = ((torch.rand(()) + 0.5) * eps).item()
        gaussian = Normal(0, 1).sample([N]) * is_cont
        laplace = Laplace(0, 1).sample([N]) * ~is_cont
        p = gaussian + laplace
        state_0 = State(q, p, is_cont)
        state = State(q, p, is_cont)
        prev_res = result
        for step in range(leapfrog_steps):
            if not math.isfinite(result.log_weight.item()):
                break
            result = integrator_step(run_prog, step * dt, dt, state, state_0)
        # Note that the acceptance probability differs from the paper in the following way:
        # In the implementation, we can have other continuous distributions than
        # just normal distributions.
        # We treat `x = sample(D)` as `x = sample(normal); score(pdf_D(x) / pdf_normal(x));`.
        # this means that the `w(q) * pdfnormal(q)` in the acceptance probability just becomes
        # `w(q) * pdf_D(q) = weight(q)`
        # because the weight in the implementation includes the prior.
        # (`w` refers to the weight as defined in the paper and
        # `weight` to the weight as used in the implementation.)
        # Similarly
        #   w(q0 after extension) * pdfnormal(q0 after extension)
        # = w(q0 before extension) * pdfnormal(q0 before extension) * pdfnormal(extended part of q0)
        # = weight(q0 before extensions) * pdfnormal(extended part of q0)
        # For this reason, we add the factor pdfnormal(extended part of q0) in U_0 below.
        K_0 = state_0.kinetic_energy()
        N2 = len(state_0.q)
        U_0 = -prev_res.log_weight - Normal(0, 1).log_prob(state_0.q[N:N2]).sum()
        K = state.kinetic_energy()
        U = -result.log_weight
        accept_prob = torch.exp(U_0 + K_0 - U - K)
        if U.item() != math.inf and torch.rand(()) < accept_prob:
            q = state.q
            is_cont = state.is_cont
            accept_count += 1
            final_samples.append(result.value)
        else:
            result = prev_res
            final_samples.append(prev_res.value)
    count = len(final_samples)
    final_samples = final_samples[burnin:]  # discard first samples (burn-in)
    print(f"acceptance ratio: {accept_count / count * 100}%")
    return final_samples


def np_lookahead_dhmc(
    run_prog: Callable[[torch.Tensor], ProbRun[T]],
    count: int,
    L: int,
    eps: float,
    K: int = 0,
    alpha: float = 1,
    burnin: int = None,
) -> Tuple[List[T], Any]:
    """Samples from a probabilistic program using "Lookahead" NP-DHMC.

    Returns a list of samples and additional information, together as a pair.

    The acceptance condition is taken from [1] (Figure 3).
    They prove that it's equivalent to Sohl-Dickstein et al.'s version in Appendix C.

    [1] Campos, Sanz-Serna: Extra Chance Generalized Hybrid Monte Carlo (https://arxiv.org/pdf/1407.8107.pdf)

    Args:
        run_prog (Callable[[torch.Tensor], ProbRun[T]]): runs the probabilistic program on a trace.
        count (int, optional): the desired number of samples. Defaults to 10_000.
        L (int): number of leapfrog steps the integrator performs.
        eps (float): the step size of the leapfrog steps.
        K (int): number of "extra chances" for Lookahead HMC (0: standard HMC)
        alpha (float): persistence factor (0: full persistence, 1: stanard HMC)
        burnin (int): number of samples to discard at the start. Defaults to `count // 10`.

    Returns:
        List[T], Any: list of samples, stats
    """
    if burnin is None:
        burnin = count // 10
    final_samples = []
    lookahead_stats = [0] * (K + 2)
    result = run_prog(torch.tensor([]))
    q = result.samples.clone().detach()
    N = len(q)
    is_cont = result.is_cont.clone().detach()
    gaussian = Normal(0, 1).sample([N]) * is_cont
    laplace = Laplace(0, 1).sample([N]) * ~is_cont
    p = gaussian + laplace
    count += burnin
    accept_count = 0
    for _ in tqdm(range(count)):
        N = len(q)
        dt = ((torch.rand(()) + 0.5) * eps).item()
        p_cont = p * math.sqrt(1 - alpha * alpha) + Normal(0, alpha).sample([N])
        p_disc = p * math.sqrt(1 - alpha * alpha) + Laplace(0, alpha).sample([N])
        p = p_cont * is_cont + p_disc * ~is_cont
        state_0 = State(q, p, is_cont)
        state = State(q, p, is_cont)
        prev_res = result
        rand_uniform = torch.rand(())
        for k in range(K + 1):
            for step in range(L):
                if not math.isfinite(result.log_weight.item()):
                    break
                result = integrator_step(run_prog, step * dt, dt, state, state_0)
            # Note that the acceptance probability differs from the paper in the following way:
            # In the implementation, we can have other continuous distributions than
            # just normal distributions.
            # We treat `x = sample(D)` as `x = sample(normal); score(pdf_D(x) / pdf_normal(x));`.
            # this means that the `w(q) * pdfnormal(q)` in the acceptance probability just becomes
            # `w(q) * pdf_D(q) = weight(q)`
            # because the weight in the implementation includes the prior.
            # (`w` refers to the weight as defined in the paper and
            # `weight` to the weight as used in the implementation.)
            # Similarly
            #   w(q0 after extension) * pdfnormal(q0 after extension)
            # = w(q0 before extension) * pdfnormal(q0 before extension) * pdfnormal(extended part of q0)
            # = weight(q0 before extensions) * pdfnormal(extended part of q0)
            # For this reason, we add the factor pdfnormal(extended part of q0) in U_0 below.
            K_0 = state_0.kinetic_energy()
            N2 = len(state_0.q)
            U_0 = -prev_res.log_weight - Normal(0, 1).log_prob(state_0.q[N:N2]).sum()
            K_new = state.kinetic_energy()
            U_new = -result.log_weight
            accept_prob = torch.exp(U_0 + K_0 - U_new - K_new)
            if U_new.item() != math.inf and rand_uniform < accept_prob:
                q = state.q
                p = state.p
                is_cont = state.is_cont
                accept_count += 1
                final_samples.append(result.value)
                lookahead_stats[k + 1] += 1
                break
        else:  # if we didn't accept the loop and exit before
            result = prev_res
            p = -p  # momentum flip
            final_samples.append(prev_res.value)
            lookahead_stats[0] += 1
    count = len(final_samples)
    final_samples = final_samples[burnin:]  # discard first samples (burn-in)
    print(f"acceptance ratio: {accept_count / count * 100}%")
    print(f"lookahead stats: {lookahead_stats}")
    return final_samples, lookahead_stats


def run_inference(
    run_prog: Callable[[torch.Tensor], ProbRun[T]],
    name: str,
    count: int,
    eps: float,
    leapfrog_steps: int,
    burnin: int = None,
    seed: int = None,
    **kwargs,
) -> dict:
    """Runs importance sampling and NP-DHMC, then saves the samples to a .pickle file.

    The file is located in the `samples_produced/` folder.

    Note: This is not needed to reproduce the results, but hopefully makes the code easier to understand.
    """

    def run(sampler: Callable) -> dict:
        if seed is not None:
            torch.manual_seed(seed)
        start = time.time()
        results = sampler()
        stop = time.time()
        elapsed = stop - start
        return {
            "time": elapsed,
            "samples": results,
        }

    adjusted_count = count * leapfrog_steps
    samples = {}
    print("Running NP-DHMC...")
    samples["hmc"] = run(
        lambda: np_dhmc(
            run_prog,
            count=count,
            eps=eps,
            leapfrog_steps=leapfrog_steps,
            burnin=burnin,
            **kwargs,
        ),
    )
    samples["hmc"]["burnin"] = burnin
    samples["hmc"]["eps"] = eps
    samples["hmc"]["leapfrog_steps"] = leapfrog_steps
    print("Running importance sampling...")
    samples["is"] = run(
        lambda: importance_resample(run_prog, count=adjusted_count),
    )
    weighted, values = samples["is"]["samples"]
    samples["is"]["samples"] = values
    samples["is"]["weighted"] = weighted

    filename = f"{name}__count{count}_eps{eps}_leapfrogsteps{leapfrog_steps}"
    samples["filename"] = filename
    with open(f"samples_produced/{filename}.pickle", "wb") as f:
        pickle.dump(samples, f)
    return samples


def run_inference_icml2022(
    run_prog: Callable[[torch.Tensor], ProbRun[T]],
    name: str,
    count: int,
    eps: float,
    L: int,
    K: int = 0,
    alpha: float = 1,
    burnin: int = None,
    seed: int = None,
    **kwargs,
) -> dict:
    """Runs NP-LA-DHMC with persistence, then saves the samples to a .pickle file.

    The file is located in the `samples_produced/` folder.

    Note: This is not needed to reproduce the results, but hopefully makes the code easier to understand.
    """

    def run(sampler: Callable) -> dict:
        if seed is not None:
            torch.manual_seed(seed)
        start = time.time()
        results, stats = sampler()
        stop = time.time()
        elapsed = stop - start
        return {
            "time": elapsed,
            "samples": results,
            "stats": stats,
        }

    # adjusted_count = count * L
    samples = {}
    # NP-LA-DHMC with persistence
    # print(f"Running NP-Lookahead-DHMC with (L = {L}, α = {alpha}, K = {K})...")
    persistentstr = "" if alpha == 1.0 else "-persistent"
    lastr = "" if K == 0 else "la"
    method = f"np{lastr}dhmc{persistentstr}"
    samples[method] = run(
        lambda: np_lookahead_dhmc(
            run_prog,
            count=count,
            eps=eps,
            L=L,
            K=K,
            burnin=burnin,
            alpha=alpha,
            **kwargs,
        )
    )
    samples[method]["burnin"] = burnin
    samples[method]["eps"] = eps
    samples[method]["L"] = L
    if alpha != 1.0:
        samples[method]["alpha"] = alpha
    if K != 0:
        samples[method]["K"] = K
    filename = f"{name}__count{count}_eps{eps}_L{L}_alpha{alpha}_K{K}"
    with open(f"lookahead_samples/{filename}.pickle", "wb") as f:
        pickle.dump(samples, f)
    return samples
