"""This file contains helpers for writing probabilistic programs.

A probabilistic context (ProbCtx) provides the constructs sample() and score().

A simple probabilistic program would look like this

    def simple_prog(ctx: ProbCtx) -> float:
        # sampling:
        sample = ctx.sample(torch.distributions.Normal(0., 1.))
        if sample < 0:
            ctx.score(torch.tensor(0.5))
            return sample.item()
        else:
            return sample.item()

This program describes a standard normal distribution where the left (negative) half is
multiplied/weighted by 0.5.
"""

import math
from typing import Callable, Generic, TypeVar

import torch

T = TypeVar("T")


class ProbCtx:
    """Probabilistic context: keeps track of a probabilistic execution (samples, weight, etc.)"""

    def __init__(self, trace: torch.Tensor) -> None:
        self.idx = 0
        """Index/address of the next sample variable"""
        self.samples = trace.clone().detach()
        """Sampled values so far in the trace"""
        self.samples.requires_grad_(True)
        """The given sample vector"""
        self.is_cont: torch.Tensor = torch.ones(self.samples.shape, dtype=torch.bool)
        """Whether the sampled value is continuous.

        A sample is discontinuous if a branch in the program depends on it."""
        self.log_weight = torch.tensor(0.0, requires_grad=True)
        """Logarithm of the weight.

        The weight of the score()s, but also the pdf for sample()s (deviating from the paper).
        """
        self.log_score = torch.tensor(0.0, requires_grad=True)
        """Logarithm of the score.

        The score is only multiplied for score()"""

        """ The log weight of the trace given """

    def constrain(
        self,
        sample,
        geq: float = None,
        lt: float = None,
    ) -> None:
        """Constrains the sample to be >= geq and < lt.

        This is necessary for random variables whose support isn't all reals.

        Args:
            sample: the sample to be constrained
            geq (float, optional): The lower bound. Defaults to None.
            lt (float, optional): The upper bound. Defaults to None.
        """
        if lt is not None:
            if sample >= lt:
                self.score_log(torch.tensor(-math.inf))
        if geq is not None:
            if sample <= geq:
                self.score_log(torch.tensor(-math.inf))

    def sample(
        self,
        dist: torch.distributions.Distribution,
        is_cont: bool,
    ) -> torch.Tensor:
        """Samples from the given distribution.

        If the distribution has support not on all reals, this needs to be followed by suitable constrain() calls.

        Args:
            dist (torch.distributions.Distribution): the distribution to sample from
            is_cont (bool): whether or not the weight function is continuous in this variable

        Returns:
            the sample
        """
        samples = self.sample_n(1, dist, is_cont)
        return samples[0]

    def sample_n(
        self,
        n: int,
        dist: torch.distributions.Distribution,
        is_cont: bool,
    ) -> torch.Tensor:
        """Samples n times from the given distribution.

        Args:
            n (int): the number of samples
            dist (torch.distributions.Distribution): the distribution to sample from
            is_cont (bool): whether or not the weight function is continuous in this variable

        Returns:
            the samples
        """
        needed = self.idx + n - len(self.samples)
        if needed > 0:
            values = dist.sample((needed,))
            values.requires_grad_(True)
            self.samples = torch.cat((self.samples, values))
            self.is_cont = torch.cat(
                (self.is_cont, torch.ones(needed, dtype=torch.bool))
            )
        for i in range(self.idx, self.idx + n):
            if math.isnan(self.samples[i]):  # TODO: can this loop be removed?
                self.samples[i] = dist.sample(())
        values = self.samples[self.idx : self.idx + n]
        self.is_cont[self.idx : self.idx + n] = torch.tensor(is_cont).repeat(n)
        try:
            self.log_weight = self.log_weight + torch.sum(dist.log_prob(values))
        except ValueError:
            self.log_weight = torch.tensor(-math.inf)
        self.idx += n
        return values

    def score(self, weight: torch.Tensor) -> None:
        """Multiplies the current trace by the given weight.

        Args:
            weight (torch.Tensor): the weight.
        """
        assert torch.is_tensor(weight), "weight is not a tensor"
        self.score_log(torch.log(weight))

    def score_log(self, log_weight: torch.Tensor) -> None:
        assert torch.is_tensor(log_weight), "weight is not a tensor"
        self.log_weight = self.log_weight + log_weight
        self.log_score = self.log_score + log_weight

    def observe(
        self,
        obs: torch.Tensor,
        dist: torch.distributions.Distribution,
    ) -> None:
        self.score_log(dist.log_prob(obs))


class ProbRun(Generic[T]):
    """Result of a probabilistic run"""

    def __init__(self, ctx: ProbCtx, value: T) -> None:
        """Creates a probabilistic run result.

        Undocumented fields are the same as for ProbCtx

        Args:
            ctx (ProbCtx): the probabilistic context used for the program.
            value (T): the return value of the probabilistic program.
        """
        self._gradU: torch.Tensor = None
        """Caches the gradient."""
        self.log_weight = ctx.log_weight
        self.log_score = ctx.log_score
        self.samples = ctx.samples
        self.len = ctx.idx
        """Number of sample statements encountered, i.e. length of the trace."""
        self.is_cont = ctx.is_cont
        self.value = value
        """Returned value of the probabilistic program."""

    def gradU(self) -> torch.Tensor:
        if self._gradU is not None:
            return self._gradU
        U = -self.log_weight
        (self._gradU,) = torch.autograd.grad(U, self.samples, allow_unused=True)
        if self._gradU is None:
            self._gradU = torch.zeros(self.samples.shape)
        return self._gradU

    def used_samples(self) -> torch.Tensor:
        return self.samples[: self.len]


def run_prob_prog(program: Callable[[ProbCtx], T], trace: torch.Tensor) -> ProbRun[T]:
    """Runs the given probabilistic program on the given trace.

    Args:
        program (Callable[[ProbCtx], T]): the probabilistic program.
        trace (torch.Tensor): the trace to replay.

    Returns:
        ProbRun: the result of the probabilistic run.
    """
    tensor_trace = trace
    while True:
        ctx = ProbCtx(tensor_trace)
        ret = None
        try:
            ret = program(ctx)
        except Exception as e:
            if ctx.log_score.item() > -math.inf or ctx.log_weight.item() > -math.inf:
                print("Exception in code with nonzero weight!")
                raise e
            else:
                print("Info: exception in branch with zero weight")
        if ctx.idx > len(tensor_trace):
            tensor_trace = ctx.samples
            continue
        return ProbRun(ctx, ret)
