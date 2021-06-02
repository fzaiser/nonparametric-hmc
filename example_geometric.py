from torch.distributions import Uniform

from infer import run_inference
from ppl import ProbCtx, run_prob_prog


def geometric(ctx: ProbCtx) -> int:
    """Describes a geometric distribution"""
    sample = ctx.sample(Uniform(0.0, 1.0), is_cont=False)
    ctx.constrain(sample, 0.0, 1.0)
    if sample < 0.2:
        return 1
    else:
        return 1 + geometric(ctx)


if __name__ == "__main__":
    count = 1_000
    repetitions = 10
    for rep in range(repetitions):
        print(f"REPETITION {rep+1}/{repetitions}")
        run_inference(
            lambda trace: run_prob_prog(geometric, trace=trace),
            name=f"geometric{rep}",
            count=count,
            burnin=100,
            leapfrog_steps=5,
            eps=0.1,
            seed=rep,
        )
