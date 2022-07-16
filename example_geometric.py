import sys
from torch.distributions import Uniform

from infer import run_inference, run_inference_icml2022
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
    if len(sys.argv) > 1 and sys.argv[1] == "icml2022":
        configs = [
            (L, alpha, K, eps)
            for L in [5, 2]
            for eps in [0.1]
            for alpha in [1.0, 0.5, 0.1]
            for K in [0, 1, 2]
        ]
        for rep in range(repetitions):
            for L, alpha, K, eps in configs:
                print(
                    f"REPETITION {rep+1}/{repetitions} ({eps=}, {L=}, {alpha=}, {K=})"
                )
                run_inference_icml2022(
                    lambda trace: run_prob_prog(geometric, trace=trace),
                    name=f"geometric_{rep}",
                    count=count,
                    burnin=0,  # 100,
                    L=L,
                    eps=eps,
                    K=K,
                    alpha=alpha,
                    seed=rep,
                )
    else:
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
