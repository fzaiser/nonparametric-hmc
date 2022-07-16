from collections import defaultdict
from copy import deepcopy

import matplotlib
import seaborn as sns

# Use type 42 (TrueType) fonts instead of the default Type 3 fonts
matplotlib.rcParams["pdf.fonttype"] = 42

# Make plots more accessible:
sns.set_palette("colorblind")

anglican_methods = []
# all_methods = ["npdhmc", "npdhmc-persistent", "is"]
# compared_methods = ["npdhmc", "npdhmc-persistent", "is"]

method_name = {
    "npdhmc": "NP-DHMC",
    "is": "IS",
    "npdhmc-persistent": "NP-DHMC pers.",
    "np-la-dhmc": "NP-Lookahead-DHMC",
    "npladhmc": "NP-Lookahead-DHMC",
    "npladhmc-persistent": "NP-Lookahead-DHMC pers.",
}


def legend_str(config) -> str:
    if type(config) == tuple:
        assert len(config) == 2
        if config[1]:
            return f"{method_name[config[0]]} ({config[1]})"
        else:
            return f"{method_name[config[0]]}"
    elif config:
        return str(config)
    else:
        return ""


def thin_list(l: list, target_size: int) -> list:
    size = len(l)
    assert size >= target_size
    result = []
    for i in range(target_size):
        result.append(l[i * size // target_size])
    return result


def thin_runs(runs: list, burnin: int = 0) -> list:
    thinned_runs = []
    for run in runs:
        thinned_runs.append(defaultdict(list))
        N = min(len(r["samples"][burnin:]) for r in run.values())
        for method in run.keys():
            thinned_runs[-1][method] = thin_list(run[method]["samples"][burnin:], N)
    return thinned_runs


def toconfigstr(L, alpha, K):
    alphastr = [] if alpha == 1.0 else [f"α={alpha}"]
    Kstr = [] if K == 0 else [f"K={K}"]
    if L is not None:
        return ", ".join([f"L={L}"] + alphastr + Kstr)
    else:
        return ", ".join(alphastr + Kstr)


def collect_values(thinned_runs: list, config=None) -> dict:
    values = defaultdict(list)
    for run in thinned_runs:
        for method in run.keys():
            if config is None:
                values[method] += run[method]
            else:
                values[(method, config)] += run[method]
    return values


def collect_chains(thinned_runs: list, config=None) -> dict:
    chains = defaultdict(list)
    for run in thinned_runs:
        for method in run.keys():
            if config is None:
                chains[method].append(run[method])
            else:
                chains[(method, config)].append(run[method])
    return chains


def print_running_time(runs: list, thinned_runs: list):
    for method in runs[0].keys():
        running_time = sum(run[method]["time"] for run in runs)
        count = sum(len(run[method]) for run in thinned_runs)
        per_sample = running_time / count
        print(
            f"{legend_str(method)}: {running_time:.2f}s    {per_sample:.4f}s per sample (after thinning)"
        )


def compute_iteration_count(L: int, K: int, lookahead_stats: list) -> int:
    lookahead_iterations = lookahead_stats[0] * (K + 1) + sum(
        lookahead_stats[i] * i for i in range(1, K + 2)
    )
    return L * lookahead_iterations


def adjust_for_iteration_time(experiments: dict) -> dict:
    effort = defaultdict(list)
    for config in experiments.keys():
        for run in experiments[config]:
            for method, data in run.items():
                L = data["L"]
                K = data.get("K", 0)
                stats = data["stats"]
                num_iterations = compute_iteration_count(L, K, stats)
                data["effort"] = num_iterations
                effort[method, config].append(num_iterations)
    min_effort = min(min(e) for e in effort.values())
    print(f"{effort=}")
    reduction_factor = {k: [x / min_effort for x in v] for k, v in effort.items()}
    print(f"{reduction_factor=}")
    adjusted_experiments = deepcopy(experiments)
    min_len = None
    for key, runs in experiments.items():
        for i, run in enumerate(runs):
            for method, data in run.items():
                cutoff = int(len(data["samples"]) / reduction_factor[method, key][i])
                if min_len is None or min_len > cutoff:
                    min_len = cutoff
                adjusted_experiments[key][i][method]["samples"] = data["samples"][
                    :cutoff
                ]
                adjusted_experiments[key][i][method]["time"] /= reduction_factor[
                    method, key
                ][i]
    print(f"{min_len=}")
    return adjusted_experiments
