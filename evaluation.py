import matplotlib

# Use type 42 (TrueType) fonts instead of the default Type 3 fonts
matplotlib.rcParams['pdf.fonttype'] = 42

palette = {
    "ours": "C0",
    "LMH": "C2",
    "PGibbs": "C1",
    "RMH": "C3",
    "IPMCMC": "C5",
    "ground truth": "C4",
    "Pyro HMC": "C8",
    "Pyro NUTS": "C9",
}

anglican_methods = ["lmh", "pgibbs", "rmh", "ipmcmc"]  # replace with [] to disable
all_methods = ["hmc", "is"] + anglican_methods
compared_methods = ["hmc", "lmh", "pgibbs", "rmh"] if anglican_methods else ["hmc"]

method_name = {
    "hmc": "ours",
    "is": "IS",
    "lmh": "LMH",
    "ipmcmc": "IPMCMC",
    "pgibbs": "PGibbs",
    "rmh": "RMH",
}


def parse_anglican_timings(filename: str) -> dict:
    timings = {}
    with open(filename) as f:
        method = None
        while True:
            line = f.readline().strip()
            if line == "":
                break
            words = line.split()
            if len(words) == 1:
                method = line
                timings[method] = []
                continue
            if words[1] == "msecs":
                timings[method].append(float(words[0]) / 1000)
    return timings


def thin_list(l: list, target_size: int) -> list:
    size = len(l)
    assert size >= target_size
    result = []
    for i in range(target_size):
        result.append(l[i * size // target_size])
    return result


def thin_runs(all_methods: list, runs: list) -> list:
    thinned_runs = []
    for run in runs:
        thinned_runs.append({})
        N = len(run["hmc"]["samples"])
        for method in all_methods:
            thinned_runs[-1][method] = thin_list(run[method]["samples"], N)
    return thinned_runs


def collect_values(all_methods: list, thinned_runs: list) -> dict:
    values = {m: [] for m in all_methods}
    for run in thinned_runs:
        N = len(run["hmc"])
        for method in all_methods:
            values[method] += run[method]
    return values


def collect_chains(all_methods: list, thinned_runs: list) -> dict:
    chains = {m: [] for m in all_methods}
    for run in thinned_runs:
        for method in all_methods:
            chains[method].append(run[method])
    return chains


def print_running_time(all_methods: list, runs: list, thinned_runs: list):
    print("\nRunning times:")
    for method in all_methods:
        running_time = sum(run[method]["time"] for run in runs)
        count = sum(len(run[method]) for run in thinned_runs)
        per_sample = running_time / count
        print(
            f"{method}: {running_time:.2f}s    {per_sample:.4f}s per sample (after thinning)"
        )
