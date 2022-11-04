Nonparametric HMC implementation
================================

This repository contains the implementation of the *Nonparametric Discontinuous Hamiltonian Monte Carlo (NP-DHMC)* algorithm, as described in

> Carol Mak, Fabian Zaiser, Luke Ong. *Nonparametric Hamiltonian Monte Carlo.* ICML 2021. [(proceedings)](https://proceedings.mlr.press/v139/mak21a.html) [(updated arxiv)](https://arxiv.org/abs/2106.10238)

It also contains the implementation of *Nonparametric Lookahead Discontinuous Hamilton Monte Carlo (NP-Lookahead-DHMC)*, as described in

> Carol Mak, Fabian Zaiser, Luke Ong. *Nonparametric Involutive Markove Chain Monte Carlo.* ICML 2022. [(proceedings)](https://proceedings.mlr.press/v162/mak22a.html) [(updated arxiv)](https://arxiv.org/abs/2211.01100)

Modifications since publication
-------------------------------

* We have been made aware of a small bug in the generation of Pyro's samples.
  (Thanks to Zirui Zhao for finding this and letting us know!)
  We have fixed the bug and the samples obtained using Pyro's HMC and NUTS samplers are still wrong.
  While the resulting plot looks somewhat different, this does not affect the conclusions of the paper.
* In the original ICML 2021 code submission, the acceptance probability differed from the pseudocode in the paper. In addition, the extended trace was not updated to the correct time. Both issues are now resolved and the fixes led to a small performance improvement.

Setup
-----

You need an installation of Python 3, Jupyter and the following packages (I used the pip package manager)

    $ pip3 install matplotlib numpy scipy pandas torch seaborn tqdm pyro-ppl numpyro

We carried the experiments out on a computer with a Intel Core i7-8700 CPU, @ 3.20 GHz x 12 and 16 GB RAM, running Ubuntu 20.04.
The exact versions of the installed packages shouldn't matter.
For reference, they were: `matplotlib-3.3.1`, `numpy-1.19.1`, `scipy-1.5.2`, `pandas-1.1.2`, `torch-1.6.0`, `seaborn-0.11.0`, `tqdm-4.48.2`, `pyro-ppl-1.4.0`, and `numpyro-0.6.0`.

Reproducing the experiments
---------------------------

To reproduce the experiments from the paper, you need to generate the samples:

1. Generate samples using Nonparametric Hamiltonian Monte Carlo (NP-DHMC).
2. Generate samples using Anglican (for comparison).
3. Run the evaluation scripts to produce the plots and tables from the paper.

If you just want to view the results of the evaluation, you can simply view the `evaluation_*.ipynb` notebooks from step 3.
But if you want to run the evaluation yourself, you'll need to complete steps 1 and 2 first.

### Step 1: Generating the NP-DHMC samples

The model code for the 4 examples is in the `example_*.py` files.

*ICML 2021:* You can simply run the experiments from our ICML 2021 paper as follows:

    $ python3 example_geometric.py
    $ python3 example_walk.py
    $ python3 example_walk.py pyro-hmc            # to run Pyro's HMC sampler on this model
    $ python3 example_walk.py pyro-nuts           # to run Pyro's NUTS sampler on this model
    $ python3 example_gmm.py
    $ python3 example_dirichlet.py

These runs will save their results in `samples_produced/{experiment_name}_{run}_{sample_count}_{hyperparameters}.pickle`.

Note that some of those runs can take a while to complete.
Especially generating the ground truth in `example_walk.py` takes several hours and Pyro is also very slow in this example.
Overall, the runs took over a day to complete.

*ICML 2022:* For the experiments from our ICML 2022 paper, run the following commands:

    $ python3 example_geometric.py icml2022
    $ python3 example_walk.py icml2022
    $ python3 example_gmm.py icml2022
    $ python3 example_dirichlet.py icml2022

These runs will save their results in `lookahead_samples/{experiment_name}_{run}_{sample_count}_{hyperparameters}.pickle`.

### Step 2: Generating the Anglican samples

Note: You can run the evaluation scripts without Anglican by setting `anglican_methods = []` in `evaluation.py`.
This will still plot the NP-DHMC samples, but won't do a comparison between Anglican and NP-DHMC.

1. To produce the Anglican samples for comparison, follow the instructions in `anglican/README.md`.
The next steps below make the results available to the evaluation script.
2. Copy the contents of `anglican/samples/` into the folder `anglican_samples/`.

        $ cp -r anglican/samples/ anglican_samples/

3. Copy the output of each of the four experiments in Anglican into `anglican_samples/{experiment_name}/timing.txt`. Concretely, the Anglican output will look like this:

        10 runs of geometric with 5000 samples and 500 burn-in
        [...]
        10 runs of random walk with 50000 samples and 5000 burn-in
        [...]
        10 runs of gmm with 50000 samples and 5000 burn-in
        [...]
        10 runs of dpmm with 2000 samples and 1000 burn-in
        [...]

    The first `[...]` needs to go into `anglican_samples/geo/timing.txt` and similarly for the others.

*Note:* Anglican does not seem to provide a way to set the random seed.
Therefore, each execution can be different, which may explain slight discrepancies with the numbers reported in the paper.

### Step 3: Evaluation

The evaluation (ESS, LPPD, plots) for the experiments is in the Jupyter notebooks `evaluation_*.ipynb` files.
If you use Jupyter, you can see the saved results and don't have to run the code.
To view the notebooks, run the following:

    $ jupyter notebook evaluation_geometric.ipynb
    $ jupyter notebook evaluation_walk.ipynb
    $ jupyter notebook evaluation_gmm.ipynb
    $ jupyter notebook evaluation_dirichlet.ipynb

For the ICML 2022 experiments, view the following notebooks:

    $ jupyter notebook evaluation_geometric_icml2022.ipynb
    $ jupyter notebook evaluation_walk_icml2022.ipynb
    $ jupyter notebook evaluation_gmm_icml2022.ipynb
    $ jupyter notebook evaluation_dirichlet_icml2022.ipynb

*Note:* If you not only want to view the notebooks but also run the code, you'll need to produce the samples for all the inference algorithms first.

When running the notebooks, if you are prompted to select a Python kernel, select *Python 3*.

The above notebooks will output the data reported in our paper.
Additionally, they produce our plots and if you execute the notebooks, the plots are saved as PDF files in the current directory.

Project architecture
--------------------

As a help to find your way around the codebase, here is a brief description of each file:

- `ppl.py` includes the probabilistic programming primitives as methods of the `ProbCtx` (probabilistic context) class
- `infer.py` implements NP-DHMC (and importance sampling for illustration) used in our experiments.
- `evaluation.py` contains utilities needed for the evaluation of the results from ICML 2021.
- `evaluation_icml2022.py` contains utilities needed for the evaluation of the results for the ICMl 2022 experiments.
- `example_*.py` are for running the experiments from the paper.
- `evaluation_*.ipynb` are for evaluating the results of each experiment from ICML 2021.
- `evaluation_*_icml2022.ipynb` are for evaluating the results of each experiment from ICML 2022.
