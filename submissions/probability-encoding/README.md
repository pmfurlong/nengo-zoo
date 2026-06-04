# probability-encoding

A spiking-neural population that estimates a probability density from samples, using Spatial Semantic Pointers (SSPs) and a Fourier Integral Estimator (FIE) normalization.

## Description

Given samples drawn from a 1-D distribution, the encoder computes a Spatial Semantic Pointer representation `mu` of the empirical distribution and a normalization constant `xi`. It then builds a spiking neural ensemble whose firing rate at the SSP-encoded query point `x` approximates the probability density `p(x)`:

```
rate(x) ≈ max_rate · max(0, mu · phi(x) − xi)
```

where `phi(.)` is the SSP encoding function from `ssp_space` and `max_rate` is the per-neuron gain.

The class exposes two construction modes via the `encoders` argument:

- **Single-neuron** (`encoders=None`, default): one neuron with `encoder = mu`. Drive the network with a swept query SSP and the neuron's response curve traces the density over the sweep.
- **Population** (`encoders = ssp_space.encode(query_points)`): one neuron per evaluation point. Drive with `mu` and the per-neuron firing rates across the population approximate `p(query_point)`.

Both modes share the same underlying ensemble construction — the encoder matrix is the only knob that differs. See `examples/example_usage.py` for both demos and `examples/probability_encoding.ipynb` for Michael Furlong's original tutorial.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The SSP encoder comes from `ctn-waterloo/ssp-bayesopt` (Michael Furlong's `ssp-bayes-opt` package; the SSP classes live under `ssp_bayes_opt.sspspace`). It is not on PyPI, hence the `git+https://…` install.

## Usage

```python
import nengo
import numpy as np
from scipy.stats import beta as beta_dist
from ssp_bayes_opt import sspspace
from probability_encoding import ProbabilityEncoder

# 1. Training samples drawn from some 1-D distribution.
samples = beta_dist.rvs(2, 5, size=500).reshape(-1, 1)

# 2. SSP space (lengthscale will be set automatically from the data).
ssp_space = sspspace.RandomSSPSpace(ssp_dim=128, domain_dim=1)

# 3. Population variant: one neuron per query point. Construct the
# encoder INSIDE the parent network so Nengo can see it at build time.
query_xs = np.linspace(0, 1, 50).reshape(-1, 1)
with nengo.Network() as net:
    pop = ProbabilityEncoder(
        ssp_space=ssp_space,
        training_samples=samples,
        domain_bounds=(0.0, 1.0),
        query_points=query_xs,
        max_rate=50.0,
    )

    # 4. Drive with mu; per-neuron firing rates ≈ p(query_xs).
    stim = nengo.Node(lambda t: pop.mu)
    nengo.Connection(stim, pop.input, synapse=None)
    probe = nengo.Probe(pop.ensemble.neurons)

with nengo.Simulator(net) as sim:
    sim.run(0.1)

estimated = sim.data[probe].mean(axis=0) / 50.0
```

## How it works

Given `n` training samples `{x_i}`, the encoder constructs the empirical SSP mean

```
mu = (1 / n) · Σ phi(x_i)  /  ls
```

where `ls` is a data-driven bandwidth (default: Furlong's empirical-characteristic-function estimator, falling back to Silverman). The normalization `xi` is chosen so that the density estimate integrates to 1 on the domain — concretely, by minimizing `(1 − ∫ max(0, mu·phi(x) − xi) dx)²` over `xi` via L-BFGS-B.

A Nengo `SpikingRectifiedLinear` ensemble with `gain = max_rate · 1` and `bias = −xi · max_rate · 1` then implements the relation

```
rate(x) ≈ max_rate · max(0, mu · phi(x) − xi).
```

The choice of `encoders` decides which "x" you're querying:
- default (no `query_points`) → one neuron whose tuning curve, evaluated by driving the input with `phi(x)`, traces `p(x)`.
- `query_points = X` → one neuron per query point, whose tuning curve evaluated at the constant input `mu` gives `p(query_point)`.

See Michael Furlong's tutorial notebook (`examples/probability_encoding.ipynb`) for the full derivation and a Gaussian-mixture example.

## Citation

This submission curates the spiking-implementation tutorial from the 2025 Nengo Summer School:

```
Furlong, P.M. (2025). Probability Encodings — Spiking Implementation.
Nengo Summer School tutorial.
https://github.com/ctn-waterloo/summerschool2025/tree/main/tutorials/vsa_probability
```

The vendored `fie_util.py` and `ssp_util.py` are included from the same source with the author's permission.

## License

GPLv2 for this submission (see `LICENSE`). The upstream `summerschool2025` repository declares no license; Michael Furlong has granted permission to ship the vendored utility modules under GPLv2 here.
