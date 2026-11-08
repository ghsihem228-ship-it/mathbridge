[![Tests](https://github.com/ghsihem228-ship-it/mathbridge/actions/workflows/tests.yml/badge.svg)](https://github.com/ghsihem228-ship-it/mathbridge/actions/workflows/tests.yml)

# mathbridge

A unified layer over NumPy / SciPy for the topics that Python leaves scattered:
**functional analysis** and **Fourier series** (more modules planned).
Every function returns a `Result` carrying the answer, the solution steps, and a plot.

```python
import numpy as np
from mathbridge import functional, fourier

# Functional analysis in L^2([-1, 1])
r = functional.norm(lambda x: x, (-1, 1), explain=True)
basis = functional.gram_schmidt([lambda x: 1, lambda x: x, lambda x: x**2], (-1, 1)).value

# Fourier series of a square wave
sq = lambda x: 1.0 if x > 0 else -1.0
fourier.partial_sum(sq, np.pi, n_terms=15, plot=True)
fourier.gibbs_overshoot(sq, np.pi, n_terms=50, explain=True)
```

## Modules

| Module | Functions |
|---|---|
| `mathbridge.functional` | `inner_product`, `norm`, `distance`, `gram_schmidt`, `project`, `galerkin_matrix`, `integral_operator`, `spectrum`, `operator_norm` |
| `mathbridge.fourier` | `coefficients`, `partial_sum`, `parseval`, `gibbs_overshoot` |

## Gallery

Square wave Fourier partial sums with N=5 and N=50 show the Gibbs phenomenon.

![Square wave and Fourier partial sums for N=5 and N=50](docs/img/gibbs.png)

Projecting x² onto a Legendre-like basis built with `gram_schmidt`: the first two
functions give the constant 1/3, while all three reproduce x² exactly.

![L2 projections of x squared onto two and three Legendre basis functions](docs/img/projection.png)

Eigenvalues of the integral operator with kernel min(x,y) on [0,1], computed
using `integral_operator`, `galerkin_matrix`, and `spectrum`, compared with the
exact values 1/(((n-1/2)π)²).

![Computed and exact eigenvalues of the min(x,y) integral operator](docs/img/spectrum.png)

Regenerate the PNGs after installing the package:

```bash
python examples/make_figures.py
```

## Roadmap

- Functional: convergence modes and convergence estimates for Galerkin operator approximations
- PDE: heat / wave / Laplace by separation of variables and finite differences
- Differential geometry: curvature, Frenet frame, geodesics
- Numerical methods with the same calling style

## Install and test

```bash
pip install -e ".[dev]"
pytest
```
