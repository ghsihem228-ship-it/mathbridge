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
