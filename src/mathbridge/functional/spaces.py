"""Hilbert-space tools for functions in L^2([a, b])."""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np
from scipy.integrate import quad

from ..result import Result, finish

Func = Callable[[float], float]
Interval = Tuple[float, float]


def _ip(f: Func, g: Func, interval: Interval) -> float:
    a, b = interval
    val, _ = quad(lambda x: f(x) * g(x), a, b, limit=200)
    return val


def inner_product(f: Func, g: Func, interval: Interval, explain: bool = False):
    """<f, g> = integral of f(x) g(x) dx over the interval (real L^2)."""
    a, b = interval
    val = _ip(f, g, interval)
    steps = [
        f"Definition: <f,g> = integral from {a} to {b} of f(x)g(x) dx",
        f"Numerical integration (adaptive quadrature) gives {val:.10g}",
    ]
    return finish(Result(val, steps), explain, False)


def norm(f: Func, interval: Interval, explain: bool = False):
    """||f||_2 = sqrt(<f, f>)."""
    ip = _ip(f, f, interval)
    val = float(np.sqrt(ip))
    steps = [
        "Definition: ||f|| = sqrt(<f,f>)",
        f"<f,f> = {ip:.10g}",
        f"||f|| = {val:.10g}",
    ]
    return finish(Result(val, steps), explain, False)


def distance(f: Func, g: Func, interval: Interval, explain: bool = False):
    """d(f, g) = ||f - g||_2."""
    r = norm(lambda x: f(x) - g(x), interval)
    steps = ["Definition: d(f,g) = ||f - g||"] + r.steps[1:]
    return finish(Result(r.value, steps), explain, False)


def gram_schmidt(
    funcs: Sequence[Func],
    interval: Interval,
    explain: bool = False,
    tol: float = 1e-12,
):
    """Orthonormalise functions in L^2 with the Gram-Schmidt process.

    Returns a Result whose value is a list of orthonormal callables.
    Linearly dependent inputs are skipped (and noted in the steps).
    """
    basis: list = []
    steps = ["Gram-Schmidt: e_k = u_k / ||u_k||, u_k = f_k - sum_j <f_k,e_j> e_j"]
    for k, f in enumerate(funcs, 1):
        coeffs = [_ip(f, e, interval) for e in basis]

        def u(x, f=f, coeffs=coeffs, basis=tuple(basis)):
            return f(x) - sum(c * e(x) for c, e in zip(coeffs, basis))

        n = float(np.sqrt(_ip(u, u, interval)))
        if n < tol:
            steps.append(f"f_{k}: dependent on previous functions, skipped")
            continue
        basis.append(lambda x, u=u, n=n: u(x) / n)
        steps.append(
            f"f_{k}: projections {[round(c, 8) for c in coeffs]}, ||u_{k}|| = {n:.8g}"
        )
    return finish(Result(basis, steps), explain, False)


def project(f: Func, orthonormal_basis: Sequence[Func], interval: Interval,
            explain: bool = False, plot: bool = False):
    """Orthogonal projection of f onto span(orthonormal_basis).

    value = (callable projection, list of coefficients <f, e_k>).
    """
    coeffs = [_ip(f, e, interval) for e in orthonormal_basis]

    def p(x):
        return sum(c * e(x) for c, e in zip(coeffs, orthonormal_basis))

    err = float(np.sqrt(max(_ip(f, f, interval) - sum(c * c for c in coeffs), 0.0)))
    steps = [
        "Projection: P f = sum_k <f,e_k> e_k  (basis assumed orthonormal)",
        f"Coefficients <f,e_k> = {[round(c, 8) for c in coeffs]}",
        f"Error ||f - P f|| = {err:.8g}  (from ||f||^2 - sum |c_k|^2)",
    ]

    def plotter():
        import matplotlib.pyplot as plt

        a, b = interval
        xs = np.linspace(a, b, 400)
        fig, ax = plt.subplots()
        ax.plot(xs, [f(x) for x in xs], label="f")
        ax.plot(xs, [p(x) for x in xs], "--", label="projection")
        ax.legend()
        ax.set_title("Orthogonal projection")
        return fig

    return finish(Result((p, coeffs), steps, plotter), explain, plot)
