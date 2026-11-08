"""Numerical evidence for convergence of function sequences in L^2([a, b])."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.integrate import quad

from ..result import Result, finish
from .spaces import Func, Interval


def _trends_to_zero(errors):
    """A finite-data heuristic: zero, or a decrease with a negative tail slope."""
    errors = np.asarray(errors, dtype=float)
    if np.all(errors == 0):
        return True
    if len(errors) < 2:
        return False
    tail = errors[-max(2, (len(errors) + 1) // 2):]
    slope = np.polyfit(np.arange(len(tail)), tail, 1)[0]
    return bool(errors[-1] < errors[0] and slope < 0)


def convergence_report(fn_seq: Callable[[int], Func], f: Func,
                       interval: Interval, n_values: Sequence[int],
                       explain: bool = False, plot: bool = False):
    """Compare f_n = fn_seq(n) with f on a finite interval.

    value is a dict with three lists in n_values order:
      pointwise: one list of absolute errors per n, at five evenly spaced
                 sample points including both endpoints;
      sup: maximum absolute error on a 2001-point grid including endpoints;
      l2: sqrt(integral |f_n - f|^2), computed with adaptive quadrature.

    The plot shows the maximum of the five pointwise samples, the grid sup
    error, and the L2 error on a logarithmic y-axis. Exact zeros are displayed
    at machine epsilon (the returned errors are unchanged).

    Steps report apparent convergence when the final error is smaller than
    the first and the last half of the errors has a negative fitted slope.
    Identically zero errors count as converged. Pointwise convergence requires
    this at every sample separately. Finite samples and trends are evidence,
    not a proof of convergence; a grid can miss narrow error peaks.
    """
    a, b = interval
    if not np.isfinite([a, b]).all() or a >= b:
        raise ValueError("interval must have finite endpoints with a < b.")
    ns = list(n_values)
    if not ns or any(not isinstance(n, (int, np.integer)) for n in ns):
        raise ValueError("n_values must be a nonempty sequence of integers.")
    if any(right <= left for left, right in zip(ns, ns[1:])):
        raise ValueError("n_values must be strictly increasing.")
    sample_points = np.linspace(a, b, 5)
    grid = np.linspace(a, b, 2001)
    errors = {"pointwise": [], "sup": [], "l2": []}
    for n in ns:
        fn = fn_seq(n)

        def error(x):
            return abs(fn(x) - f(x))

        errors["pointwise"].append([float(error(x)) for x in sample_points])
        errors["sup"].append(float(max(error(x) for x in grid)))
        squared_error = quad(lambda x: error(x)**2, a, b, limit=200)[0]
        errors["l2"].append(float(np.sqrt(squared_error)))

    pointwise_trends = [_trends_to_zero(column)
                        for column in np.asarray(errors["pointwise"]).T]
    trends = {
        "Pointwise at all sample points": all(pointwise_trends),
        "Sup-norm on the fine grid (uniform convergence)": _trends_to_zero(errors["sup"]),
        "L2": _trends_to_zero(errors["l2"]),
    }
    steps = [
        f"Compare f_n with f over [{a}, {b}] for n = {ns}",
        f"Pointwise absolute errors sampled at x = {sample_points.tolist()}",
        "Sup-norm estimate: maximum absolute error on a 2001-point grid, including endpoints",
        "L2 error: sqrt(integral |f_n - f|^2 dx) using adaptive quadrature",
        "Trend heuristic: final error below initial error and negative fitted slope over the last half; zero throughout also qualifies",
    ]
    for mode, converges in trends.items():
        verdict = "appears to converge" if converges else "does not appear to converge"
        steps.append(f"{mode}: {verdict} to zero over the sampled n values")
    for x, converges in zip(sample_points, pointwise_trends):
        verdict = "appears to converge" if converges else "does not appear to converge"
        steps.append(f"Pointwise at x = {x:.6g}: error {verdict} to zero")
    steps.append("Finite samples and decreasing errors do not prove convergence; the grid may miss narrow peaks.")

    def plotter():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        series = {
            "Pointwise (max over 5 samples)": np.max(errors["pointwise"], axis=1),
            "Sup (2001-point grid)": errors["sup"],
            "L2 (quadrature)": errors["l2"],
        }
        for label, values in series.items():
            ax.semilogy(ns, np.maximum(values, np.finfo(float).eps), "o-", label=label)
        ax.set(xlabel="n", ylabel="Error (zeros shown at machine epsilon)",
               title="Function sequence convergence")
        ax.grid(alpha=0.2)
        ax.legend()
        return fig

    return finish(Result(errors, steps, plotter), explain, plot)
