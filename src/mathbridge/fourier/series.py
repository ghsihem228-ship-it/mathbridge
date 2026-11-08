"""Real Fourier series on [-L, L] (period T = 2L)."""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import quad

from ..result import Result, finish

Func = Callable[[float], float]


def coefficients(f: Func, L: float = np.pi, n_terms: int = 10,
                 explain: bool = False):
    """Fourier coefficients a_0, a_n, b_n of f on [-L, L].

    value = (a0, a, b) with a[k-1] = a_k and b[k-1] = b_k, using
    f ~ a0/2 + sum a_n cos(n pi x / L) + b_n sin(n pi x / L).
    """
    a0 = quad(f, -L, L, limit=200)[0] / L
    a, b = [], []
    for n in range(1, n_terms + 1):
        w = n * np.pi / L
        a.append(quad(lambda x: f(x) * np.cos(w * x), -L, L, limit=200)[0] / L)
        b.append(quad(lambda x: f(x) * np.sin(w * x), -L, L, limit=200)[0] / L)
    steps = [
        f"Interval [-L, L] with L = {L:.6g}",
        "a_0 = (1/L) int f dx, a_n = (1/L) int f cos(n pi x/L) dx, "
        "b_n = (1/L) int f sin(n pi x/L) dx",
        f"a_0 = {a0:.8g}",
        f"a_1..a_{n_terms} = {[round(v, 8) for v in a]}",
        f"b_1..b_{n_terms} = {[round(v, 8) for v in b]}",
    ]
    return finish(Result((a0, np.array(a), np.array(b)), steps), explain, False)


def partial_sum(f: Func, L: float = np.pi, n_terms: int = 10,
                explain: bool = False, plot: bool = False):
    """Partial sum S_N of the Fourier series. value = callable S_N(x)."""
    a0, a, b = coefficients(f, L, n_terms).value

    def S(x):
        x = np.asarray(x, dtype=float)
        out = np.full_like(x, a0 / 2.0)
        for n in range(1, n_terms + 1):
            w = n * np.pi / L
            out = out + a[n - 1] * np.cos(w * x) + b[n - 1] * np.sin(w * x)
        return out

    steps = [f"S_N(x) = a0/2 + sum_(n=1)^N [a_n cos + b_n sin], N = {n_terms}"]

    def plotter():
        import matplotlib.pyplot as plt

        xs = np.linspace(-L, L, 800)
        fig, ax = plt.subplots()
        ax.plot(xs, [f(x) for x in xs], label="f")
        ax.plot(xs, S(xs), "--", label=f"S_{n_terms}")
        ax.legend()
        ax.set_title("Fourier partial sum")
        return fig

    return finish(Result(S, steps, plotter), explain, plot)


def parseval(f: Func, L: float = np.pi, n_terms: int = 50,
             explain: bool = False):
    """Check Parseval: (1/L) int f^2 = a0^2/2 + sum (a_n^2 + b_n^2).

    value = (lhs, rhs_partial, gap); the gap tends to 0 as n_terms grows.
    """
    a0, a, b = coefficients(f, L, n_terms).value
    lhs = quad(lambda x: f(x) ** 2, -L, L, limit=200)[0] / L
    rhs = a0 ** 2 / 2 + float(np.sum(a ** 2 + b ** 2))
    steps = [
        "Parseval: (1/L) int f^2 dx = a_0^2/2 + sum (a_n^2 + b_n^2)",
        f"Left side = {lhs:.10g}",
        f"Right side with {n_terms} terms = {rhs:.10g}",
        f"Gap = {lhs - rhs:.3e}",
    ]
    return finish(Result((lhs, rhs, lhs - rhs), steps), explain, False)


def gibbs_overshoot(f: Func, L: float = np.pi, n_terms: int = 50,
                    explain: bool = False, plot: bool = False):
    """Peak of S_N near a jump, to show the Gibbs phenomenon.

    value = max of S_N on [-L, L]. For a unit square wave the overshoot
    (jump from -1 to 1) the peak tends to 1 + 0.0895 * 2 = 1.179, i.e. an
    overshoot of about 8.95% of the jump size.
    """
    S = partial_sum(f, L, n_terms).value
    xs = np.linspace(-L, L, 20001)
    peak = float(np.max(S(xs)))
    steps = [
        f"Sample S_{n_terms} on a fine grid over [-L, L]",
        f"Maximum value = {peak:.6f}",
    ]

    def plotter():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(xs, S(xs))
        ax.axhline(peak, color="gray", ls=":")
        ax.set_title(f"Gibbs phenomenon, N = {n_terms}")
        return fig

    return finish(Result(peak, steps, plotter), explain, plot)
