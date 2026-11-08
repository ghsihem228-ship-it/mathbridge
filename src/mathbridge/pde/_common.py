"""Shared quadrature, grids, interpolation, and plots for PDE solvers."""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import RegularGridInterpolator


def positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def integer_at_least(value, minimum, name):
    if not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def sine_coefficients(f, L, n_terms):
    integer_at_least(n_terms, 1, "n_terms")
    return np.array([2 / L * quad(lambda x: f(x) * np.sin(k * np.pi * x / L),
                                  0, L, limit=200)[0]
                     for k in range(1, n_terms + 1)])


def coordinates(x, t, L, T=None):
    x, t = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(t, dtype=float))
    if (not np.isfinite(x).all() or not np.isfinite(t).all()
            or np.any((x < 0) | (x > L)) or np.any(t < 0)):
        raise ValueError("Require finite coordinates with 0 <= x <= L and t >= 0.")
    if T is not None and np.any(t > T):
        raise ValueError("Finite-difference solutions require t <= T.")
    return x, t


def value_with_boundaries(values, x, L):
    values = np.where((x == 0) | (x == L), 0.0, values)
    return float(values) if values.ndim == 0 else values


def fd_grid(L, nx, dt, T):
    positive(L, "L")
    integer_at_least(nx, 3, "nx")
    positive(dt, "dt")
    positive(T, "T")
    nt = int(np.ceil(T / dt))
    xs = np.linspace(0, L, nx)
    times = np.linspace(0, T, nt + 1)
    return xs, times, L / (nx - 1), T / nt


def interpolated_solution(xs, times, history):
    interpolator = RegularGridInterpolator((times, xs), history)
    L, T = xs[-1], times[-1]

    def u(x, t):
        x, t = coordinates(x, t, L, T)
        points = np.column_stack((t.ravel(), x.ravel()))
        values = interpolator(points).reshape(x.shape)
        return value_with_boundaries(values, x, L)

    return u


def snapshot_plotter(u, L, times, title):
    def plotter():
        import matplotlib.pyplot as plt

        xs = np.linspace(0, L, 401)
        fig, ax = plt.subplots()
        for t in times:
            ax.plot(xs, u(xs, t), label=f"t = {t:.4g}")
        ax.set(xlabel="x", ylabel="u(x,t)", title=title)
        ax.legend()
        ax.grid(alpha=0.2)
        return fig

    return plotter
