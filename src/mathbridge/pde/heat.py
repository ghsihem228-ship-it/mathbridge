"""Solve u_t = alpha u_xx on [0,L], with u(0,t) = u(L,t) = 0."""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from ..result import Result, finish
from ._common import (coordinates, fd_grid, interpolated_solution, positive,
                      sine_coefficients, snapshot_plotter, value_with_boundaries)


def heat_separation(f, L, alpha, n_terms, explain=False, plot=False):
    """Return a sine-series solution u(x,t) with initial displacement f(x).

    value is callable on 0 <= x <= L, t >= 0 and accepts broadcastable arrays.
    Coefficients use adaptive quadrature. At t=0 the truncated sine series
    approximates f; boundary values are always zero. Plots show four snapshots
    over two decay times of the fundamental mode.
    """
    positive(L, "L")
    positive(alpha, "alpha")
    coeffs = sine_coefficients(f, L, n_terms)
    frequencies = np.arange(1, n_terms + 1) * np.pi / L

    def u(x, t):
        x, t = coordinates(x, t, L)
        values = np.zeros_like(x)
        for coefficient, frequency in zip(coeffs, frequencies):
            values += coefficient * np.sin(frequency * x) * np.exp(-alpha * frequency**2 * t)
        return value_with_boundaries(values, x, L)

    steps = [
        "Heat equation: u_t = alpha u_xx, u(0,t) = u(L,t) = 0, u(x,0) = f(x)",
        "Sine coefficients b_k = (2/L) integral_0^L f(x) sin(k pi x/L) dx, using adaptive quadrature",
        f"Keep {n_terms} modes; coefficients = {coeffs}",
        "u(x,t) = sum_k b_k exp(-alpha (k pi/L)^2 t) sin(k pi x/L)",
    ]
    times = np.linspace(0, 2 / (alpha * (np.pi / L)**2), 4)
    plotter = snapshot_plotter(u, L, times, "Heat equation: separation of variables")
    return finish(Result(u, steps, plotter), explain, plot)


def heat_fd(f, L, alpha, nx, dt, T, explain=False, plot=False):
    """Crank–Nicolson solution of u_t = alpha u_xx with zero end values.

    nx counts spatial grid points including the endpoints. The time step is
    reduced to T/ceil(T/dt) to reach T exactly. value is a callable u(x,t) on
    [0,L] x [0,T], with bilinear interpolation of stored grid values.
    Initial endpoint values are set to zero. Plots show four times in [0,T].
    """
    positive(alpha, "alpha")
    xs, times, dx, step = fd_grid(L, nx, dt, T)
    r = alpha * step / dx**2
    history = np.zeros((len(times), nx))
    history[0, 1:-1] = [f(x) for x in xs[1:-1]]
    # A u^{m+1} = B u^m on the interior; endpoint contributions vanish.
    banded = np.zeros((3, nx - 2))
    banded[0, 1:] = -r / 2
    banded[1, :] = 1 + r
    banded[2, :-1] = -r / 2
    for m in range(len(times) - 1):
        old = history[m]
        rhs = (1 - r) * old[1:-1] + r / 2 * (old[:-2] + old[2:])
        history[m + 1, 1:-1] = solve_banded((1, 1), banded, rhs)
    u = interpolated_solution(xs, times, history)
    steps = [
        "Heat equation: u_t = alpha u_xx with zero Dirichlet endpoints",
        f"Crank-Nicolson: nx = {nx}, dx = {dx:.8g}, requested dt = {dt:.8g}",
        f"Use {len(times) - 1} time steps of dt = {step:.8g} to reach T = {T:.8g}",
        "(I - alpha dt D_xx/2) u^{m+1} = (I + alpha dt D_xx/2) u^m",
        "Solve the tridiagonal system each step; zero endpoints at every time",
        "Return u(x,t) using linear interpolation in space and time on the stored grid",
    ]
    plotter = snapshot_plotter(u, L, np.linspace(0, T, 4), "Heat equation: Crank-Nicolson")
    return finish(Result(u, steps, plotter), explain, plot)
