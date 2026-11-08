"""Solve u_tt = c^2 u_xx on [0,L], with u(0,t) = u(L,t) = 0."""
from __future__ import annotations

import numpy as np

from ..result import Result, finish
from ._common import (coordinates, fd_grid, interpolated_solution, positive,
                      sine_coefficients, snapshot_plotter, value_with_boundaries)


def wave_separation(f, g, L, c, n_terms, explain=False, plot=False):
    """Return u(x,t) from sine-series initial displacement f and velocity g.

    value is callable on 0 <= x <= L, t >= 0 and accepts broadcastable arrays.
    Both initial conditions are approximated by n_terms sine modes using
    adaptive quadrature. Plots span half a fundamental oscillation period.
    """
    positive(L, "L")
    positive(c, "c")
    displacement = sine_coefficients(f, L, n_terms)
    velocity = sine_coefficients(g, L, n_terms)
    frequencies = np.arange(1, n_terms + 1) * np.pi / L

    def u(x, t):
        x, t = coordinates(x, t, L)
        values = np.zeros_like(x)
        for a, b, frequency in zip(displacement, velocity, frequencies):
            omega = c * frequency
            values += (a * np.cos(omega * t) + b / omega * np.sin(omega * t)) * np.sin(frequency * x)
        return value_with_boundaries(values, x, L)

    steps = [
        "Wave equation: u_tt = c^2 u_xx, u(0,t) = u(L,t) = 0",
        "Initial displacement u(x,0) = f(x), initial velocity u_t(x,0) = g(x)",
        "Compute sine coefficients a_k of f and b_k of g with (2/L) integral_0^L h(x) sin(k pi x/L) dx",
        f"Adaptive quadrature, {n_terms} modes; a = {displacement}, b = {velocity}",
        "omega_k = c k pi/L; u = sum_k [a_k cos(omega_k t) + b_k sin(omega_k t)/omega_k] sin(k pi x/L)",
    ]
    plotter = snapshot_plotter(u, L, np.linspace(0, L / c, 4), "Wave equation: separation of variables")
    return finish(Result(u, steps, plotter), explain, plot)


def wave_fd(f, g, L, c, nx, dt, T, explain=False, plot=False):
    """Leapfrog solution of u_tt = c^2 u_xx with zero end values.

    nx counts spatial points including endpoints. The time step is reduced to
    T/ceil(T/dt); the effective Courant number c*dt/dx must be <= 1.
    The first step includes initial velocity g and half the initial acceleration.
    value is callable u(x,t) with bilinear interpolation on [0,L] x [0,T].
    Initial endpoint values are set to zero. Plots show four times in [0,T].
    """
    positive(c, "c")
    xs, times, dx, step = fd_grid(L, nx, dt, T)
    courant = c * step / dx
    if courant > 1 + 1e-12:
        raise ValueError("Leapfrog stability requires c * effective dt / dx <= 1 (CFL condition).")
    history = np.zeros((len(times), nx))
    history[0, 1:-1] = [f(x) for x in xs[1:-1]]
    initial_velocity = np.array([g(x) for x in xs[1:-1]])
    initial = history[0]
    history[1, 1:-1] = (initial[1:-1] + step * initial_velocity
                        + courant**2 / 2 * (initial[:-2] - 2 * initial[1:-1] + initial[2:]))
    for m in range(1, len(times) - 1):
        old = history[m]
        history[m + 1, 1:-1] = (2 * old[1:-1] - history[m - 1, 1:-1]
                                + courant**2 * (old[:-2] - 2 * old[1:-1] + old[2:]))
    u = interpolated_solution(xs, times, history)
    steps = [
        "Wave equation: u_tt = c^2 u_xx with zero Dirichlet endpoints",
        f"Leapfrog: nx = {nx}, dx = {dx:.8g}, requested dt = {dt:.8g}",
        f"Use {len(times) - 1} time steps of dt = {step:.8g} to reach T = {T:.8g}",
        f"Courant number c dt/dx = {courant:.8g} <= 1 (CFL stability)",
        "First step: u^1 = f + dt g + (c dt/dx)^2 (f_{i-1} - 2 f_i + f_{i+1})/2",
        "Leapfrog: u^{m+1} = 2 u^m - u^{m-1} + (c dt/dx)^2 (u^m_{i-1} - 2 u^m_i + u^m_{i+1})",
        "Return u(x,t) using linear interpolation in space and time on the stored grid",
    ]
    plotter = snapshot_plotter(u, L, np.linspace(0, T, 4), "Wave equation: leapfrog")
    return finish(Result(u, steps, plotter), explain, plot)
