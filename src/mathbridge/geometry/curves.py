"""Curvature, torsion, and Frenet frames for curves in real three-space."""
from __future__ import annotations

import numpy as np
import sympy as sp

from ..result import Result, finish


def _vector(value):
    value = np.asarray(value, dtype=float)
    if value.size != 3 or not np.isfinite(value).all():
        raise ValueError("r(t) must return three finite real coordinates.")
    return value.reshape(3)


def _derivatives(r, h):
    if callable(r):
        h = np.finfo(float).eps**(1 / 7) if h is None else h
        if not np.isfinite(h) or h <= 0:
            raise ValueError("h must be finite and positive.")

        def evaluate(t):
            return _vector(r(float(t)))

        # Seven-point centered stencils: O(h^6) for derivatives 1 and 2,
        # O(h^4) for derivative 3. A larger h reduces third-derivative roundoff.
        weights = np.array([
            [-1 / 60, 3 / 20, -3 / 4, 0, 3 / 4, -3 / 20, 1 / 60],
            [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90],
            [1 / 8, -1, 13 / 8, 0, -13 / 8, 1, -1 / 8],
        ])

        def derivatives(t):
            center = evaluate(t)
            samples = np.array([evaluate(t + k * h) - center for k in range(-3, 4)])
            return (weights @ samples) / np.array([h, h**2, h**3])[:, None]

        return evaluate, derivatives, f"Numerical derivatives: seven-point centered differences with h = {h:.8g}"

    curve = sp.Matrix(r)
    if len(curve) != 3 or len(curve.free_symbols) != 1:
        raise ValueError("A symbolic curve must have three coordinates and exactly one free parameter; substitute other parameters first.")
    parameter = next(iter(curve.free_symbols))
    curve = curve.reshape(3, 1)
    position = sp.lambdify(parameter, curve, "numpy")
    differentiated = [sp.lambdify(parameter, curve.diff(parameter, order), "numpy")
                      for order in (1, 2, 3)]

    def evaluate(t):
        return _vector(position(float(t)))

    def derivatives(t):
        return np.array([_vector(fn(float(t))) for fn in differentiated])

    return evaluate, derivatives, f"Exact symbolic derivatives r', r'', r''' using SymPy with respect to {parameter}; evaluate numerically"


def _frame(velocity, cross, speed, cross_norm):
    if np.any(cross_norm == 0):
        raise ValueError("The Frenet frame and torsion require nonzero curvature (r' cross r'' != 0).")
    tangent = velocity / speed[..., None]
    binormal = cross / cross_norm[..., None]
    normal = np.cross(binormal, tangent)
    return np.stack((tangent, normal, binormal), axis=-2)


def _curve_result(r, t, mode, explain, plot, h, plot_interval):
    ts = np.asarray(t, dtype=float)
    if ts.size == 0 or not np.isfinite(ts).all():
        raise ValueError("t must contain finite parameter values and cannot be empty.")
    evaluate, differentiate, method = _derivatives(r, h)
    derivatives = np.array([differentiate(point) for point in ts.ravel()])
    derivatives = derivatives.reshape(ts.shape + (3, 3))
    velocity, acceleration, jerk = (derivatives[..., i, :] for i in range(3))
    speed = np.linalg.norm(velocity, axis=-1)
    if np.any(speed == 0):
        raise ValueError("Curve geometry requires nonzero speed (r'(t) != 0).")
    cross = np.cross(velocity, acceleration)
    cross_norm = np.linalg.norm(cross, axis=-1)
    if mode == "curvature":
        value = cross_norm / speed**3
        formula = "Curvature kappa = ||r' cross r''|| / ||r'||^3"
    elif mode == "torsion":
        if np.any(cross_norm == 0):
            raise ValueError("Torsion requires nonzero curvature (r' cross r'' != 0).")
        value = np.sum(cross * jerk, axis=-1) / cross_norm**2
        formula = "Torsion tau = (r' cross r'') dot r''' / ||r' cross r''||^2"
    else:
        value = _frame(velocity, cross, speed, cross_norm)
        formula = "Frenet frame rows: T = r'/||r'||, B = (r' cross r'')/||r' cross r''||, N = B cross T"
    value = np.asarray(value)
    steps = [method, formula, f"Evaluate at t = {ts}", f"{mode} = {value}"]

    def plotter():
        import matplotlib.pyplot as plt

        point = float(ts.flat[0])
        start, end = plot_interval if plot_interval is not None else (point - 1, point + 1)
        if not np.isfinite([start, end]).all() or not start < end or not start <= point <= end:
            raise ValueError("plot_interval must be finite, increasing, and contain the first evaluation point.")
        frame = _frame(velocity.reshape(-1, 3)[0], cross.reshape(-1, 3)[0],
                       speed.reshape(-1)[0], cross_norm.reshape(-1)[0])
        positions = np.array([evaluate(s) for s in np.linspace(start, end, 401)])
        origin = evaluate(point)
        scale = 0.2 * np.linalg.norm(np.ptp(positions, axis=0))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(*positions.T, color="0.4", label="r(t)")
        ax.scatter(*origin, color="black")
        for direction, label, color in zip(frame, ("T", "N", "B"), ("tab:red", "tab:green", "tab:blue")):
            ax.quiver(*origin, *direction, length=scale, color=color, label=label)
        # Use equal units on all axes, including for planar curves.
        extent = np.vstack((positions, origin + scale * frame))
        lower, upper = extent.min(axis=0), extent.max(axis=0)
        center = (lower + upper) / 2
        radius = max(float(np.max(upper - lower)) * 0.55, np.finfo(float).eps)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.set(xlabel="x", ylabel="y", zlabel="z", title=f"Curve and Frenet frame at t = {point:.4g}")
        ax.legend()
        return fig

    return finish(Result(value, steps, plotter), explain, plot)


def curvature(r, t, explain=False, plot=False, h=None, plot_interval=None):
    """Return curvature as a NumPy array with the same shape as t (0-D for scalar t).

    r is a callable returning three coordinates, or three SymPy expressions
    (including a Matrix) with one free parameter. Symbolic derivatives are
    exact before numerical evaluation. Callables use centered differences;
    h optionally sets their step, default machine_epsilon**(1/7). They must be
    smooth on a neighborhood of radius 3*h around each t.

    The plot draws the curve and T, N, B at the first evaluation point over
    plot_interval, default (t_first-1, t_first+1). Nonzero speed is required;
    plotting also requires nonzero curvature to define the Frenet frame.
    """
    return _curve_result(r, t, "curvature", explain, plot, h, plot_interval)


def torsion(r, t, explain=False, plot=False, h=None, plot_interval=None):
    """Return torsion as a NumPy array with t's shape; require nonzero curvature.

    r, h, and plotting options follow curvature(). The scalar triple product
    preserves the sign of torsion. Numerical third derivatives are sensitive
    to roundoff and the differentiation step.
    """
    return _curve_result(r, t, "torsion", explain, plot, h, plot_interval)


def frenet_frame(r, t, explain=False, plot=False, h=None, plot_interval=None):
    """Return an array of shape t.shape + (3,3), with rows T, N, B.

    For scalar t, value is a (3,3) array. The frame is orthonormal and oriented
    so T cross N = B. Nonzero speed and curvature are required. r, h, and
    plotting options follow curvature().
    """
    return _curve_result(r, t, "frenet_frame", explain, plot, h, plot_interval)
