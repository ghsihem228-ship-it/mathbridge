"""Finite-dimensional approximations of operators on real L^2([a, b])."""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import quad

from ..result import Result, finish
from .spaces import Func, Interval


def galerkin_matrix(op: Callable[[Func], Func], basis: Sequence[Func],
                    interval: Interval, explain: bool = False, plot: bool = False):
    """Return A[i, j] = <op(e_j), e_i> for a real orthonormal basis.

    op maps a function to a function. Operator and basis callables must be
    deterministic; evaluations of each transformed basis function are cached
    during its column's quadrature. Orthonormality is assumed.
    """
    if not basis:
        raise ValueError("The basis must contain at least one function.")
    a, b = interval
    matrix = np.empty((len(basis), len(basis)))
    for j, ej in enumerate(basis):
        transformed = lru_cache(maxsize=None)(op(ej))
        for i, ei in enumerate(basis):
            matrix[i, j] = quad(lambda x: transformed(x) * ei(x),
                                a, b, limit=200)[0]
    steps = [
        "Galerkin approximation in an orthonormal basis: A[i,j] = <T e_j, e_i>",
        f"Integrate (T e_j)(x) e_i(x) over [{a}, {b}] with adaptive quadrature",
        f"Matrix shape = {matrix.shape}",
    ]

    def plotter():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        im = ax.imshow(matrix, origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set(xlabel="Basis index j", ylabel="Basis index i",
               title="Galerkin matrix")
        return fig

    return finish(Result(matrix, steps, plotter), explain, plot)


def integral_operator(kernel: Callable[[float, float], float], interval: Interval,
                      explain: bool = False, plot: bool = False):
    """Return a Result whose value is T: f -> (x -> integral K(x,y) f(y) dy).

    The integral is over interval. The plot displays K on interval x interval.
    """
    a, b = interval

    def T(f):
        def transformed(x):
            # Splitting at y=x also resolves kernels with a diagonal kink.
            points = [x] if a < x < b else None
            return quad(lambda y: kernel(x, y) * f(y), a, b,
                        points=points, limit=200)[0]

        return transformed

    steps = [
        f"Integral operator: (T f)(x) = integral from {a} to {b} of K(x,y) f(y) dy",
        "Evaluate each integral with adaptive quadrature (split at y=x when inside the interval)",
    ]

    def plotter():
        import matplotlib.pyplot as plt

        xs = np.linspace(a, b, 100)
        values = [[kernel(x, y) for y in xs] for x in xs]
        fig, ax = plt.subplots()
        im = ax.imshow(values, extent=(a, b, a, b), origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set(xlabel="y", ylabel="x", title="Integral operator kernel K(x,y)")
        return fig

    return finish(Result(T, steps, plotter), explain, plot)


def spectrum(matrix, self_adjoint: bool = True,
             explain: bool = False, plot: bool = False):
    """Return eigenvalues sorted by decreasing absolute value.

    With self_adjoint=True, matrix is assumed symmetric (Hermitian for complex
    entries). Otherwise complex eigenvalues are retained. The plot uses the
    complex plane for either case.
    """
    matrix = np.asarray(matrix)
    if self_adjoint:
        values = np.linalg.eigvalsh(matrix)
        method = "numpy.linalg.eigvalsh for a self-adjoint (Hermitian) matrix"
    else:
        values = np.linalg.eigvals(matrix)
        method = "numpy.linalg.eigvals for a general matrix"
    values = values[np.argsort(-np.abs(values), kind="stable")]
    steps = [
        f"Compute eigenvalues using {method}",
        "Sort eigenvalues by decreasing absolute value",
        f"Eigenvalues = {values}",
    ]

    def plotter():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.scatter(values.real, values.imag)
        ax.axhline(0, color="gray", ls=":")
        ax.axvline(0, color="gray", ls=":")
        ax.set(xlabel="Real part", ylabel="Imaginary part", title="Operator spectrum")
        return fig

    return finish(Result(values, steps, plotter), explain, plot)


def operator_norm(matrix, explain: bool = False, plot: bool = False):
    """Return the largest singular value, approximating ||T|| in an orthonormal basis.

    Rectangular matrices are supported. The plot displays all singular values.
    """
    singular_values = np.linalg.svd(np.asarray(matrix), compute_uv=False)
    value = float(np.max(singular_values))
    steps = [
        "Compute singular values using numpy.linalg.svd",
        "The induced Euclidean norm is the largest singular value",
        "For an orthonormal Galerkin basis this approximates the L^2 operator norm",
        f"||A||_2 = {value:.10g}",
    ]

    def plotter():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(np.arange(1, len(singular_values) + 1), singular_values, "o-")
        ax.set(xlabel="Index", ylabel="Singular value", title="Operator norm and singular values")
        return fig

    return finish(Result(value, steps, plotter), explain, plot)
