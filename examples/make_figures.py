"""Regenerate the README gallery: python examples/make_figures.py.

Install mathbridge first with pip install -e ".[dev]". Output paths are
relative to this file's repository, so the script can run from any directory.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from mathbridge import functional, fourier


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "img"


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def make_gibbs():
    def square(x):
        return 1.0 if x > 0 else -1.0

    xs = np.linspace(-np.pi, np.pi, 4001)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(xs, [square(x) for x in xs], color="0.35", ls="--",
            lw=1.4, label="Square wave")
    for n_terms in (5, 50):
        partial = fourier.partial_sum(square, np.pi, n_terms=n_terms).value
        ax.plot(xs, partial(xs), lw=1.7, label=f"N = {n_terms}")
    ax.set(xlabel="x", ylabel="f(x), S_N(x)",
           title="Gibbs phenomenon: Fourier partial sums", xlim=(-np.pi, np.pi))
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                  [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")
    save_figure(fig, "gibbs.png")


def make_projection():
    interval = (-1, 1)
    basis = functional.gram_schmidt(
        [lambda x: 1.0, lambda x: x, lambda x: x**2], interval
    ).value
    f = lambda x: x**2
    linear, _ = functional.project(f, basis[:2], interval).value
    quadratic, _ = functional.project(f, basis, interval).value
    xs = np.linspace(*interval, 501)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.plot(xs, f(xs), color="0.25", lw=3, label=r"$f(x) = x^2$")
    ax.plot(xs, [linear(x) for x in xs], ls=":", lw=2,
            label=r"Projection onto span$(e_0,e_1)$: $1/3$")
    ax.plot(xs, [quadratic(x) for x in xs], ls="--", lw=2,
            label=r"Projection onto span$(e_0,e_1,e_2)$: $x^2$")
    ax.set(xlabel="x", ylabel="Function value", xlim=interval,
           title="L² projection onto a Gram–Schmidt (Legendre) basis")
    ax.grid(alpha=0.2)
    ax.legend()
    save_figure(fig, "projection.png")


def make_spectrum():
    n_terms = 12
    # These half-integer sine functions are orthonormal eigenfunctions of
    # K(x,y)=min(x,y), whose boundary conditions are f(0)=0 and f'(1)=0.
    basis = [lambda x, k=k: np.sqrt(2) * np.sin((k - 0.5) * np.pi * x)
             for k in range(1, n_terms + 1)]
    operator = functional.integral_operator(lambda x, y: min(x, y), (0, 1)).value
    matrix = functional.galerkin_matrix(operator, basis, (0, 1)).value
    values = functional.spectrum(matrix).value
    indices = np.arange(1, n_terms + 1)
    exact = 1 / (((indices - 0.5) * np.pi) ** 2)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.semilogy(indices, exact, color="0.35", ls="--", lw=1.5,
                label=r"Exact: $1/((n-1/2)\pi)^2$")
    ax.semilogy(indices, values, "o", ms=6, fillstyle="none",
                label=f"Galerkin eigenvalues ({n_terms} basis functions)")
    ax.set(xlabel="Eigenvalue index n", ylabel=r"$\lambda_n$",
           title="Integral operator spectrum: K(x, y) = min(x, y) on [0, 1]")
    ax.set_xticks(indices)
    ax.grid(alpha=0.2, which="both")
    ax.legend()
    save_figure(fig, "spectrum.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_gibbs()
    make_projection()
    make_spectrum()


if __name__ == "__main__":
    main()
