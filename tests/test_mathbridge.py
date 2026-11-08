import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from mathbridge import fourier, functional

I = (-1.0, 1.0)


def test_inner_product_and_norm():
    assert functional.inner_product(lambda x: x, lambda x: x, I).value == pytest.approx(2 / 3)
    assert functional.norm(lambda x: 1.0, I).value == pytest.approx(np.sqrt(2))


def test_distance():
    d = functional.distance(lambda x: x, lambda x: 0.0, I).value
    assert d == pytest.approx(np.sqrt(2 / 3))


def test_gram_schmidt_gives_legendre_shape():
    basis = functional.gram_schmidt([lambda x: 1.0, lambda x: x, lambda x: x ** 2], I).value
    assert len(basis) == 3
    for i, ei in enumerate(basis):
        for j, ej in enumerate(basis):
            ip = functional.inner_product(ei, ej, I).value
            assert ip == pytest.approx(1.0 if i == j else 0.0, abs=1e-8)


def test_gram_schmidt_skips_dependent():
    r = functional.gram_schmidt([lambda x: 1.0, lambda x: 2.0], I)
    assert len(r.value) == 1


def test_projection_of_x_squared_onto_constants():
    e0 = functional.gram_schmidt([lambda x: 1.0], I).value
    p, coeffs = functional.project(lambda x: x ** 2, e0, I).value
    assert p(0.3) == pytest.approx(1 / 3)


def square(x):
    return 1.0 if x > 0 else -1.0


def test_square_wave_coefficients():
    a0, a, b = fourier.coefficients(square, np.pi, 5).value
    assert a0 == pytest.approx(0, abs=1e-8)
    assert b[0] == pytest.approx(4 / np.pi, rel=1e-5)
    assert b[1] == pytest.approx(0, abs=1e-6)
    assert b[2] == pytest.approx(4 / (3 * np.pi), rel=1e-5)


def test_parseval_gap_shrinks():
    g_small = abs(fourier.parseval(lambda x: x, np.pi, 5).value[2])
    g_big = abs(fourier.parseval(lambda x: x, np.pi, 60).value[2])
    assert g_big < g_small
    # for f(x)=x the tail of sum 4/n^2 behaves like 4/N
    assert g_big == pytest.approx(4 / 60, rel=0.1)


def test_gibbs_overshoot_near_9_percent_of_jump():
    # square wave jumps from -1 to 1 (jump = 2): peak -> 1 + 0.0895 * 2
    peak = fourier.gibbs_overshoot(square, np.pi, 40).value
    assert peak == pytest.approx(1.179, abs=0.01)


def test_explain_and_plot_run():
    r = fourier.partial_sum(lambda x: x, np.pi, 5, explain=True, plot=True)
    assert "S_N" in r.explain()
