import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mathbridge import Result, functional


def sine_basis(n):
    return [lambda x, k=k: np.sqrt(2) * np.sin(k * np.pi * x)
            for k in range(1, n + 1)]


def test_min_kernel_eigenvalues():
    # Integer-frequency sines are not eigenfunctions of this kernel. Use enough
    # of them for the Galerkin truncation error to fall below one percent.
    T = functional.integral_operator(lambda x, y: min(x, y), (0, 1)).value
    matrix = functional.galerkin_matrix(T, sine_basis(48), (0, 1)).value
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-7)
    values = functional.spectrum(matrix).value
    exact = 1 / (((np.arange(1, 4) - 0.5) * np.pi) ** 2)
    assert values[:3] == pytest.approx(exact, rel=1e-2)


def test_identity_galerkin_matrix():
    matrix = functional.galerkin_matrix(lambda f: f, sine_basis(5), (0, 1)).value
    np.testing.assert_allclose(matrix, np.eye(5), atol=1e-10)


def test_galerkin_row_and_column_orientation():
    # For basis [1, sqrt(3)*(2*x-1)], T(f)=integral f is a projection.
    # T(f)=x*integral f instead gives a nonsymmetric matrix with a known column.
    basis = [lambda x: 1.0, lambda x: np.sqrt(3) * (2 * x - 1)]
    mean = functional.integral_operator(lambda x, y: x, (0, 1)).value
    matrix = functional.galerkin_matrix(mean, basis, (0, 1)).value
    np.testing.assert_allclose(matrix, [[0.5, 0], [np.sqrt(3) / 6, 0]], atol=1e-10)


def test_integral_operator_action():
    T = functional.integral_operator(lambda x, y: min(x, y), (0, 1)).value
    f = T(lambda y: 1.0)
    for x in [0, 0.2, 0.7, 1]:
        assert f(x) == pytest.approx(x - x**2 / 2)


def test_self_adjoint_spectrum_sorted_by_magnitude():
    values = functional.spectrum(np.diag([2.0, -5.0, 1.0])).value
    np.testing.assert_allclose(values, [-5, 2, 1])


def test_general_spectrum_retains_complex_values():
    matrix = np.array([[0, -2, 0], [2, 0, 0], [0, 0, 0.5]])
    values = functional.spectrum(matrix, self_adjoint=False).value
    np.testing.assert_allclose(np.abs(values), [2, 2, 0.5])
    assert sorted(values[:2].imag) == pytest.approx([-2, 2])


def test_operator_norm_uses_singular_values():
    # All eigenvalues are zero, but the largest singular value is 3.
    assert functional.operator_norm([[0, 3], [0, 0]]).value == pytest.approx(3)
    assert functional.operator_norm([[3, 0, 0], [0, 4, 0]]).value == pytest.approx(4)


@pytest.mark.parametrize("make_result", [
    lambda: functional.galerkin_matrix(lambda f: f, sine_basis(2), (0, 1),
                                      explain=True, plot=True),
    lambda: functional.integral_operator(lambda x, y: x * y, (0, 1),
                                        explain=True, plot=True),
    lambda: functional.spectrum([[1, 0], [0, -2]], explain=True, plot=True),
    lambda: functional.operator_norm([[1, 0], [0, -2]], explain=True, plot=True),
])
def test_result_explain_and_plot(make_result, capsys):
    try:
        result = make_result()
        assert isinstance(result, Result)
        assert result.steps
        assert result.explain() in capsys.readouterr().out
        assert plt.get_fignums()
        assert isinstance(result.plot(), plt.Figure)
    finally:
        plt.close("all")
