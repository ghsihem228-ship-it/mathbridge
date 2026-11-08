import matplotlib

matplotlib.use("Agg")

import math

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mathbridge import Result, functional


def test_powers_converge_in_l2_but_not_uniformly():
    ns = [1, 2, 4, 8, 16, 32, 64, 128]
    report = functional.convergence_report(
        lambda n: lambda x: x**n, lambda x: 0.0, (0, 1), ns,
    )
    errors = report.value
    assert set(errors) == {"pointwise", "sup", "l2"}
    assert all(isinstance(values, list) and len(values) == len(ns)
               for values in errors.values())
    np.testing.assert_allclose(errors["l2"], 1 / np.sqrt(2 * np.array(ns) + 1))
    assert errors["l2"][-1] < errors["l2"][0] / 8
    assert errors["sup"] == pytest.approx([1.0] * len(ns))
    points = np.linspace(0, 1, 5)
    np.testing.assert_allclose(errors["pointwise"], [points**n for n in ns])
    assert errors["pointwise"][-1][-1] == 1.0
    assert "L2: appears to converge to zero" in report.explain()
    assert "uniform convergence): does not appear to converge" in report.explain()
    assert "Pointwise at x = 1: error does not appear to converge" in report.explain()


def test_scaled_sines_converge_in_all_three_modes():
    ns = [1, 2, 4, 8, 16, 32, 64, 128]
    report = functional.convergence_report(
        lambda n: lambda x: math.sin(n * x) / n, lambda x: 0.0, (0, 1), ns,
    )
    errors = report.value
    n = np.array(ns)
    exact_l2 = np.sqrt(0.5 - np.sin(2 * n) / (4 * n)) / n
    np.testing.assert_allclose(errors["l2"], exact_l2, rtol=1e-7)
    assert np.all(np.asarray(errors["sup"]) <= 1 / n + 1e-12)
    for mode in ("sup", "l2"):
        assert errors[mode][-1] < errors[mode][0] / 50
    assert max(errors["pointwise"][-1]) < max(errors["pointwise"][0]) / 50
    assert "Pointwise at all sample points: appears to converge" in report.explain()
    assert "uniform convergence): appears to converge" in report.explain()
    assert "L2: appears to converge" in report.explain()


def test_explain_plot_and_exact_zero_errors(capsys):
    try:
        report = functional.convergence_report(
            lambda n: lambda x: x, lambda x: x, (-1, 1), [1, 2, 4],
            explain=True, plot=True,
        )
        assert isinstance(report, Result)
        assert report.explain() in capsys.readouterr().out
        assert report.value["l2"] == [0.0, 0.0, 0.0]
        assert report.value["sup"] == [0.0, 0.0, 0.0]
        assert report.value["pointwise"] == [[0.0] * 5] * 3
        assert "L2: appears to converge" in report.explain()
        assert plt.get_fignums()
        fig = report.plot()
        assert fig.axes[0].get_yscale() == "log"
        assert len(fig.axes[0].lines) == 3
    finally:
        plt.close("all")


def test_single_nonzero_observation_is_not_a_trend():
    report = functional.convergence_report(
        lambda n: lambda x: 1.0, lambda x: 0.0, (0, 1), [1],
    )
    assert "L2: does not appear to converge" in report.explain()


@pytest.mark.parametrize("indices", [[], [2, 1], [1, 1], [1.5, 2.5]])
def test_invalid_indices(indices):
    with pytest.raises(ValueError, match="n_values"):
        functional.convergence_report(lambda n: lambda x: x, lambda x: x, (0, 1), indices)


@pytest.mark.parametrize("interval", [(1, 0), (1, 1), (0, np.inf)])
def test_invalid_interval(interval):
    with pytest.raises(ValueError, match="interval"):
        functional.convergence_report(lambda n: lambda x: x, lambda x: x, interval, [1, 2])
