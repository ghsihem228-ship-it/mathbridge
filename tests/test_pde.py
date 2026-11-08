import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mathbridge import Result, pde


def test_heat_sine_mode_and_fd_agreement():
    L, alpha, T = 2.0, 0.3, 0.43
    f = lambda x: np.sin(np.pi * x / L)
    separation = pde.heat_separation(f, L, alpha, 5).value
    finite_difference = pde.heat_fd(f, L, alpha, 81, 0.0037, T).value
    xs = np.linspace(0, L, 103)[None, :]
    times = np.array([0, 0.017, 0.19, T])[:, None]
    exact = np.exp(-alpha * (np.pi / L)**2 * times) * np.sin(np.pi * xs / L)
    np.testing.assert_allclose(separation(xs, times), exact, atol=1e-10)
    np.testing.assert_allclose(finite_difference(xs, times), exact, atol=1e-2)
    np.testing.assert_allclose(finite_difference(xs, times), separation(xs, times), atol=1e-2)


def test_wave_sine_mode_and_fd_agreement():
    L, c, T = 2.0, 1.5, 0.83
    f = lambda x: np.sin(np.pi * x / L)
    g = lambda x: 0.0
    separation = pde.wave_separation(f, g, L, c, 5).value
    finite_difference = pde.wave_fd(f, g, L, c, 81, 0.0037, T).value
    xs = np.linspace(0, L, 103)[None, :]
    times = np.array([0, 0.017, 0.37, T])[:, None]
    exact = np.sin(np.pi * xs / L) * np.cos(c * np.pi * times / L)
    np.testing.assert_allclose(separation(xs, times), exact, atol=1e-10)
    np.testing.assert_allclose(finite_difference(xs, times), exact, atol=1e-2)
    np.testing.assert_allclose(finite_difference(xs, times), separation(xs, times), atol=1e-2)


def test_wave_nonzero_initial_velocity():
    L, c, T = 1.3, 0.7, 0.6
    omega = 2 * c * np.pi / L
    f = lambda x: 0.0
    g = lambda x: np.sin(2 * np.pi * x / L)
    separation = pde.wave_separation(f, g, L, c, 4).value
    finite_difference = pde.wave_fd(f, g, L, c, 101, 0.002, T).value
    xs = np.linspace(0, L, 107)
    for t in [0, 0.1, 0.32, T]:
        exact = np.sin(2 * np.pi * xs / L) * np.sin(omega * t) / omega
        np.testing.assert_allclose(separation(xs, t), exact, atol=1e-10)
        np.testing.assert_allclose(finite_difference(xs, t), exact, atol=1e-2)


def test_heat_multiple_modes():
    f = lambda x: np.sin(np.pi * x) + 0.4 * np.sin(3 * np.pi * x)
    separation = pde.heat_separation(f, 1, 0.2, 4).value
    finite_difference = pde.heat_fd(f, 1, 0.2, 101, 0.002, 0.3).value
    xs = np.linspace(0, 1, 103)
    for t in [0, 0.03, 0.11, 0.3]:
        exact = (np.exp(-0.2 * np.pi**2 * t) * np.sin(np.pi * xs)
                 + 0.4 * np.exp(-0.2 * (3 * np.pi)**2 * t) * np.sin(3 * np.pi * xs))
        np.testing.assert_allclose(separation(xs, t), exact, atol=1e-10)
        np.testing.assert_allclose(finite_difference(xs, t), exact, atol=1e-2)


@pytest.mark.parametrize("solver", [
    lambda **kw: pde.heat_separation(lambda x: np.sin(np.pi * x), 1, 0.2, 4, **kw),
    lambda **kw: pde.heat_fd(lambda x: np.sin(np.pi * x), 1, 0.2, 41, 0.003, 0.1, **kw),
    lambda **kw: pde.wave_separation(lambda x: np.sin(np.pi * x), lambda x: 0, 1, 1, 4, **kw),
    lambda **kw: pde.wave_fd(lambda x: np.sin(np.pi * x), lambda x: 0, 1, 1, 41, 0.003, 0.1, **kw),
])
def test_result_boundaries_coordinates_and_snapshots(solver, capsys):
    try:
        result = solver(explain=True, plot=True)
        assert isinstance(result, Result)
        assert result.explain() in capsys.readouterr().out
        assert isinstance(result.value(0.5, 0.02), float)
        np.testing.assert_array_equal(result.value([0, 1], 0.07), [0, 0])
        assert plt.get_fignums()
        fig = result.plot()
        assert len(fig.axes[0].lines) == 4
        for x, t in [(-0.1, 0), (1.1, 0), (0.5, -0.1), (np.nan, 0)]:
            with pytest.raises(ValueError, match="coordinates"):
                result.value(x, t)
    finally:
        plt.close("all")


def test_fd_single_step_and_minimal_spatial_grid():
    # One interior node exercises both ends of the tridiagonal system.
    heat = pde.heat_fd(lambda x: 1.0, 1, 1, 3, 1.0, 0.1).value
    assert heat(0.5, 0.1) == pytest.approx(3 / 7)
    # At one interior node, f=0 and g=2 give u(dt)=2*dt.
    wave = pde.wave_fd(lambda x: 0.0, lambda x: 2.0, 1, 1, 3, 1.0, 0.1).value
    assert wave(0.5, 0.1) == pytest.approx(0.2)
    for solution in [heat, wave]:
        with pytest.raises(ValueError, match="t <= T"):
            solution(0.5, 0.11)


def test_wave_fd_rejects_unstable_time_step():
    with pytest.raises(ValueError, match="CFL"):
        pde.wave_fd(lambda x: x, lambda x: 0, 1, 1, 101, 0.02, 1)


@pytest.mark.parametrize("kwargs", [
    {"L": 0}, {"alpha": -1}, {"nx": 2}, {"nx": 3.5}, {"dt": 0}, {"T": -1},
])
def test_heat_fd_invalid_parameters(kwargs):
    args = dict(f=lambda x: x, L=1, alpha=1, nx=21, dt=0.01, T=0.1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        pde.heat_fd(**args)


def test_separation_invalid_parameters():
    with pytest.raises(ValueError, match="n_terms"):
        pde.heat_separation(lambda x: x, 1, 1, 0)
    with pytest.raises(ValueError, match="c"):
        pde.wave_separation(lambda x: x, lambda x: 0, 1, 0, 3)
