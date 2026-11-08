import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import sympy as sp

from mathbridge import Result, geometry


@pytest.mark.parametrize("symbolic", [False, True])
@pytest.mark.parametrize("radius", [0.4, 2.3])
def test_circle_curvature_and_torsion(symbolic, radius):
    s = sp.Symbol("s", real=True)
    r = (sp.Matrix([radius * sp.cos(s), radius * sp.sin(s), 0]) if symbolic
         else lambda t: np.array([radius * np.cos(t), radius * np.sin(t), 0]))
    ts = np.array([-1.2, 0, 0.7, 2.1])
    kappa = geometry.curvature(r, ts).value
    tau = geometry.torsion(r, ts).value
    assert isinstance(kappa, np.ndarray)
    assert kappa.shape == ts.shape
    np.testing.assert_allclose(kappa, 1 / radius, rtol=1e-7)
    np.testing.assert_allclose(tau, 0, atol=1e-8)


@pytest.mark.parametrize("symbolic", [False, True])
@pytest.mark.parametrize("b", [-0.6, 0.6])
def test_helix_geometry_and_frame(symbolic, b):
    a = 1.7
    s = sp.Symbol("s", real=True)
    r = ([a * sp.cos(s), a * sp.sin(s), b * s] if symbolic
         else lambda t: np.array([a * np.cos(t), a * np.sin(t), b * t]))
    ts = np.array([[-1.2, 0], [0.7, 2.1]])
    np.testing.assert_allclose(geometry.curvature(r, ts).value, a / (a*a + b*b), rtol=1e-7)
    np.testing.assert_allclose(geometry.torsion(r, ts).value, b / (a*a + b*b), rtol=1e-6)
    frame = geometry.frenet_frame(r, ts).value
    assert frame.shape == ts.shape + (3, 3)
    tangent = np.stack((-a * np.sin(ts), a * np.cos(ts), np.full_like(ts, b)), axis=-1)
    tangent /= np.sqrt(a*a + b*b)
    normal = np.stack((-np.cos(ts), -np.sin(ts), np.zeros_like(ts)), axis=-1)
    binormal = np.cross(tangent, normal)
    np.testing.assert_allclose(frame, np.stack((tangent, normal, binormal), axis=-2), atol=1e-8)
    np.testing.assert_allclose(frame @ np.swapaxes(frame, -1, -2),
                               np.broadcast_to(np.eye(3), ts.shape + (3, 3)), atol=1e-10)
    np.testing.assert_allclose(np.cross(frame[..., 0, :], frame[..., 1, :]), frame[..., 2, :], atol=1e-10)


def test_symbolic_nonuniform_parametrization():
    s = sp.Symbol("s", real=True)
    r = sp.Matrix([2 * sp.cos(s**2), 2 * sp.sin(s**2), 0])
    np.testing.assert_allclose(geometry.curvature(r, [0.4, 1.2, 2.0]).value, 0.5, atol=1e-12)


@pytest.mark.parametrize("solver", [geometry.curvature, geometry.torsion, geometry.frenet_frame])
@pytest.mark.parametrize("symbolic", [False, True])
def test_result_explain_and_frenet_plot(solver, symbolic, capsys):
    s = sp.Symbol("s", real=True)
    r = (sp.Matrix([sp.cos(s), sp.sin(s), s]) if symbolic
         else lambda t: [np.cos(t), np.sin(t), t])
    try:
        result = solver(r, 0.4, explain=True, plot=True, plot_interval=(-1, 2))
        assert isinstance(result, Result)
        assert isinstance(result.value, np.ndarray)
        assert result.value.shape == ((3, 3) if solver is geometry.frenet_frame else ())
        assert result.explain() in capsys.readouterr().out
        assert ("Exact symbolic derivatives" if symbolic else "Numerical derivatives") in result.explain()
        assert plt.get_fignums()
        fig = result.plot()
        assert fig.axes[0].name == "3d"
        assert fig.axes[0].get_legend_handles_labels()[1] == ["r(t)", "T", "N", "B"]
    finally:
        plt.close("all")


def test_zero_curvature_and_stationary_points():
    s = sp.Symbol("s", real=True)
    line = [s, 0, 0]
    assert geometry.curvature(line, 1).value == 0
    for solver in [geometry.torsion, geometry.frenet_frame]:
        with pytest.raises(ValueError, match="nonzero curvature"):
            solver(line, 1)
    for solver in [geometry.curvature, geometry.torsion, geometry.frenet_frame]:
        with pytest.raises(ValueError, match="nonzero speed"):
            solver([s**2, s**3, 0], 0)


def test_numerical_step_and_input_validation():
    r = lambda t: [np.cos(t), np.sin(t), t]
    assert geometry.torsion(r, 0.5, h=0.01).value == pytest.approx(0.5, rel=1e-6)
    with pytest.raises(ValueError, match="h must"):
        geometry.curvature(r, 0, h=0)
    for t in [[], [np.nan]]:
        with pytest.raises(ValueError, match="parameter values"):
            geometry.curvature(r, t)
    with pytest.raises(ValueError, match="three finite"):
        geometry.curvature(lambda t: [t, t], 0)
    s, a = sp.symbols("s a")
    with pytest.raises(ValueError, match="substitute"):
        geometry.curvature([a * sp.cos(s), a * sp.sin(s), s], 0)
