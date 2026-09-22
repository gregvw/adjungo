"""Envelope enforcement: unsupported configurations must be refused.

NUMERICS.md C-6.2 requires hard guards at construction, with no override. The
failure mode this prevents is the one that produced precedent R-4: a placeholder
returned zeros, a Gauss-2 forward solve returned the initial condition unchanged,
and nothing distinguished that from a correct answer.
"""

import numpy as np
import pytest

from adjungo.core.method import GLMethod
from adjungo.core.problem import Linearity, ProblemStructure
from adjungo.methods.experimental.multistep import bdf2, bdf3
from adjungo.methods.runge_kutta import explicit_euler, gauss2, rk4
from adjungo.optimization.interface import GLMOptimizer


class _Decay:
    """y' = -y + u."""

    state_dim = 1
    control_dim = 1

    def f(self, y, u, t):
        return -y + u

    def F(self, y, u, t):
        return np.array([[-1.0]])

    def G(self, y, u, t):
        return np.array([[1.0]])


class _Terminal:
    def evaluate(self, trajectory, u):
        return float(0.5 * trajectory.Y[-1, 0, 0] ** 2)

    def dJ_dy_terminal(self, y):
        return y

    def dJ_dy(self, y, t):
        return np.zeros_like(y)

    def dJ_du(self, z, u, t):
        return np.zeros_like(u)


def _build(method, **kwargs):
    params = {
        "problem": _Decay(),
        "objective": _Terminal(),
        "method": method,
        "t_span": (0.0, 1.0),
        "N": 10,
        "y0": np.array([1.0]),
    }
    params.update(kwargs)
    return GLMOptimizer(**params)


@pytest.mark.parametrize("factory", [bdf2, bdf3])
def test_multistep_is_refused(factory):
    """C-6.2: r > 1 has no starting procedure and must be refused."""
    with pytest.raises(NotImplementedError, match=r"r > 1"):
        _build(factory())


def test_fully_implicit_is_refused():
    """C-6.1: the fully implicit family is not certified until M3."""
    with pytest.raises(NotImplementedError, match=r"not certified"):
        _build(gauss2())


def test_fully_implicit_solver_cannot_be_constructed_directly():
    """The refusal must not be bypassable by building the solver directly.

    Precedent R-4: this class previously returned Z = 0 silently.
    """
    from adjungo.solvers.implicit import ImplicitStageSolver

    with pytest.raises(NotImplementedError, match=r"not implemented"):
        ImplicitStageSolver()


@pytest.mark.parametrize("bad_N", [0, -1])
def test_nonpositive_step_count_is_refused(bad_N):
    """C-12: N = 0 gives h = inf; refuse rather than produce it."""
    with pytest.raises(ValueError, match=r"N must be at least 1"):
        _build(rk4(), N=bad_N)


def test_degenerate_time_span_is_refused():
    """C-12: a zero-extent interval gives h = 0."""
    with pytest.raises(ValueError, match=r"nonzero extent"):
        _build(rk4(), t_span=(1.0, 1.0))


def test_certified_families_are_accepted():
    """The guard must not refuse what the contract certifies."""
    for method in (explicit_euler(), rk4()):
        optimizer = _build(method)
        u = np.zeros((10, method.s, 1))
        assert np.isfinite(optimizer.objective_value(u))


def test_malformed_tableau_is_refused_at_construction():
    """C-8.2: A must be square, else `s` is silently wrong."""
    with pytest.raises(ValueError, match=r"A must be square"):
        GLMethod(
            A=np.zeros((1, 2)),
            U=np.eye(2),
            B=np.ones((2, 1)),
            V=np.eye(2),
            c=np.array([1.0, 0.0]),
        )


def test_mismatched_tableau_blocks_are_refused():
    """C-8.2: U, B, c shapes must agree with s and r."""
    with pytest.raises(ValueError, match=r"GLMethod.U must have shape"):
        GLMethod(
            A=np.zeros((1, 1)),
            U=np.ones((1, 3)),
            B=np.ones((1, 1)),
            V=np.eye(1),
            c=np.array([0.0]),
        )


def test_problem_structure_override_does_not_bypass_the_guard():
    """An explicit structure argument must not reopen a refused family."""
    structure = ProblemStructure(
        linearity=Linearity.LINEAR,
        jacobian_constant=True,
        jacobian_control_dependent=False,
        has_second_derivatives=False,
    )
    with pytest.raises(NotImplementedError):
        _build(gauss2(), problem_structure=structure)
