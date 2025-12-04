import numpy as np

from adjungo.core.method import GLMethod
from adjungo.optimization.interface import GLMOptimizer
from adjungo.stepping.trajectory import Trajectory
from adjungo.stepping.adjoint import AdjointTrajectory


class DummyProblem:
    state_dim = 1
    control_dim = 1

    def f(self, y, u, t):
        return y + u

    def F(self, y, u, t):
        return np.ones((1, 1))

    def G(self, y, u, t):
        return np.ones((1, 1))

    def F_yy_action(self, y, u, t, v):
        return np.zeros((1, 1))

    def F_yu_action(self, y, u, t, v):
        return np.zeros((1, 1))

    def F_uu_action(self, y, u, t, v):
        return np.zeros((1, 1))


class DummyObjective:
    def evaluate(self, trajectory, u):
        return float(u.sum())

    def dJ_dy_terminal(self, y_final):
        return np.zeros_like(y_final)

    def dJ_dy(self, y, step):
        return np.zeros_like(y)

    def dJ_du(self, u_stage, step, stage):
        return u_stage

    def d2J_du2(self, u_stage, step, stage):
        return np.eye(u_stage.shape[-1])


def _dummy_method():
    return np.array([[0.0]]), np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]]), np.array([0.0])


def test_optimizer_caches_forward_and_adjoint(monkeypatch):
    A, U, B, V, c = _dummy_method()

    forward_calls = []
    adjoint_calls = []
    gradient_calls = []

    def fake_forward(y0, u, t_span, N, problem, method, stage_solver):
        forward_calls.append(u.copy())
        Y = np.zeros((N + 1, 1, 1))
        Z = np.zeros((N, 1, 1))
        return Trajectory(Y=Y, Z=Z, caches=[None] * N)

    def fake_adjoint(trajectory, objective, method, stage_solver, h):
        adjoint_calls.append(True)
        Lambda = np.zeros_like(trajectory.Y)
        Mu = np.zeros_like(trajectory.Z)
        WeightedAdj = np.zeros_like(trajectory.Z)
        return AdjointTrajectory(Lambda=Lambda, Mu=Mu, WeightedAdj=WeightedAdj)

    def fake_gradient(trajectory, adjoint, u, objective, method, problem, h):
        gradient_calls.append(True)
        return np.ones_like(u)

    def fake_requirements(method, structure, state_dim):
        return "requirements"

    class FakeStageSolver:
        pass

    def fake_solver_factory(method, requirements, structure):
        return FakeStageSolver()

    monkeypatch.setattr("adjungo.optimization.interface.deduce_requirements", fake_requirements)
    monkeypatch.setattr("adjungo.optimization.interface.create_stage_solver", fake_solver_factory)
    monkeypatch.setattr("adjungo.optimization.interface.forward_solve", fake_forward)
    monkeypatch.setattr("adjungo.optimization.interface.adjoint_solve", fake_adjoint)
    monkeypatch.setattr("adjungo.optimization.interface.assemble_gradient", fake_gradient)

    method = GLMethod(A=A, U=U, B=B, V=V, c=c)
    optimizer = GLMOptimizer(
        problem=DummyProblem(),
        objective=DummyObjective(),
        method=method,
        t_span=(0.0, 1.0),
        N=2,
        y0=np.array([0.0]),
    )

    u = np.zeros((2, 1, 1))

    value = optimizer.objective_value(u)
    assert value == 0.0
    assert len(forward_calls) == 1

    grad = optimizer.gradient(u)
    assert np.array_equal(grad, np.ones_like(u))
    assert len(forward_calls) == 1  # cached
    assert len(adjoint_calls) == 1
    assert len(gradient_calls) == 1

    # Calling again with same control should not trigger new solves
    optimizer.gradient(u)
    assert len(forward_calls) == 1
    assert len(adjoint_calls) == 1
    assert len(gradient_calls) == 2

