"""Factorization reuse, demonstrated by counting rather than timing.

Run directly::

    .venv/bin/python examples/factorization_reuse_counts.py

This example is executed by ``tests/test_examples.py``.

Why this example exists
-----------------------

Reuse is the one optimization no accuracy test can check. Refactoring the same
matrix produces the same answer, so a solver that silently stopped reusing
would still be correct, and a solver that wrongly reused a *stale* matrix would
be wrong in the adjoint while the forward solve still converged. That second
failure is precedent R-9 in ``NUMERICS.md``: a probe that varied only ``u`` and
``t`` concluded the Jacobian was constant, the adjoint was handed another
stage's matrix, and the gradient was wrong by 4.24e-05 relative while the whole
test suite passed.

So reuse is **counted, not timed**, and it is **declared, not inferred**. This
example shows both halves:

1. With ``ProblemStructure(jacobian_constant=True)`` declared, the entire solve
   takes one factorization, independent of ``N``.
2. Without the declaration, the same problem — unchanged, still genuinely
   constant — takes one factorization per stage per Newton iteration per step.
   The library does not go looking for structure the caller did not state.

A declaration that is *false* is not accepted on trust either: before any
stored factorization is returned, the matrix it was taken from is compared with
the matrix now being asked for, element for element, and a mismatch raises.

Problem
-------

A linear time-invariant system, which is the simplest setting in which
``jacobian_constant=True`` is truthful:

.. math::

    \\dot{y} = A y + B u

with ``A`` fixed. ``df/dy = A`` at every state, stage and step, so one LU
factorization of ``I - h*gamma*A`` serves the whole solve.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from adjungo import GLMOptimizer, ProblemStructure
from adjungo.core.problem import Linearity
from adjungo.core.requirements import deduce_requirements
from adjungo.methods.runge_kutta import sdirk3

T_FINAL = 1.0
N_STEPS = 8
Y0 = np.array([1.0, -0.5])

A_MATRIX = np.array([[-2.0, 1.0], [0.5, -3.0]])
B_MATRIX = np.array([[1.0], [0.0]])


class LinearSystem:
    """``dy/dt = A y + B u`` with constant ``A``."""

    state_dim = 2
    control_dim = 1

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return A_MATRIX @ y + B_MATRIX @ u

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        """``df/dy = A``: the same matrix at every argument. That is the fact
        the declaration below asserts, and that the store re-checks."""
        return A_MATRIX

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return B_MATRIX

    def F_yy_action(self, y, u, t, w) -> NDArray:
        return np.zeros((2, 2))

    def F_yu_action(self, y, u, t, w) -> NDArray:
        return np.zeros((2, 1))

    def F_uu_action(self, y, u, t, w) -> NDArray:
        return np.zeros((1, 1))


class TrackingObjective:
    """Terminal tracking plus control energy."""

    def evaluate(self, trajectory, u: NDArray) -> float:
        y_final = trajectory.Y[-1][0]
        return float(0.5 * np.dot(y_final, y_final) + 0.005 * np.sum(u**2))

    def dJ_dy_terminal(self, y_final: NDArray) -> NDArray:
        return y_final

    def dJ_dy(self, y: NDArray, step: int) -> NDArray:
        return np.zeros(2)

    def dJ_du(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        return 0.01 * u_stage

    def d2J_du2(self, u_stage: NDArray, step: int, stage: int) -> NDArray:
        return 0.01 * np.eye(1)

    def d2J_dy2(self, y: NDArray, step: int) -> NDArray:
        return np.zeros((2, 2))

    def d2J_dy2_terminal(self, y_final: NDArray) -> NDArray:
        return np.eye(2)


#: The declaration that permits reuse. ``jacobian_constant`` is the only field
#: that gates it; the others describe the problem for dispatch.
DECLARED_CONSTANT = ProblemStructure(
    linearity=Linearity.SEMILINEAR,
    jacobian_constant=True,
    jacobian_control_dependent=False,
    has_second_derivatives=True,
)

#: The same problem, described without the constancy claim.
UNDECLARED = ProblemStructure(
    linearity=Linearity.NONLINEAR,
    jacobian_constant=False,
    jacobian_control_dependent=True,
    has_second_derivatives=True,
)


def build_optimizer(
    structure: ProblemStructure, n_steps: int = N_STEPS
) -> GLMOptimizer:
    return GLMOptimizer(
        problem=LinearSystem(),
        objective=TrackingObjective(),
        method=sdirk3(),
        t_span=(0.0, T_FINAL),
        N=n_steps,
        y0=Y0,
        problem_structure=structure,
    )


def count_factorizations(
    structure: ProblemStructure, n_steps: int = N_STEPS
) -> tuple[int, int, NDArray]:
    """Return ``(factorizations, reuses, gradient)`` for one gradient solve."""
    optimizer = build_optimizer(structure, n_steps)
    store = optimizer.stage_solver.factorizations
    store.reset_counts()

    u = np.zeros((n_steps, sdirk3().s, 1))
    gradient = optimizer.gradient(u)
    return store.factorizations, store.reuses, gradient


def main() -> None:
    method = sdirk3()

    print("Declaration-gated factorization reuse (sdirk3, constant Jacobian)")
    print("=" * 66)
    print(f"  steps N        : {N_STEPS}")
    print(f"  stages s       : {method.s}")
    print()

    predicted = deduce_requirements(
        method, DECLARED_CONSTANT, LinearSystem.state_dim
    ).factorizations_for_solve(N_STEPS)

    declared_count, declared_reuses, declared_grad = count_factorizations(
        DECLARED_CONSTANT
    )
    undeclared_count, undeclared_reuses, undeclared_grad = count_factorizations(
        UNDECLARED
    )

    print("  declared jacobian_constant=True")
    print(f"    predicted factorizations : {predicted}")
    print(f"    observed  factorizations : {declared_count}")
    print(f"    reuses                   : {declared_reuses}")
    print()
    print("  same problem, nothing declared")
    print(f"    observed  factorizations : {undeclared_count}")
    print(f"    reuses                   : {undeclared_reuses}")
    print()

    discrepancy = float(np.max(np.abs(declared_grad - undeclared_grad)))
    print(f"  max |gradient difference|  : {discrepancy:.3e}")
    print()
    print("Reuse changes the amount of work, never the answer. That is exactly")
    print("why it is asserted by counting: no accuracy test can detect either")
    print("its absence or, more dangerously, its misuse. See NUMERICS.md C-15")
    print("and precedent R-9.")


if __name__ == "__main__":
    main()
