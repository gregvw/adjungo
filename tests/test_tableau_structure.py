"""Tableau structure is exact, and a declared stage type is checked (C-8.3).

Two defects motivate this module, and they need different discriminators:

* A DIRK whose diagonal entries agreed to about ``1e-5`` relative was
  classified SDIRK by a tolerance and solved with ``A[0,0]`` at every stage.
  That integrates a different method, and the independent reference sees it.
* A stage whose diagonal entry was below the old ``1e-8`` absolute tolerance
  was treated as explicit and its implicit term dropped. On the fixture here
  that moves the gradient by only ``6.8e-12`` relative, below the certified
  tolerance, so no accuracy test can see it; the factorization count can.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import scipy.linalg

from adjungo import GLMOptimizer, TableauDeclarationError
from adjungo.core.method import GLMethod, PropType, StageType
from adjungo.core.problem import Linearity
from adjungo.methods import runge_kutta
from adjungo.methods.glm import create_custom_glm
from adjungo.methods.runge_kutta import rk4, sdirk2
from adjungo.solvers.dirk import DIRKStageSolver
from adjungo.solvers.explicit import ExplicitStageSolver
from adjungo.solvers.implicit import ImplicitStageSolver
from adjungo.solvers.sdirk import SDIRKStageSolver
from adjungo.utils.kronecker import block_solve, kronecker_eye
from adjungo.validation import reference_gradient, reference_hessian
from tests.problems import FullCostObjective, LinearTimeVarying, make_controls

T_SPAN = (0.35, 1.55)
N_STEPS = 12
Y0 = np.array([0.6, -0.4])

#: sdirk2's diagonal coefficient, as the library computes it.
GAMMA = 1.0 - 1.0 / np.sqrt(2.0)

# Basis (R-13), measured over this module's six reference comparisons.
#
# Rounding side: the package and the independent reference agree to at most
# 1.1e-16 relative for the gradient and 5.6e-17 for the Hessian-vector product.
# They are not the same computation done twice. The package solves each stage
# equation directly, one small LU per implicit stage. The reference forms the
# monolithic residual over the whole trajectory, 74 or 122 unknowns here, and
# applies Newton to it. Because f is affine in y, that converges in one
# iteration to a residual of 3.7e-16 to 1.4e-15.
#
# Defect side: the near-SDIRK defect below moved the gradient by 2.07e-9 and
# the Hessian-vector product by 3.6e-10.
#
# 1e-11 sits five orders above the rounding and more than an order below the
# defect.
REFERENCE_RTOL = 1e-11


class _DeclaredLinearTimeVarying(LinearTimeVarying):
    """``F = A(t)``, routed on the linear stage path so that the comparison
    with the reference is limited by LU round-off, not by a Newton stopping
    test."""

    linearity = Linearity.LINEAR


def _tableau(A: np.ndarray, **declaration) -> GLMethod:
    """A one-step method with weights taken from the last row of ``A``."""
    return GLMethod(
        A=A,
        U=np.ones((A.shape[0], 1)),
        B=A[-1:].copy(),
        V=np.eye(1),
        c=A.sum(axis=1),
        **declaration,
    )


def _near_sdirk() -> np.ndarray:
    """A DIRK the old tolerance classified SDIRK: the second diagonal entry
    differs from the first by ``1e-6`` relative."""
    return np.array([[GAMMA, 0.0], [1.0 - GAMMA, GAMMA * (1.0 + 1e-6)]])


def _tiny_diagonal() -> np.ndarray:
    """A DIRK whose first stage the old tolerance treated as explicit."""
    return np.array([[1e-9, 0.0], [0.5, 0.5]])


def _relative_error(got: np.ndarray, ref: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(ref))), 1.0)
    return float(np.max(np.abs(got - ref))) / scale


def _assert_matches_reference(method: GLMethod) -> None:
    problem = _DeclaredLinearTimeVarying()
    objective = FullCostObjective(nx=2, nu=2)
    optimizer = GLMOptimizer(problem, objective, method, T_SPAN, N_STEPS, Y0)
    u = make_controls(N_STEPS, method.s, 2, seed=7)
    v = make_controls(N_STEPS, method.s, 2, seed=8)

    grad_ref = reference_gradient(
        Y0, u, T_SPAN, N_STEPS, problem, method, objective
    )
    hessian = reference_hessian(
        Y0, u, T_SPAN, N_STEPS, problem, method, objective
    )
    hvp_ref = (hessian.reshape(u.size, u.size) @ v.ravel()).reshape(u.shape)

    assert _relative_error(optimizer.gradient(u), grad_ref) < REFERENCE_RTOL
    assert (
        _relative_error(optimizer.hessian_vector_product(u, v), hvp_ref)
        < REFERENCE_RTOL
    )


# ---------------------------------------------------------------------------
# Exact structure
# ---------------------------------------------------------------------------


def test_a_near_sdirk_tableau_is_a_dirk() -> None:
    method = _tableau(_near_sdirk())
    assert method.structural_stage_type is StageType.DIRK
    assert method.stage_type is StageType.DIRK
    assert method.sdirk_gamma is None


def test_a_near_sdirk_tableau_is_solved_as_written() -> None:
    """The discriminating case for the first defect. Routed as SDIRK, the
    second stage was solved with ``A[0,0]`` and the gradient disagreed with
    the reference by 2.07e-9."""
    _assert_matches_reference(_tableau(_near_sdirk()))


def test_a_tiny_diagonal_entry_is_implicit() -> None:
    """The second defect is invisible to the reference comparison at this
    scale, so the certified quantities are structural: the stage is not
    listed as explicit, and it takes a factorization at every step."""
    method = _tableau(_tiny_diagonal())
    assert method.explicit_stage_indices == []
    assert method.structural_stage_type is StageType.DIRK

    optimizer = GLMOptimizer(
        _DeclaredLinearTimeVarying(),
        FullCostObjective(nx=2, nu=2),
        method,
        T_SPAN,
        N_STEPS,
        Y0,
    )
    store = optimizer.stage_solver.factorizations
    store.reset_counts()
    optimizer.gradient(make_controls(N_STEPS, method.s, 2, seed=7))
    assert store.factorizations == 2 * N_STEPS

    _assert_matches_reference(method)


def test_the_tangent_sweep_treats_a_tiny_diagonal_entry_as_implicit(
    monkeypatch,
) -> None:
    """The tangent sweep decides explicit stages for itself, and at this
    scale a wrong decision is below every accuracy tolerance too. So the
    certified quantity is structural again: with the trajectory and adjoint
    already cached, a Hessian-vector product solves untransposed only in the
    tangent sweep, once per implicit stage per step."""
    method = _tableau(_tiny_diagonal())
    optimizer = GLMOptimizer(
        _DeclaredLinearTimeVarying(),
        FullCostObjective(nx=2, nu=2),
        method,
        T_SPAN,
        N_STEPS,
        Y0,
    )
    u = make_controls(N_STEPS, method.s, 2, seed=7)
    v = make_controls(N_STEPS, method.s, 2, seed=8)
    optimizer.gradient(u)

    untransposed = []
    original = scipy.linalg.lu_solve

    def counting(lu_and_piv, b, trans=0, **kwargs):
        if trans == 0:
            untransposed.append(1)
        return original(lu_and_piv, b, trans=trans, **kwargs)

    monkeypatch.setattr(scipy.linalg, "lu_solve", counting)
    optimizer.hessian_vector_product(u, v)
    assert len(untransposed) == 2 * N_STEPS


@pytest.mark.parametrize(
    "A, expected",
    [
        (np.array([[0.0, 1e-17], [0.5, 0.0]]), StageType.IMPLICIT),
        (np.array([[1e-300, 0.0], [0.5, 0.0]]), StageType.DIRK),
        (np.array([[0.0, 0.0], [0.5, 0.0]]), StageType.EXPLICIT),
        (np.array([[0.3, 0.0], [0.5, 0.3]]), StageType.SDIRK),
        (np.array([[0.0, 0.0], [0.5, 0.3]]), StageType.DIRK),
    ],
    ids=["stray-upper", "denormal-diagonal", "explicit", "sdirk", "esdirk"],
)
def test_structure_is_decided_by_exact_zeros(A, expected) -> None:
    assert _tableau(A).structural_stage_type is expected


def test_propagation_structure_is_exact() -> None:
    A = np.array([[0.0, 0.0], [0.5, 0.0]])
    kwargs = {"A": A, "U": np.eye(2), "B": np.eye(2), "c": np.array([0.0, 0.5])}
    assert GLMethod(V=np.eye(2), **kwargs).prop_type is PropType.IDENTITY
    nearly_identity = np.eye(2)
    nearly_identity[0, 1] = 1e-17
    assert GLMethod(V=nearly_identity, **kwargs).prop_type is PropType.DENSE


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


def _library_tableaux() -> list:
    return [
        pytest.param(factory, id=name)
        for name, factory in inspect.getmembers(runge_kutta, inspect.isfunction)
        if factory.__module__ == runge_kutta.__name__ and not name.startswith("_")
    ]


@pytest.mark.parametrize("factory", _library_tableaux())
def test_library_tableaux_declare_their_exact_structure(factory) -> None:
    """Every factory in ``methods/runge_kutta.py`` declares its class, and the
    class it declares is the most specific one its coefficients satisfy.
    Construction has already checked admission; this asserts the declaration
    is not merely general. The retained experimental tableaux are left
    undeclared (C-8.3)."""
    method = factory()
    assert method.declared_stage_type is not None
    assert method.declared_stage_type is method.structural_stage_type


@pytest.mark.parametrize(
    "A, declared, reported",
    [
        (_near_sdirk(), StageType.SDIRK, ["diagonal differs", "A[1,1] = "]),
        (
            np.array([[0.3, 1e-17], [0.5, 0.3]]),
            StageType.DIRK,
            ["above the diagonal", "A[0,1] = 1e-17"],
        ),
        (
            np.array([[0.3, 0.0], [0.5, 0.3]]),
            StageType.EXPLICIT,
            ["on the diagonal", "A[0,0] = 0.3", "A[1,1] = 0.3"],
        ),
        (
            np.array([[0.0, 0.0], [0.5, 0.3]]),
            StageType.SDIRK,
            ["zero on the diagonal", "A[0,0] = 0.0"],
        ),
    ],
    ids=["sdirk-unequal-diagonal", "dirk-stray-upper", "explicit-with-diagonal",
         "sdirk-zero-diagonal"],
)
def test_a_narrower_declaration_is_refused_and_names_the_entries(
    A, declared, reported
) -> None:
    with pytest.raises(TableauDeclarationError) as excinfo:
        _tableau(A, declared_stage_type=declared)
    message = str(excinfo.value)
    assert declared.name in message
    assert "C-8.3" in message
    for fragment in reported:
        assert fragment in message, (fragment, message)


def test_the_refusal_is_a_value_error() -> None:
    assert issubclass(TableauDeclarationError, ValueError)


def test_a_declaration_must_be_a_stage_type() -> None:
    with pytest.raises(TypeError):
        _tableau(np.array([[0.3, 0.0], [0.5, 0.3]]), declared_stage_type="SDIRK")


@pytest.mark.parametrize(
    "factory, declared, solver_class",
    [
        (rk4, StageType.DIRK, DIRKStageSolver),
        (rk4, StageType.IMPLICIT, ImplicitStageSolver),
        (sdirk2, StageType.DIRK, DIRKStageSolver),
        (sdirk2, StageType.IMPLICIT, ImplicitStageSolver),
    ],
    ids=["explicit-as-dirk", "explicit-as-coupled", "sdirk-as-dirk",
         "sdirk-as-coupled"],
)
def test_a_more_general_declaration_routes_by_the_declaration(
    factory, declared, solver_class
) -> None:
    """Declaring a class more general than the structure is admitted and
    selects that class's solver, which is correct for the narrower
    structure too. Its derivatives must still be those of the method."""
    natural = factory()
    method = GLMethod(
        A=natural.A,
        U=natural.U,
        B=natural.B,
        V=natural.V,
        c=natural.c,
        declared_stage_type=declared,
    )
    assert method.stage_type is declared
    assert method.structural_stage_type is natural.structural_stage_type

    optimizer = GLMOptimizer(
        _DeclaredLinearTimeVarying(),
        FullCostObjective(nx=2, nu=2),
        method,
        T_SPAN,
        N_STEPS,
        Y0,
    )
    assert isinstance(optimizer.stage_solver, solver_class)
    _assert_matches_reference(method)


def test_the_natural_routes_are_unchanged() -> None:
    assert isinstance(
        GLMOptimizer(
            _DeclaredLinearTimeVarying(),
            FullCostObjective(nx=2, nu=2),
            rk4(),
            T_SPAN,
            N_STEPS,
            Y0,
        ).stage_solver,
        ExplicitStageSolver,
    )
    assert isinstance(
        GLMOptimizer(
            _DeclaredLinearTimeVarying(),
            FullCostObjective(nx=2, nu=2),
            sdirk2(),
            T_SPAN,
            N_STEPS,
            Y0,
        ).stage_solver,
        SDIRKStageSolver,
    )


def test_create_custom_glm_checks_the_declaration() -> None:
    A = _near_sdirk()
    kwargs = {
        "U": np.ones((2, 1)),
        "B": A[-1:].copy(),
        "V": np.eye(1),
        "c": A.sum(axis=1),
    }
    assert (
        create_custom_glm(A, declared_stage_type=StageType.DIRK, **kwargs).stage_type
        is StageType.DIRK
    )
    with pytest.raises(TableauDeclarationError):
        create_custom_glm(A, declared_stage_type=StageType.SDIRK, **kwargs)


def test_block_solve_decides_triangularity_exactly() -> None:
    """``block_solve`` chooses substitution from the structure of ``A``. An
    entry of ``1e-9`` above the diagonal was below the old tolerance, so the
    system was solved as if it were absent. Basis: the dense solve of this
    O(1) system is exact to a few ULPs, and ignoring the entry moves the
    solution by about ``1e-9``; ``1e-12`` sits three orders from each."""
    A = np.array([[1.0, 1e-9], [0.5, 2.0]])
    n = 3
    b = np.linspace(-1.0, 1.0, 2 * n)
    expected = np.linalg.solve(kronecker_eye(A, n), b)
    assert np.max(np.abs(block_solve(A, b, 2, n) - expected)) < 1e-12
