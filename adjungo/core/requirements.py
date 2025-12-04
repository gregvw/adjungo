"""Solver requirements deduction."""

from dataclasses import dataclass
from adjungo.core.method import GLMethod, StageType
from adjungo.core.problem import ProblemStructure, Linearity


@dataclass
class SolverRequirements:
    """What computational primitives are needed per time step."""

    # Stage solve requirements
    needs_newton: bool
    newton_system_size: int  # n for DIRK, ns for fully implicit

    # Factorization strategy
    factorizations_per_step: int
    factorization_size: int
    can_reuse_across_stages: bool   # SDIRK with constant Jacobian
    can_reuse_across_steps: bool    # Linear problem

    # What to store for optimization
    store_jacobians: bool
    store_stage_values: bool
    trajectory_vectors_per_step: int  # ns + nr for (Z, y)


def deduce_requirements(
    method: GLMethod,
    problem: ProblemStructure,
    state_dim: int,
) -> SolverRequirements:
    """Dispatch logic from linalg_requirements.tex."""

    is_explicit = method.stage_type == StageType.EXPLICIT
    is_linear = problem.linearity in (
        Linearity.LINEAR,
        Linearity.BILINEAR,
        Linearity.QUASILINEAR,
    )

    if is_explicit:
        return SolverRequirements(
            needs_newton=False,
            newton_system_size=0,
            factorizations_per_step=0,
            factorization_size=0,
            can_reuse_across_stages=False,
            can_reuse_across_steps=False,
            store_jacobians=True,  # Still need F, G for adjoints
            store_stage_values=True,
            trajectory_vectors_per_step=method.s + method.r,
        )

    needs_newton = not is_linear
    n = state_dim

    if method.stage_type == StageType.SDIRK:
        # Key optimization: single factorization if Jacobian constant
        can_reuse_stages = (
            problem.jacobian_constant or not problem.jacobian_control_dependent
        )
        return SolverRequirements(
            needs_newton=needs_newton,
            newton_system_size=n,
            factorizations_per_step=1 if can_reuse_stages else method.s,
            factorization_size=n,
            can_reuse_across_stages=can_reuse_stages,
            can_reuse_across_steps=problem.jacobian_constant,
            store_jacobians=True,
            store_stage_values=True,
            trajectory_vectors_per_step=method.s + method.r,
        )

    if method.stage_type == StageType.DIRK:
        return SolverRequirements(
            needs_newton=needs_newton,
            newton_system_size=n,
            factorizations_per_step=method.s,
            factorization_size=n,
            can_reuse_across_stages=False,
            can_reuse_across_steps=problem.jacobian_constant,
            store_jacobians=True,
            store_stage_values=True,
            trajectory_vectors_per_step=method.s + method.r,
        )

    # Fully implicit
    return SolverRequirements(
        needs_newton=needs_newton,
        newton_system_size=n * method.s,
        factorizations_per_step=1,
        factorization_size=n * method.s,
        can_reuse_across_stages=True,
        can_reuse_across_steps=problem.jacobian_constant,
        store_jacobians=True,
        store_stage_values=True,
        trajectory_vectors_per_step=method.s + method.r,
    )
