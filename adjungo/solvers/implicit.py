"""Fully implicit stage solver.

NOT YET IMPLEMENTED. This class exists as an explicit refusal (NUMERICS.md
C-6.2) and is certified in milestone M3.

History, recorded as NUMERICS.md precedent R-4: both methods previously returned
zero arrays as placeholders. A Gauss-2 forward solve therefore returned the
initial condition unchanged, with no exception and no warning, and the caller
had no way to distinguish that from a correct result. Under C-7 a silent
sentinel is a contract violation rather than an incomplete feature, so the
bodies were deleted rather than left reachable.
"""

from numpy.typing import NDArray

from adjungo.core.method import GLMethod
from adjungo.core.problem import Problem
from adjungo.solvers.base import StageSolver, StepCache

_MESSAGE = (
    "Fully implicit stage solves are not implemented (NUMERICS.md C-6.1; "
    "scheduled for milestone M3). A tableau with a dense stage matrix A "
    "requires a coupled (s*n) x (s*n) Newton solve. Use an explicit, DIRK, or "
    "SDIRK method instead."
)


class ImplicitStageSolver(StageSolver):
    """Refuses fully implicit tableaux until milestone M3 certifies them."""

    def __init__(self) -> None:
        raise NotImplementedError(_MESSAGE)

    def solve_stages(
        self,
        y_history: NDArray,
        u_stages: NDArray,
        t_n: float,
        h: float,
        problem: Problem,
        method: GLMethod,
    ) -> tuple[NDArray, StepCache]:
        """Unreachable: construction raises."""
        raise NotImplementedError(_MESSAGE)

    def solve_adjoint_stages(
        self,
        lambda_ext: NDArray,
        cache: StepCache,
        method: GLMethod,
        h: float,
    ) -> NDArray:
        """Unreachable: construction raises."""
        raise NotImplementedError(_MESSAGE)
