"""General Linear Method specification."""

from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property

import numpy as np
from numpy.typing import NDArray


class StageType(Enum):
    """Classification of stage matrix structure."""
    EXPLICIT = auto()   # A strictly lower triangular
    DIRK = auto()       # A lower triangular, varying diagonal
    SDIRK = auto()      # A lower triangular, constant diagonal γ
    IMPLICIT = auto()   # A dense


class TableauDeclarationError(ValueError):
    """A declared stage type is contradicted by the tableau's exact structure.

    Raised at construction, never deferred to a solve. The declaration states
    the class the author meant; the coefficients are what will be integrated.
    When they disagree, one of them is wrong, and routing by either would
    silently integrate a method nobody chose: the declared class's solver would
    drop the entries that do not fit it, and the structural class would run a
    method the author did not write down. See NUMERICS.md C-8.3.
    """


class PropType(Enum):
    """Classification of propagation matrix structure."""
    IDENTITY = auto()   # V = I (RK)
    SHIFT = auto()      # V is shift matrix (Adams/BDF)
    TRIANGULAR = auto() # V lower triangular
    DENSE = auto()      # V general


@dataclass(frozen=True)
class GLMethod:
    """General Linear Method tableau."""

    A: NDArray  # (s, s) - internal stage coefficients
    U: NDArray  # (s, r) - history → stages
    B: NDArray  # (r, s) - stages → output
    V: NDArray  # (r, r) - history → output
    c: NDArray  # (s,)   - abscissae
    #: The stage type the author intends, checked against the exact structure
    #: of ``A`` at construction (C-8.3). It may name a class more general than
    #: the structure -- the DIRK solver handles explicit stages and a constant
    #: diagonal, the coupled solver handles any ``A`` -- but never a narrower
    #: one. ``None`` routes by the exact structure alone. Declaring is the
    #: recommended practice: it is the only way a coefficient typo that moves
    #: the tableau into another class is caught rather than integrated.
    declared_stage_type: StageType | None = None

    def __post_init__(self) -> None:
        """Enforce the NUMERICS.md C-8.2 shape invariants, then check any
        declared stage type against the exact structure (C-8.3).

        A tableau that stores per-history coefficients in the rows of ``A`` is
        malformed, not merely unconventional: ``s`` is read from ``A.shape[0]``,
        so such a tableau silently reports the wrong stage count instead of
        failing. Validating here makes that a construction-time error.
        """
        for name in ("A", "U", "B", "V", "c"):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"GLMethod.{name} must be a numpy array, got "
                    f"{type(value).__name__}."
                )

        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError(
                f"GLMethod.A must be square (s, s); got shape {self.A.shape}. "
                f"A tableau needing per-history coefficients is not "
                f"representable this way under C-8."
            )

        s = self.A.shape[0]

        if self.V.ndim != 2 or self.V.shape[0] != self.V.shape[1]:
            raise ValueError(
                f"GLMethod.V must be square (r, r); got shape {self.V.shape}."
            )

        r = self.V.shape[0]

        expected = {"U": (s, r), "B": (r, s), "c": (s,)}
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise ValueError(
                    f"GLMethod.{name} must have shape {shape} for s={s}, r={r}; "
                    f"got {actual}. See NUMERICS.md C-8.2."
                )

        declared = self.declared_stage_type
        if declared is not None:
            if not isinstance(declared, StageType):
                raise TypeError(
                    f"GLMethod.declared_stage_type must be a StageType or "
                    f"None, got {type(declared).__name__}."
                )
            violations = stage_structure_violations(self.A, declared)
            if violations:
                raise TableauDeclarationError(
                    f"GLMethod declares stage type {declared.name}, but the "
                    f"exact structure of A is "
                    f"{_classify_stage_structure(self.A).name}: "
                    + "; ".join(violations)
                    + ". A declaration may name a class more general than "
                    "the structure, never a narrower one. Correct the "
                    "coefficients or the declaration (NUMERICS.md C-8.3)."
                )

    @cached_property
    def s(self) -> int:
        """Number of internal stages."""
        return int(self.A.shape[0])

    @cached_property
    def r(self) -> int:
        """Number of external stages."""
        return int(self.V.shape[0])

    @cached_property
    def structural_stage_type(self) -> StageType:
        """The most specific class the exact structure of ``A`` admits."""
        return _classify_stage_structure(self.A)

    @cached_property
    def stage_type(self) -> StageType:
        """The class that selects the stage solver.

        The declaration when one was made, which construction has already
        checked against the structure; otherwise the structure itself.
        """
        if self.declared_stage_type is not None:
            return self.declared_stage_type
        return self.structural_stage_type

    @cached_property
    def prop_type(self) -> PropType:
        """Classify the propagation matrix structure."""
        return _classify_prop_structure(self.V)

    @cached_property
    def sdirk_gamma(self) -> float | None:
        """Return γ if SDIRK, otherwise None."""
        if self.stage_type == StageType.SDIRK:
            return float(self.A[0, 0])
        return None

    @cached_property
    def explicit_stage_indices(self) -> list[int]:
        """Stages i where a_{ii} is exactly zero (evaluated explicitly)."""
        return [i for i in range(self.s) if self.A[i, i] == 0.0]


#: Entries listed per violation before the report is truncated.
_MAX_REPORTED = 6


def _entry(A: NDArray, i: int, j: int) -> str:
    return f"A[{i},{j}] = {float(A[i, j])!r}"


def stage_structure_violations(A: NDArray, stage_type: StageType) -> list[str]:
    """The entries of ``A`` that keep it out of ``stage_type``, exactly.

    Zero means exactly zero and equal means exactly equal (C-8.3). A tolerance
    here would decide which method is integrated: a DIRK whose diagonal
    entries agree to ``1e-5`` relative was classified SDIRK and solved with
    ``A[0,0]`` at every stage. Exactness can only err toward a more general
    class, which is slower and still the method the coefficients describe.

    This one function answers both questions asked of a tableau -- which class
    its structure is in, and whether a declared class admits it -- so the two
    cannot drift apart. An empty list means ``A`` is in the class.
    """
    s = A.shape[0]
    upper = [(i, j) for i in range(s) for j in range(i + 1, s) if A[i, j] != 0.0]
    diagonal = [(i, i) for i in range(s) if A[i, i] != 0.0]

    def report(label: str, entries: list[tuple[int, int]]) -> list[str]:
        if not entries:
            return []
        shown = ", ".join(_entry(A, i, j) for i, j in entries[:_MAX_REPORTED])
        more = len(entries) - _MAX_REPORTED
        suffix = f" and {more} more" if more > 0 else ""
        return [f"{label}: {shown}{suffix}"]

    if stage_type is StageType.IMPLICIT:
        return []
    if stage_type is StageType.DIRK:
        return report("nonzero above the diagonal", upper)
    if stage_type is StageType.EXPLICIT:
        return report("nonzero above the diagonal", upper) + report(
            "nonzero on the diagonal", diagonal
        )
    # SDIRK: lower triangular with one nonzero value on the whole diagonal.
    zero_diagonal = [(i, i) for i in range(s) if A[i, i] == 0.0]
    unequal = [(i, i) for i in range(s) if A[i, i] != A[0, 0]]
    return (
        report("nonzero above the diagonal", upper)
        + report("zero on the diagonal", zero_diagonal)
        + report(
            f"diagonal differs from A[0,0] = {float(A[0, 0])!r}",
            [] if zero_diagonal else unequal,
        )
    )


def _classify_stage_structure(A: NDArray) -> StageType:
    """The most specific stage type whose exact structure ``A`` satisfies."""
    for candidate in (StageType.EXPLICIT, StageType.SDIRK, StageType.DIRK):
        if not stage_structure_violations(A, candidate):
            return candidate
    return StageType.IMPLICIT


def _classify_prop_structure(V: NDArray) -> PropType:
    """Classify propagation matrix structure, exactly (C-8.3)."""
    r = V.shape[0]

    if np.array_equal(V, np.eye(r)):
        return PropType.IDENTITY

    if r > 1 and np.array_equal(V, np.eye(r, k=1)):
        return PropType.SHIFT

    if not np.any(np.triu(V, 1)):
        return PropType.TRIANGULAR

    return PropType.DENSE
