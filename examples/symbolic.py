"""Derive the callbacks the ``Problem`` protocol requires from sympy expressions.

Run the examples that use this module, not this module itself. It is exercised
by ``tests/test_examples.py``.

Why this module exists
----------------------

``Problem`` asks for six callbacks: ``f``, its two Jacobians ``F = df/dy`` and
``G = df/du``, and the three contracted second derivatives ``F_yy_action``,
``F_yu_action`` and ``F_uu_action``. Only the first is the model. The other
five are consequences of it, and writing them by hand is where an example goes
wrong: a sign, a transpose, or a term dropped from a contraction produces a
gradient that is wrong by a little, which is precisely the failure this
repository's oracle hierarchy exists to catch.

Given ``f`` as a sympy expression, all five follow by differentiation. The
contractions are the ones C-9.4 and the ``Problem`` docstrings specify:

.. math::

    \\texttt{F\\_yy\\_action}(y, u, t, v)_{ij} = \\sum_\\ell v_\\ell
        \\frac{\\partial^2 f_\\ell}{\\partial y_i \\partial y_j}

and likewise for the mixed and control blocks. ``v`` is the adjoint variable
the second-order sweep contracts against, so these are matrices, not tensors.

What this does not change
-------------------------

This is an authoring convenience for examples. Per C-1 the library is a
reference for a C++ port, and sympy does not port, so nothing under
``adjungo/`` imports this module and nothing here participates in a
certification. The generated callbacks are ordinary numpy functions; the
optimizer cannot tell where they came from.

The derivation is checked rather than trusted. ``tests/test_examples.py``
compares every generated callback for the rocket problem against derivatives
written out by hand, which is an independent path to the same quantity.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import sympy as sp
from numpy.typing import NDArray

__all__ = ["SymbolicDynamics"]


def _lambdify(
    arguments: Sequence[Any], expression: Any
) -> Callable[..., Any]:
    """``sympy.lambdify`` against numpy, with array output made explicit.

    ``lambdify`` returns nested lists for a matrix expression and a bare
    Python float for a constant one. Both are hostile to callers that expect
    an array of a fixed shape, so the wrapper settles the type here rather
    than leaving each call site to cope.
    """
    raw = sp.lambdify(arguments, expression, modules="numpy")

    def evaluate(*values: Any) -> Any:
        return np.asarray(raw(*values), dtype=float)

    return evaluate


class SymbolicDynamics:
    """A ``Problem`` whose derivatives are differentiated from ``f``.

    Args:
        f: Right-hand side, a sympy matrix or sequence of ``n`` expressions.
        state: The ``n`` symbols appearing as state components, in order.
        control: The ``nu`` symbols appearing as control components, in order.
        time: The symbol standing for ``t``. Supply one even for an autonomous
            problem; ``f`` need not contain it.

    The constructor does the differentiation once. Evaluation is then numpy
    calls on lambdified expressions, so the cost does not depend on sympy.
    """

    def __init__(
        self,
        f: Any,
        state: Sequence[sp.Symbol],
        control: Sequence[sp.Symbol],
        time: sp.Symbol,
    ) -> None:
        self._state = list(state)
        self._control = list(control)
        self._time = time

        f_matrix = sp.Matrix(f)
        if f_matrix.shape[1] != 1:
            raise ValueError(f"f must be a column of expressions, got {f_matrix.shape}")

        self._n = f_matrix.shape[0]
        self._nu = len(self._control)

        unknown = f_matrix.free_symbols - set(self._state) - set(self._control) - {time}
        if unknown:
            raise ValueError(
                "f contains symbols that are neither state, control nor time: "
                f"{sorted(str(s) for s in unknown)}. Substitute parameters "
                "before constructing, or the callbacks cannot be evaluated."
            )

        y_vec = sp.Matrix(self._state)
        u_vec = sp.Matrix(self._control)
        # `v` contracts the second derivatives. It is the adjoint variable at
        # the point of use, so it has one component per equation of `f`.
        v_syms = sp.symbols(f"_v0:{self._n}")
        v_vec = sp.Matrix(v_syms)

        signature = (self._state, self._control, time)
        contracted = (self._state, self._control, time, list(v_syms))

        self._f = _lambdify(signature, f_matrix)
        self._F = _lambdify(signature, f_matrix.jacobian(y_vec))
        self._G = _lambdify(
            signature,
            f_matrix.jacobian(u_vec) if self._nu else sp.zeros(self._n, 0),
        )

        # Contract first, then differentiate twice: d^2 (v . f) / da db is the
        # same matrix as sum_l v_l d^2 f_l / da db, and avoids building the
        # rank-three object only to sum it away.
        scalar = (v_vec.T * f_matrix)[0, 0]
        self._F_yy = _lambdify(contracted, sp.hessian(scalar, self._state))
        self._F_yu = _lambdify(
            contracted, sp.Matrix([[sp.diff(scalar, a, b) for b in self._control]
                                   for a in self._state])
        )
        self._F_uu = _lambdify(contracted, sp.hessian(scalar, self._control))

    @property
    def state_dim(self) -> int:
        return self._n

    @property
    def control_dim(self) -> int:
        return self._nu

    def f(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self._f(y, u, t).reshape(self._n)

    def F(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self._F(y, u, t).reshape(self._n, self._n)

    def G(self, y: NDArray, u: NDArray, t: float) -> NDArray:
        return self._G(y, u, t).reshape(self._n, self._nu)

    def F_yy_action(self, y: NDArray, u: NDArray, t: float, v: NDArray) -> NDArray:
        return self._F_yy(y, u, t, v).reshape(self._n, self._n)

    def F_yu_action(self, y: NDArray, u: NDArray, t: float, v: NDArray) -> NDArray:
        return self._F_yu(y, u, t, v).reshape(self._n, self._nu)

    def F_uu_action(self, y: NDArray, u: NDArray, t: float, v: NDArray) -> NDArray:
        return self._F_uu(y, u, t, v).reshape(self._nu, self._nu)
