import numpy as np

from adjungo.core.method import GLMethod, StageType, PropType
from adjungo.methods import multistep


def test_stage_and_prop_classification():
    A_explicit = np.array([[0.0, 0.0], [0.1, 0.0]])
    U = np.eye(2)
    B = np.eye(2)
    V_identity = np.eye(2)
    c = np.array([0.0, 1.0])

    method_explicit = GLMethod(A=A_explicit, U=U, B=B, V=V_identity, c=c)

    assert method_explicit.s == 2
    assert method_explicit.r == 2
    assert method_explicit.stage_type is StageType.EXPLICIT
    assert method_explicit.prop_type is PropType.IDENTITY
    assert method_explicit.explicit_stage_indices == [0, 1]

    A_sdirk = np.array([[0.5, 0.0], [0.3, 0.5]])
    V_shift = np.eye(2, k=1)

    method_sdirk = GLMethod(A=A_sdirk, U=U, B=B, V=V_shift, c=c)

    assert method_sdirk.stage_type is StageType.SDIRK
    assert np.isclose(method_sdirk.sdirk_gamma, 0.5)
    assert method_sdirk.prop_type is PropType.SHIFT
    assert method_sdirk.explicit_stage_indices == []

    V_triangular = np.tril(np.ones((2, 2)))
    method_triangular = GLMethod(A=A_sdirk, U=U, B=B, V=V_triangular, c=c)

    assert method_triangular.prop_type is PropType.TRIANGULAR

    V_dense = np.array([[1.0, 0.2], [0.3, 0.4]])
    method_dense = GLMethod(A=A_sdirk, U=U, B=B, V=V_dense, c=c)

    assert method_dense.prop_type is PropType.DENSE


def test_multistep_tableau_shapes_and_types():
    bdf2 = multistep.bdf2()
    ab2 = multistep.adams_bashforth2()
    am2 = multistep.adams_moulton2()

    # Basic shape consistency
    assert bdf2.A.shape == (1, 1)
    assert bdf2.U.shape == (1, 2)
    assert bdf2.B.shape == (2, 1)
    assert bdf2.V.shape == (2, 2)
    assert len(bdf2.c) == bdf2.s

    # Explicit Adams-Bashforth should remain explicit with shift propagation
    assert ab2.stage_type is StageType.EXPLICIT
    assert ab2.prop_type is PropType.SHIFT
    assert ab2.A.shape[0] == len(ab2.c)
    assert ab2.B.shape == (ab2.r, ab2.s)

    # Implicit Adams-Moulton should expose its implicit stage and shift history
    assert am2.stage_type is StageType.DIRK or am2.stage_type is StageType.SDIRK
    assert am2.prop_type is PropType.SHIFT
    assert am2.A.shape[0] == len(am2.c)
    assert am2.B.shape == (am2.r, am2.s)
