import numpy as np

from adjungo.core.method import GLMethod, StageType, PropType


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
