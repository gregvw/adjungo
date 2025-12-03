┌─────────────────────────────────────────────────────────────────┐
│                     Problem Specification                       │
├─────────────────────────────────────────────────────────────────┤
│  • State dimension n, control dimension ν                       │
│  • RHS components: f^(1), ..., f^(N)  (for additive splitting)  │
│  • Objective: J(y,u) with terminal/running costs                │
│  • Callbacks for:                                               │
│      - f^(ν)(y,u,t)           → value                           │
│      - F^(ν) = ∂f^(ν)/∂y      → Jacobian (or AD)                │
│      - G^(ν) = ∂f^(ν)/∂u      → Jacobian (or AD)                │
│      - F^(ν)_yy[v], F^(ν)_yu[v], F^(ν)_uu[v]  → Hessian-vector  │
│  • Or: just f^(ν), and let AD handle all derivative             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Method Specification                        │
├─────────────────────────────────────────────────────────────────┤
│  GLM structure:                                                 │
│  • s = number of internal stages                                │
│  • r = number of external stages                                │
│  • Tableaux: A^(ν) ∈ ℝ^{s×s}, U ∈ ℝ^{s×r}                       │
│              B^(ν) ∈ ℝ^{r×s}, V ∈ ℝ^{r×r}                       │
│  • Partitioning type:                                           │
│      - STANDARD: single tableau, all components                 │
│      - ADDITIVE: N tableaux for RHS splitting (IMEX, ARK)       │
│      - PARTITIONED: tableaux per state block (PRK, Nyström)     │
│  • Starting method (if r > 1)                                   │
│  • Abscissae c for control interpolation                        │
│      - STANDARD: single tableau, all components                 │
│      - ADDITIVE: N tableaux for RHS splitting (IMEX, ARK)       │
│      - PARTITIONED: tableaux per state block (PRK, Nyström)     │
│  • Starting method (if r > 1)                                   │
│  • Abscissae c for control interpolation                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Core Solver Engine                         │
├─────────────────────────────────────────────────────────────────┤
│  Generic implementations of:                                    │
│  • Forward state solve                                          │
│  • Backward adjoint solve                                       │
│  • Forward state sensitivity                                    │
│  • Backward adjoint sensitivity                                 │
│  • Gradient assembly                                            │
│  • Hessian-vector product assembly                              │
│                                                                 │
│  All templated/generic over:                                    │
│  • Scalar type (double, AD type, complex)                       │
│  • Linear algebra backend                                       │
│  • Problem callbacks                                            │
│  • Method tableaux                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Optimization Interface                      │
├─────────────────────────────────────────────────────────────────┤
│  Provides to outer optimizer:                                   │
│  • J(u)          — objective evaluation                         │
│  • ∇J(u)         — gradient                                     │
│  • H(u)·v        — Hessian-vector product                       │
│                                                                 │
│  Compatible with:                                               │
│  • Gradient descent, L-BFGS (first-order)                       │
│  • Newton-CG, trust-region (second-order)                       │
│  • SQP if adding inequality constraints                         │
└─────────────────────────────────────────────────────────────────┘
