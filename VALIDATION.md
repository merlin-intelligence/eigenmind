# Package validation

Validated on 11 October 2026.

- Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1 (Linux).
- All 37 built-in tests passed: Spectral Attention (12), Semantic Bridges (6), Supported Diversity (10), Graph construction (4), Selection baselines (5).
- The three report examples are reproduced by regression tests (eigenvalue 0.189986 and selection {A,C,D,F}; bridge C–D at 0.707107; Supported Diversity {A,C} proven optimal).
- All three CLI demonstrations completed and produced parseable JSON.
- `scale_check.py --v17 <v1.7 folder>` reproduces the figures of report Section 10.
- The report compiles with pdfLaTeX (three passes) without overfull boxes or undefined references; the PDF has 15 pages and identifies itself as version 1.8.

`requirements.txt` sets SciPy ≥ 1.9 (needed for `scipy.optimize.milp`) and leaves other versions unpinned; only the environment above was tested.

These checks establish implementation and packaging consistency, not semantic usefulness.
