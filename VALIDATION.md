# Validation — eigenmind 1.8.0

Validated on 13 October 2026, on Linux:

| Python | NumPy | SciPy | `pytest` | `ruff check` / `ruff format --check` |
|---|---|---|---|---|
| 3.10.20 | 2.2.6 | 1.15.3 | 67 passed | clean |
| 3.13.16 | 2.5.3 | 1.18.1 | 67 passed | clean |

- The 67 tests comprise the package tests (`tests/`) and the 37 self-tests embedded in the modules: Spectral Attention (12), Semantic Bridges (6), Supported Diversity (10), graph construction (4), selection baselines (5).
- The self-tests reproduce the three worked examples of the Eigenmind 1.8 report (eigenvalue 0.189986 and selection {A, C, D, F}; bridge C–D at cosine 0.707107; Supported Diversity {A, C} proven optimal) and check the MILP against exhaustive search on 45 seeded graphs.
- The deprecated pre-1.8 API is covered by its original tests, run unchanged apart from the renamed `legacy_selection_tags` and a filter for the expected warnings, plus a test that every deprecated name warns.
- `python -m build` produces the sdist and wheel without warnings; `twine check` passes.

These checks establish implementation and packaging consistency, not semantic usefulness.
