# ShaplEIG reproduction: five claims under one CPU contract

This repository reproduces claims from
**“ShaplEIG: Bayesian Experimental Design for Shapley Value Estimation”**
([arXiv 2606.02247](https://arxiv.org/abs/2606.02247)). The strongest new
finding is a direct official-task counterexample to the GP-ablation claim:
GP+Uncertainty has lower arithmetic-mean Shapley MSE at budgets 16, 24, 32,
and 48; ShaplEIG wins only at 64.

| Claim | Paper result | Observed result | Assessment |
|---|---|---|---|
| Closed-form complexity | \(O(p^4+t^3)\) vs. \(O(4^p t)\) | operation slope 3.970; 21 explicit checks, max error \(4.54\times10^{-10}\); \(p=101\) completed | **VERIFIED** |
| Evaluation scope | 15 tasks, 4 families, budget 512 | exact inventory confirmed; 9 runnable, 6 blocked | **BLOCKED** |
| SOTA performance | lower MSE on all 15 tasks | lower MSE on one official ViT-9 task only, budget ≤64 | **BLOCKED** |
| GP ablation | EIG beats Random, Leverage, Uncertainty | GP+Uncertainty wins 4/5 mean-MSE budgets | **FALSIFIED** |
| Per-iteration overhead | EIG <30 s; refit up to ~25 min | exact code located; author runtime and manual data unavailable | **BLOCKED** |

The agreed compute is local CPU: Apple arm64, 8 logical CPUs, Python 3.12.11,
one locked repository `.venv`, no GPU, and $0 external compute cost. The
cumulative run took 365.15 seconds. The retained ImageNet result uses all 30
official ViT-9 games but is intentionally not described as the 15-task claim.

[Read the illustrated technical report](reports/shapleig-claims/report.md) ·
[Open the tutorial marimo notebook](notebooks/shapleig_claims.py)

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/MachineLearning-Nerd/icml26-repro-ub9PwBtHqD-shapleig/blob/master/notebooks/shapleig_claims.py)

## Experiment log

Every formal node uses the exact same command:
`uv sync --frozen && uv run python repro/src/reproduce.py`.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| [`orx/frozen-judged-baseline`](https://github.com/MachineLearning-Nerd/icml26-repro-ub9PwBtHqD-shapleig/tree/orx/frozen-judged-baseline) | Freeze judged state and locked environment | `uv sync --frozen && uv run python repro/src/reproduce.py` | Retained judged evidence; 18 tests passed | local CPU, 8m16s |
| [`orx/c1-esp-empirical-scaling`](https://github.com/MachineLearning-Nerd/icml26-repro-ub9PwBtHqD-shapleig/tree/orx/c1-esp-empirical-scaling) | Exact-vs-polynomial complexity audit | `uv sync --frozen && uv run python repro/src/reproduce.py` | Claim 1 VERIFIED | local CPU, 4m25s |
| [`orx/c5-per-iteration-overhead-audit`](https://github.com/MachineLearning-Nerd/icml26-repro-ub9PwBtHqD-shapleig/tree/orx/c5-per-iteration-overhead-audit) | Pin timed code and audit prerequisites | `uv sync --frozen && uv run python repro/src/reproduce.py` | Claim 5 BLOCKED | local CPU, 4m25s |
| [`orx/c2-c3-exact-task-availability-audit`](https://github.com/MachineLearning-Nerd/icml26-repro-ub9PwBtHqD-shapleig/tree/orx/c2-c3-exact-task-availability-audit) | Inventory 15 tasks; smoke exact public routes | `uv sync --frozen && uv run python repro/src/reproduce.py` | Claims 2–3 BLOCKED | local CPU, 4m45s |
| [`orx/cumulative-five-claim-evidence-gate`](https://github.com/MachineLearning-Nerd/icml26-repro-ub9PwBtHqD-shapleig/tree/orx/cumulative-five-claim-evidence-gate) | Package Claim 4 and rerun every accepted gate | `uv sync --frozen && uv run python repro/src/reproduce.py` | C1 VERIFIED; C4 FALSIFIED; C2/C3/C5 BLOCKED | local CPU, 6m10s |
| [`orx/release-candidate-report-and-logbook`](https://github.com/MachineLearning-Nerd/icml26-repro-ub9PwBtHqD-shapleig/tree/orx/release-candidate-report-and-logbook) | Reader-facing report, notebook, protected Space candidate | `uv sync --frozen && uv run python repro/src/reproduce.py` | Release-gate run pending | local CPU |
| `master` | Public publication surface | Not run as an experiment (publication surface) | Awaiting explicit release approval | — |

## Reproduce

```bash
uv sync --frozen
uv run python repro/src/reproduce.py
```

The command downloads and hash-checks the 30 pinned public games, runs the
tests, regenerates raw results and figures, runs independent claim checkers,
and exits nonzero on a failed accepted gate. Evidence is under
`.openresearch/artifacts/claim_1` through `.openresearch/artifacts/claim_5`.

For the tutorial notebook:

```bash
uvx marimo edit notebooks/shapleig_claims.py
uvx marimo run notebooks/shapleig_claims.py
```

The notebook embeds small accepted results and does not trigger the formal
experiment.

## Scope and limitations

The paper source is pinned by SHA-256 and theorem/figure anchors. The public
task inventory contains six complete precomputed tasks and three executable
tree tasks. Exact reproduction of the remaining TabPFN and YAHPO tasks needs
unpublished/manual inputs and a different author runtime contract. Adaptive GP
choices are sensitive to near-tied floating-point scores, although the Claim 4
budget-level outcome is stable across the judged and cumulative runs. No score
increase is claimed; only a future live judge can change the judged score.
