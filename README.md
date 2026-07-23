# ShaplEIG reproduction: exact CPU counterexamples on official tasks

This repository reproduces claims from
**“ShaplEIG: Bayesian Experimental Design for Shapley Value Estimation”**
([arXiv 2606.02247](https://arxiv.org/abs/2606.02247)). The strongest new
finding is a matched counterexample to the judge's broad Figure 1 performance
claim. On all three exact 10-player data-valuation tasks, Regression MSR has
lower mean MSE than ShaplEIG at budget 16; all three comparisons survive a
paired bootstrap, Wilcoxon test, and Holm correction over the complete
60-comparison search.

| Claim | Paper result | Observed result | Assessment |
|---|---|---|---|
| Closed-form complexity | \(O(p^4+t^3)\) vs. \(O(4^p t)\) | operation slope 3.970; 21 explicit checks, max error \(4.54\times10^{-10}\); \(p=101\) completed | **VERIFIED** |
| Evaluation scope | 15 tasks, 4 families, budget 512 | four exact tasks executed across two families, 600 checked rows, budget 64 | **TOY subset; full claim BLOCKED** |
| SOTA performance | lower MSE across the 15 Figure 1 tasks and varying budgets | Regression MSR wins at budget 16 on all three exact public data-valuation tasks | **FALSIFIED** |
| GP ablation | EIG beats Random, Leverage, Uncertainty | GP+Uncertainty wins 4/5 mean-MSE budgets | **FALSIFIED** |
| Per-iteration overhead | EIG <30 s; refit up to ~25 min | 80 decomposed timing cells at \(p=9–10\); max fit 0.678 s and EIG 0.421 s | **TOY subset; full claim BLOCKED** |

The agreed compute is local CPU: Apple arm64, 8 logical CPUs, Python 3.12.11,
one locked repository `.venv`, no GPU, and $0 external compute cost. The
exact three-task search took 582.80 seconds; its first cumulative run took
990.35 seconds. A subsequent immutable release-gate run regenerated the full
suite in 29m26s. The paper prose separately allows very short early-budget
exceptions, so the report states that nuance and does not claim a complete
15-task rerun.

[Read the illustrated technical report](reports/shapleig-claims/report.md) ·
[Read the Claims 2 and 5 partial-evidence report](reports/c2-c5-partial/report.md) ·
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
| [`orx/release-candidate-report-and-logbook`](https://github.com/MachineLearning-Nerd/icml26-repro-ub9PwBtHqD-shapleig/tree/orx/release-candidate-report-and-logbook) | Reader-facing report, notebook, protected Space candidate | `uv sync --frozen && uv run python repro/src/reproduce.py` | Five-claim release gate passed | local CPU, 9m31s |
| [`orx/c3-exact-data-valuation-counterexample-search`](https://github.com/MachineLearning-Nerd/icml26-repro-ub9PwBtHqD-shapleig/tree/orx/c3-exact-data-valuation-counterexample-search) | Three exact tasks, 30 games each, 60 corrected comparisons | `uv sync --frozen && uv run python repro/src/reproduce.py` | Claim 3 FALSIFIED under the registered judge-claim contract | local CPU, 16m36s |
| [`orx/c3-falsification-release-gate`](https://github.com/MachineLearning-Nerd/icml26-repro-ub9PwBtHqD-shapleig/tree/orx/c3-falsification-release-gate) | Regenerate all claims and validate the additive logbook/report candidate | `uv sync --frozen && uv run python repro/src/reproduce.py` | C1 VERIFIED; C3/C4 FALSIFIED; C2/C5 BLOCKED; release validation passed | local CPU, 29m26s |
| [`orx/claim-3-final-release-report`](https://github.com/MachineLearning-Nerd/icml26-repro-ub9PwBtHqD-shapleig/tree/orx/claim-3-final-release-report) | Presentation-only provenance update after the immutable gate | Not run as an experiment (publication surface) | Published in prior Space revision | — |
| [`orx/c5-decomposed-exact-task-timing-subset`](https://github.com/MachineLearning-Nerd/icml26-repro-ub9PwBtHqD-shapleig/tree/orx/c5-decomposed-exact-task-timing-subset) | Measure GP fitting and EIG separately on four exact public tasks | `uv sync --frozen && uv run python repro/src/reproduce.py` | Claim 5 TOY subset; full claim remains BLOCKED | local CPU, 50m14s |
| [`orx/c2-four-task-executed-subset`](https://github.com/MachineLearning-Nerd/icml26-repro-ub9PwBtHqD-shapleig/tree/orx/c2-four-task-executed-subset) | Register 600 checked ShaplEIG observations over four exact tasks | `uv sync --frozen && uv run python repro/src/reproduce.py` | Claim 2 TOY subset; full claim remains BLOCKED | local CPU, 21m48s |
| `master` | Public publication surface | Not run as an experiment (publication surface) | HF revision `6902e5b8`; awaiting live judge | — |

## Reproduce

```bash
uv sync --frozen
uv run python repro/src/reproduce.py
```

The command downloads and hash-checks 120 pinned public games (30 ViT-9 plus
90 data-valuation games), runs the tests, regenerates raw results and figures,
runs independent claim checkers, and exits nonzero on a failed accepted gate.
Evidence is under
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
tree tasks. Claim 3's counterexample uses three of those exact public tasks;
it is not presented as complete Figure 1 recreation. Exact reproduction of
the remaining TabPFN and YAHPO tasks needs
unpublished/manual inputs and a different author runtime contract. Adaptive GP
choices are sensitive to near-tied floating-point scores, although the Claim 4
budget-level outcome is stable across the judged and cumulative runs. No score
increase is claimed; only a future live judge can change the judged score.
