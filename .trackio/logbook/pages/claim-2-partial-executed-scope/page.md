# Claim 2 — executed four-task subset

## Full verdict: BLOCKED · Partial assessment: TOY

The fixed reproduction command now executes ShaplEIG on four exact public
tasks rather than only the retained ImageNet task:

| Family | Exact task | Players |
|---|---|---:|
| data valuation | Bike Sharing / random forest | 10 |
| data valuation | Bike Sharing / gradient boosting | 10 |
| data valuation | California Housing / gradient boosting | 10 |
| local explanation | ImageNet / ViT-9 | 9 |

The registered matrix contains **600 complete observations**: four tasks, 30
distinct games per task, and budgets 16, 24, 32, 48, and 64. There are 120
distinct pinned game hashes. A separate CSV parser reconstructs all 20
task/budget aggregates with zero errors. A negative control that removes the
complete ViT-9 task is detected.

| Scope dimension | Paper | Executed |
|---|---:|---:|
| tasks | 15 | 4 |
| task families | 4 | 2 |
| maximum budget | 512 | 64 |

This is stronger partial evidence, but it does not satisfy the exact 15-task
claim. Feature importance and hyperparameter importance remain unavailable
under the frozen public-data contract. No previous evidence is removed or
reclassified.

Formal run: `b49c7271-93c4-4c1e-9c46-94f87cafc59a`  
Git SHA: `8fd97aad01a7500de9492769810700d95264b11c`  
Fixed command: `uv sync --frozen && uv run python repro/src/reproduce.py`

Evidence: `evidence/claim_2/executed_subset_*`.
