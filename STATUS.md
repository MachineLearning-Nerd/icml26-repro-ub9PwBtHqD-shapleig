# STATUS — ShaplEIG (`ub9PwBtHqD`)

**Session:** perfect-score campaign. **Last updated:** 2026-07-18. **State:** full-scale C2 repair published; under official verdict.

GitHub: `MachineLearning-Nerd/icml26-repro-ub9PwBtHqD-shapleig` at public SHA
`4c19e53` (implementation `1d9b620`). HF Space: `DineshAI/ub9PwBtHqD` at
exact SHA `69a5618000de4028d81d357542710df9d19145fb`.

## Source
- arXiv 2606.02247. Clean-room from PDF. Linear-Gaussian Shapley BED model.

## Evidence
- **C1 verified:** closed-form EIG (Eq. 3) matches independent MC mutual
  information (worst 0.0023); posterior cov matches GP Schur form (9.4e-16).
- **C2 full-scale repair verified locally:** all 30 official ImageNet/ViT-9
  local-explanation games (512 coalitions each), 11 methods, five budgets, and
  realized MSE. Adaptive ShaplEIG has a paired geometric-mean MSE ratio below
  one against all six SOTA baselines; worst ratio is **0.578** vs RegressionMSR
  (95% CI **[0.457, 0.724]**). All paired p-values `<5e-5`.
- LeverageSHAP and RegressionMSR match pinned author source exactly on five
  comparisons (`max_abs_diff=0.0`).
- **18/18 tests pass**; 14 approach/ablation routes executed.

## Next
- Poll the exact Space SHA for the official 4/4 verdict. The current dataset
  still contains the prior `e42a5ea` 3/4 verdict and must not be treated as the
  result of this repair.
