# STATUS — ShaplEIG (`ub9PwBtHqD`)

**Session:** perfect-score campaign. **Last updated:** 2026-07-18. **State:** C2 full-scale repair passed the publish gate; publishing.

GitHub: `MachineLearning-Nerd/icml26-repro-ub9PwBtHqD-shapleig` at public SHA `ec0b99a`.
HF Space (queued): `DineshAI/ub9PwBtHqD`.

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
- Push GitHub and sync `DineshAI/ub9PwBtHqD`, then poll the exact Space SHA for
  the official 4/4 verdict.
