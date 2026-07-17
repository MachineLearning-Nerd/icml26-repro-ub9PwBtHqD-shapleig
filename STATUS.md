# STATUS — ShaplEIG (`ub9PwBtHqD`)

**Session:** NewPaper. **Last updated:** 2026-07-17. **State:** locally complete; GitHub push pending; HF queued.

## Source
- arXiv 2606.02247. Clean-room from PDF. Linear-Gaussian Shapley BED model.

## Evidence (locally complete)
- **C1 verified:** closed-form EIG (Eq. 3) matches independent MC mutual
  information (worst 0.0023); posterior cov matches GP Schur form (9.4e-16).
- **C2 verified:** greedy-EIG acquisition reduces posterior Shapley uncertainty
  vs random (det lower 100%, expected-MSE/trace lower 72%).
- **15/15 pytest tests pass** (<1 s).
- Trackio complete/tagged/pinned/command-captured.

## Next
- Push GitHub `MachineLearning-Nerd/icml26-repro-ub9PwBtHqD-shapleig`.
- Publish `DineshAI/ub9PwBtHqD` after HF quota reset; verify tags/bucket; `under_verdict`.
