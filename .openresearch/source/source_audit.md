# Paper source audit

- Paper: *ShaplEIG: Bayesian Experimental Design for Shapley Value Estimation*
- arXiv identifier and version: `2606.02247v1`
- Retrieved URL: `https://ar5iv.labs.arxiv.org/html/2606.02247`
- Retrieval time: `2026-07-23T07:14:23Z`
- HTTP status: `200`
- Retrieval User-Agent: `Mozilla/5.0 (compatible; OpenResearch-Reproduction/1.0; +https://github.com/MachineLearning-Nerd/icml26-repro-ub9PwBtHqD-shapleig)`
- Retrieved HTML SHA-256: `81809d39fa180bab590aa3eb2c3a0c37ac3d17df20c91ac47dc4f805c981dff1`
- Retrieved HTML size: `982071` bytes

## Claim anchors and exact scope

1. Closed-form and complexity:
   - Theorem 3.1: HTML anchor `#S3.Thmtheorem1`.
   - Naive computation: `#S3.SS4.SSS0.Px1`, stated as
     `O(4^p * t)` for a specific candidate.
   - Theorem 3.1 states that the EIG for a candidate
     `z^(i) in {0,1}^p` is computable in `O(p^4 + t^3)`.
   - Appendix B.4 (`#A2.SS4`) refines evaluation over a candidate set `W` to
     `O(p^4 + t^3 + |W| t^2)`. A benchmark must distinguish per-candidate,
     reusable, and candidate-set costs.
2. Fifteen-task protocol:
   - Experimental setup: `#S5.SS1`.
   - Table 1: `#S5.T1`; it has 15 rows across feature importance, data
     valuation, hyperparameter importance, and local explanation, with
     `8 <= p <= 101` and either 30 or 100 repetitions.
   - Section 5.1 states a maximum of 512 evaluations or exhaustion of all
     coalitions, exhaustive or at-most-1024 candidate optimization, and
     `T0 = p + 1`.
3. Accuracy:
   - Figure 1: `#S5.F1`.
   - Results: `#S5.SS2`. The paper says ShaplEIG has the best overall accuracy
     "across all tasks," is at least as good except over very short intervals,
     strictly dominates all established competitors at all budgets in the
     majority of tasks, and is never outperformed by a substantial margin in
     the remaining games.
4. Ablation:
   - Main-text ablation claim: `#S5.SS2.SSS0.Px1`.
   - Full ablation figure: `#A4.F3`.
   - The comparison fixes the GP surrogate and initial design and changes only
     coalition acquisition among EIG, random, leverage, and uncertainty.
5. Computational cost:
   - Main-text claim: `#S5.SS2.SSS0.Px2`.
   - Figures 5 and 6: `#A4.F5` and `#A4.F6`; Appendix D.2.2:
     `#A4.SS2.SSS2`.
   - The stated quantifiers are: for games with at most 16 players, refitting
     reaches about two minutes while EIG is always below one second; for games
     up to 100 players, refitting reaches about 25 minutes while EIG remains
     below 30 seconds. These are per-iteration overheads across all tasks and
     budgets, averaged over repetitions with SEM.

## Assumptions that affect experiments

- Homoscedastic i.i.d. additive Gaussian observation noise.
- Zero-mean, unit-variance GP prior after standardizing training outputs.
- Weighted Hamming kernel; MAP lengthscales with the stated log-normal prior.
- Fixed variance `1e-6` (quasi-noiseless).
- Leverage-score initial design of size `p + 1`.
- For `p > 16`, the fixed hyperparameter refitting schedule in Appendix D.1.4.
- Official reproducibility environment: Python 3.11.13; 32 CPU cores and
  64 GB RAM; Torch 2.9.1, GPyTorch 1.14, BoTorch 0.14.0, shapiq 1.4.1.
- Frozen judged baseline environment: Python 3.12.11 because the existing
  repository documents Python 3.12 and pins SciPy 1.18.0, which requires
  Python 3.12 or newer. This is an explicit deviation from the author
  environment, not an equivalence claim.
