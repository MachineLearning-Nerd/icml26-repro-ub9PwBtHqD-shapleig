# Claim 3 source audit

- Paper: arXiv:2606.02247v1, Figure 1 and Section 5.2.
- Public data: `mmschlk/shapiq` commit
  `799cfd0f2c32f17446130247a7ac3519e68cce82`.
- Author code: `slds-lmu/shapleig` commit
  `162ce44fe380c7c11b959fc85206b5dcdeddff58`.
- Exact configuration:
  `src/xac/experiments/conf/shapleig_crv_shapiq_dv_10p.yaml`.
- Quantifiers tested here: three named data-valuation tasks, 30 repetitions,
  exact 10-player games, four named baselines, and five early budgets.
- Source nuance: Section 5.2 permits exceptions over very short intervals.
  Budget 16 is the earliest tested point, five evaluations after the `p+1`
  initial design, so the verdict is scoped to the judge's broad Claim 3
  wording and does not claim contradiction of every narrower prose sentence.
