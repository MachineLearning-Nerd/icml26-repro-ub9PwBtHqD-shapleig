# Claim 3 method

This audit runs all 30 pinned public games for each of the three exact
10-player data-valuation tasks. It uses the author configuration's leverage
initial design (`p+1`), weighted-Hamming GP, five MAP restarts, refitting every
iteration, exhaustive candidate search, seeds 1–30, and the four named
baselines. Errors are realized MSE against exhaustive 1024-coalition Shapley
values. The registered early-budget grid is 16, 24, 32, 48, and 64.

For every task/budget/baseline cell, inference is paired over the 30 games.
A counterexample requires arithmetic and geometric mean MSE to favor the
baseline, a 20,000-resample paired bootstrap ratio interval wholly above one,
and a one-sided paired Wilcoxon test surviving Holm correction over all 60
registered comparisons.
