# Claim 5 source audit

The paper source at SHA-256
`81809d39fa180bab590aa3eb2c3a0c37ac3d17df20c91ac47dc4f805c981dff1`
states at `#S5.SS2.SSS0.Px2` that GP hyperparameter refitting reaches about
120 seconds per iteration for games up to 16 players and about 1,500 seconds
for larger games up to 100 players. It states vectorized EIG is below one
second for small games and below 30 seconds for larger games.

Appendix `#A4.SS2.SSS2` and Figures `#A4.F5`/`#A4.F6` define the scope as all
15 tasks and evaluation budgets, averaged over repetitions with SEM.

At author commit `162ce44fe380c7c11b959fc85206b5dcdeddff58`,
`experiment_runner.py` times `gp.fit()` and the acquisition-function call
separately with `time.perf_counter()`. The acquisition is the vectorized
`EIGFunctionProperty`; it is not the one-candidate complexity diagnostic used
for Claim 1.

The large-game configurations
`shapleig_crv_tree_{corrgroup,nhanesi,crime}.yaml` set `scalability_mode: true`,
`Subset.subset_size: 1024`, five MAP restarts, and the
`init_64_factor_4` refit schedule. They run 451, 432, and 410 adaptive
iterations after p+1 initial coalitions, respectively, reaching total budget
512 for p=60, 79, and 101. `ShapleyApplication.run_lazy_setup` constructs the
fixed candidate subset with the uniform-cardinality PolySHAP sampler and
pairing trick. This source was retrieved from GitHub on 2026-07-24 at the
commit above.
