# Claim 5 method

This node performs a prerequisite and scope audit before timing. It compares
the immutable reproduction lock and Python contract with the authors' exact
Poetry lock, checks all required runtime modules without importing them, and
checks the manually provisioned YAHPO and TabPFN paths named by the author
README.

The verifier permits `BLOCKED` only while a live prerequisite is absent and
rejects any attempt to treat the single-candidate Claim 1 benchmark as the
paper's vectorized acquisition timing. Its negative control marks every
prerequisite available; under that mutation the blocker must disappear.

No substitute optimizer or synthetic response surface is timed because neither
would test the exact implementation-level claim in Figures 5-6.

The judge-response extension times the exact three public large tree-game
constructors at p=60, 79, and 101. It ports the pinned author's 1,024-candidate
sampling procedure, five-restart ARD Hamming-GP fit, and polynomial vectorized
EIG equations to the frozen NumPy/SciPy environment. It measures the p+1
initial archive and budget 512 separately. A small-grid independent posterior
checker verifies the batch EIG implementation candidate by candidate.

The t=512 design is a deterministic prefix of the source-sampled pool rather
than the full adaptive trajectory, and one paper seed is run. These deviations
prevent promotion to full verification; the purpose is a direct, reproducible
large-regime test of the timing bounds requested by the judge.
