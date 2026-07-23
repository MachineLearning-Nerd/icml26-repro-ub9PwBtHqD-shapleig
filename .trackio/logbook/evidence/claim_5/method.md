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
