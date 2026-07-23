import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # ShaplEIG claim-by-claim reproduction

    **Strongest result first:** on the official 30-game ViT-9 ablation, GP+Uncertainty has lower arithmetic-mean Shapley MSE at four of five budgets. This notebook embeds the accepted results; it does not rerun the six-minute formal experiment.
    """)
    return


@app.cell
def _():
    budgets = [16, 24, 32, 48, 64]
    ablation = {
        "ShaplEIG": [0.0131612, 0.0061901, 0.0037685, 0.0023682, 0.0010531],
        "GP+Uncertainty": [0.0076601, 0.0054042, 0.0032461, 0.0019719, 0.0010797],
        "GP+Random": [0.0112008, 0.0072469, 0.0047516, 0.0026655, 0.0018284],
        "GP+Leverage": [0.0157645, 0.0046350, 0.0037052, 0.0030497, 0.0016768],
    }
    return ablation, budgets


@app.cell
def _(ablation, budgets):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, values in ablation.items():
        ax.plot(budgets, values, marker="o", linewidth=2, label=method)
    ax.set_yscale("log")
    ax.set_xlabel("Coalition-evaluation budget")
    ax.set_ylabel("Arithmetic mean realized Shapley MSE")
    ax.set_title("Official ViT-9 GP ablation")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig
    return


@app.cell
def _(mo):
    verdicts = [
        {"Claim": 1, "Result": "VERIFIED", "Evidence": "21 explicit checks; p=101 scaling"},
        {"Claim": 2, "Result": "BLOCKED", "Evidence": "9/15 exact tasks runnable"},
        {"Claim": 3, "Result": "BLOCKED", "Evidence": "one task is not the 15-task suite"},
        {"Claim": 4, "Result": "FALSIFIED", "Evidence": "GP+Uncertainty wins 4/5 budgets"},
        {"Claim": 5, "Result": "BLOCKED", "Evidence": "author runtime/manual data absent"},
    ]
    mo.ui.table(verdicts, selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why Claim 1 is different

    The closed-form EIG was checked against an explicit \(2^p\) coalition construction for 21 cases. The maximum absolute error was \(4.54\times10^{-10}\). Its modeled operation slope is 3.970 (\(R^2=0.999993\)), while the explicit kernel would require 8 TiB at \(p=20\). That directly tests the paper's complexity contrast.

    ## What remains

    The full evaluation needs three TabPFN feature-importance tasks and three YAHPO hyperparameter-importance tasks. Exact overhead timing also needs the author's Python <3.12 stack, GP dependencies, and manual datasets. Substituting toy timings would not answer those claims.
    """)
    return


if __name__ == "__main__":
    app.run()
