"""
Draw the model-comparison and hyperparameter figures for the report.
Input : results/metrics.csv with columns
        model, hidden_units, learning_rate, window, epochs, val_rmse, rmse, mae
        (written by train.py, one row per training run)
Output: results/fig_model_comparison.png, results/fig_hyperparameters.png
Run   : python plot_results.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).parent / "results"
HYPERPARAMS = ["hidden_units", "learning_rate", "window", "epochs"]

df = pd.read_csv(OUT / "metrics.csv")
trained = df[df["params"] > 0] if "params" in df.columns else df.dropna(subset=["hidden_units"])   # without the two baselines
HYPERPARAMS = [h for h in HYPERPARAMS if h in trained.columns and trained[h].nunique() > 1]

# Figure 1: best run (lowest RMSE) of each model, RMSE and MAE side by side
key = "val_rmse" if "val_rmse" in df.columns else "rmse"   # choose runs on validation error
best = df.loc[df.groupby("model")[key].idxmin()].set_index("model")[["rmse", "mae"]]
ax = best.plot.bar(rot=0, figsize=(6, 3.6), color=["#1f4e9c", "#ff8a1f"])
ax.set_ylabel("Error on test set")
ax.set_title("Test error of the best run of each model")
plt.tight_layout()
plt.savefig(OUT / "fig_model_comparison.png", dpi=200)
plt.close()

# Figure 2: RMSE against each hyperparameter, one line per model
if HYPERPARAMS:
    fig, axes = plt.subplots(1, len(HYPERPARAMS), figsize=(3.2 * len(HYPERPARAMS), 3.4), squeeze=False)
    for ax, h in zip(axes[0], HYPERPARAMS):
        for model, g in trained.groupby("model"):
            g = g.groupby(h)["rmse"].mean().sort_index()   # average over the other settings
            ax.plot(g.index, g.values, marker="o", label=model)
        ax.set_xlabel(h)
        ax.set_ylabel("mean test RMSE")
        if h == "learning_rate":
            ax.set_xscale("log")
    axes[0][0].legend()
    plt.tight_layout()
    plt.savefig(OUT / "fig_hyperparameters.png", dpi=200)
    plt.close()
# Figure 3: training, validation and test error of the best run of each model (overfitting check)
if "train_rmse" in trained.columns:
    pick = trained.loc[trained.groupby("model")["val_rmse"].idxmin()].set_index("model")[["train_rmse", "val_rmse", "rmse"]]
    ax = pick.rename(columns={"rmse": "test_rmse"}).plot.bar(rot=0, figsize=(6, 3.4), color=["#9AA0A6", "#ff8a1f", "#1f4e9c"])
    ax.set_ylabel("RMSE (vehicles per 15 min)")
    ax.set_title("Training, validation and test error of the best run")
    plt.tight_layout()
    plt.savefig(OUT / "fig_generalisation.png", dpi=200)
    plt.close()
print("Saved figures to", OUT)
