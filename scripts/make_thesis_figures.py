"""Recreate Figures 4.1-4.4 of the thesis from the CSV files in results/.

Run from the repository root:
    python scripts/make_thesis_figures.py
Figures are saved to figures/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

MODELS = ["LR", "RF", "GBDT", "XGBoost", "MLP"]
MAIN = "results/main/final_20seeds_v2_bestgamma.csv"
SENS = "results/sensitivity/perf_gamma2_NL{}_S{}.csv"
GREEN, BLUE, BLACK, GREY = "#0EE071", "#3777FF", "#000000", "#8C8C8C"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def deltas(path):
    """Mean NH - STD differences per classifier (Brier: STD - NH) and Wilcoxon p-values."""
    d = pd.read_csv(path)
    m = d[d["model"] != "LABEL"]
    out = {}
    for mod in MODELS:
        x = {k: m[m.model == mod].pivot_table(index="seed", columns="pipeline", values=k)
             for k in ["roc_auc", "pr_auc", "brier"]}
        out[mod] = dict(
            roc=(x["roc_auc"].NH - x["roc_auc"].STD).mean(),
            pr=(x["pr_auc"].NH - x["pr_auc"].STD).mean(),
            brier=(x["brier"].STD - x["brier"].NH).mean(),
            p_roc=wilcoxon(x["roc_auc"].NH, x["roc_auc"].STD).pvalue,
            p_pr=wilcoxon(x["pr_auc"].NH, x["pr_auc"].STD).pvalue,
        )
    return pd.DataFrame(out).T.astype(float)


def averages(path):
    return deltas(path)[["roc", "pr", "brier"]].mean()


def main():
    os.makedirs("figures", exist_ok=True)
    main_d = deltas(MAIN)

    # Figure 4.1 - mean confidence across gamma
    g = pd.read_csv("results/gamma_sweep/gamma_sweep2_results.csv")
    conf = g[g.model == "LABEL"].groupby(["gamma", "pipeline"]).rej_mean_conf.mean().unstack()
    fig, ax = plt.subplots(figsize=(6.3, 3.4)); x = np.arange(len(conf))
    ax.axvspan(1.6, 2.4, color="#EDEDED", zorder=0)
    ax.plot(x, conf["NH"], "-o", color=GREEN, lw=2.5, label="NH", zorder=3)
    ax.plot(x, conf["STD"], "-o", color=BLUE, lw=2.5, label="STD", zorder=3)
    ax.set_xticks(x, [f"{v:g}" for v in conf.index]); ax.set_xlabel("γ (RBF kernel locality)")
    ax.set_ylabel("Mean confidence"); ax.set_ylim(0.6, 0.9); ax.grid(axis="y", color="#D9D9D9", lw=0.6)
    ax.legend(frameon=False, loc="upper left"); fig.tight_layout(); fig.savefig("figures/fig4_1.png", dpi=300); plt.close()

    # Figure 4.2 - heatmap of NH - STD differences
    cols = [("roc", "p_roc", "Δ ROC-AUC"), ("pr", "p_pr", "Δ PR-AUC"), ("brier", None, "Brier Improvement")]
    gc, bc = np.array([0x0E, 0xE0, 0x71]) / 255, np.array([0x37, 0x77, 0xFF]) / 255
    fig, ax = plt.subplots(figsize=(6.3, 3.3)); ax.set_xlim(0, 3); ax.set_ylim(5, -0.45); ax.axis("off")
    for j, (k, pk, h) in enumerate(cols):
        mx = main_d[k].abs().max(); ax.text(j + 0.5, -0.12, h, ha="center", va="bottom", fontweight="bold")
        for i, mod in enumerate(MODELS):
            v = main_d.loc[mod, k]; t = 0.35 + 0.65 * abs(v) / mx; c = 1 - (1 - (gc if v > 0 else bc)) * t
            ax.add_patch(plt.Rectangle((j + 0.03, i + 0.04), 0.94, 0.92, color=c))
            s = f"{v:+.4f}".replace("-", "−") + ("*" if pk and main_d.loc[mod, pk] < 0.05 else "")
            ax.text(j + 0.5, i + 0.5, s, ha="center", va="center", color="white" if (v < 0 and t > 0.6) else "black")
    for i, mod in enumerate(MODELS):
        ax.text(-0.05, i + 0.5, mod, ha="right", va="center", fontweight="bold")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.97, bottom=0.02); fig.savefig("figures/fig4_2.png", dpi=300); plt.close()

    # Figure 4.3 - ranking vs Brier trade-off
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.3), sharey=True)
    for ax, (k, xl, lim) in zip(axs, [("roc", "Δ ROC-AUC (NH − STD)", 0.02), ("pr", "Δ PR-AUC (NH − STD)", 0.01)]):
        ax.axhline(0, color=GREY, lw=0.8); ax.axvline(0, color=GREY, lw=0.8)
        for mod in MODELS:
            ax.plot(main_d.loc[mod, k], main_d.loc[mod, "brier"], "o", color=BLACK)
            ax.annotate(mod, (main_d.loc[mod, k], main_d.loc[mod, "brier"]), xytext=(5, 4),
                        textcoords="offset points", fontsize=9, fontweight="bold")
        ax.set_xlim(-lim, lim); ax.set_ylim(-0.03, 0.03); ax.set_xlabel(xl)
    axs[0].set_ylabel("Brier Improvement"); fig.tight_layout(); fig.savefig("figures/fig4_3.png", dpi=300); plt.close()

    # Figure 4.4 - sensitivity analyses (averages across classifiers)
    nl = {"10k": averages(MAIN), **{f"{n}k": averages(SENS.format(n * 1000, 3000)) for n in (20, 30, 50)}}
    s_ = {"1k": averages(SENS.format(10000, 1000)), "1.5k": averages(SENS.format(10000, 1500)),
          "3k": averages(MAIN), "5k": averages(SENS.format(10000, 5000))}
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.3), sharey=True)
    series = [("Δ ROC-AUC", "roc", BLACK, "-"), ("Δ PR-AUC", "pr", GREY, "--"), ("Brier Improvement", "brier", GREEN, "-")]
    for ax, (dd, title, xl) in zip(axs, [(nl, "Accepted-size (S = 3,000)", "$N_L$"), (s_, "Rejected-size ($N_L$ = 10,000)", "S")]):
        xs = np.arange(len(dd)); ax.axhline(0, color=GREY, lw=0.8)
        for lab, k, c, ls in series:
            ax.plot(xs, [dd[q][k] for q in dd], ls, marker="o", color=c, lw=2, ms=4.5, label=lab)
        ax.set_xticks(xs, list(dd)); ax.set_xlabel(xl); ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", color="#D9D9D9", lw=0.6)
    axs[0].set_ylabel("Average NH − STD difference")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("figures/fig4_4.png", dpi=300); plt.close()
    print("Figures saved to figures/")


if __name__ == "__main__":
    main()
