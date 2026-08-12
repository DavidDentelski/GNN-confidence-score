"""
Appendix C: shotwise pairing of the two confidence scores.

2D histograms of (|g_MWPM|, |g_GNN|) on the same shots for (d, r) = (9, 9)
at p = 0.005, split into three panels:

    (a) all shots
    (b) shots where the MWPM decoder fails
    (c) shots where the GNN decoder fails

This is a direct visualization of where the two confidence scores
agree and disagree at the level of individual shots.
The Spearman rank correlation (computed on a fixed 10^7-shot subsample)
is annotated on panel (a).

The 2D histograms and the correlation are cached as JSON in
figures3/derived/, so the figure regenerates without the raw arrays.

Output: figures3/appC_score_pairing.pdf/.png
"""

import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_runs")

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures3")
DERIVED_DIR = os.path.join(FIG_DIR, "derived")

D, DT = 9, 9
P = 0.005
SHOTS = 100_000_000

G_MAX_VIEW = 150   # dB; axis cap for readability (tails extend far beyond)
BIN_DB = 2         # bin width in dB
SPEARMAN_SAMPLE = 10_000_000
SEED = 1

RECOMPUTE = False

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
})

PANEL_TITLES = ["All shots", "MWPM decoder fails", "GNN decoder fails"]


def compute_pairing():
    tag = f"d{D}_dt{DT}_p{P}_shots{SHOTS}"
    dm = np.load(os.path.join(SAVE_DIR, f"delta_mwpm_{tag}.npy"))
    dg = np.load(os.path.join(SAVE_DIR, f"delta_gnn_{tag}.npy"))

    conf_m = np.abs(dm)
    conf_g = np.abs(dg)
    fail_m = dm < 0
    fail_g = dg < 0
    del dm, dg

    from scipy.stats import spearmanr
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(conf_m), size=SPEARMAN_SAMPLE, replace=False)
    rho, _ = spearmanr(conf_m[idx], conf_g[idx])

    edges = np.arange(0, G_MAX_VIEW + BIN_DB, BIN_DB)
    masks = [np.ones_like(fail_m), fail_m, fail_g]
    hists = []
    for mask in masks:
        h, _, _ = np.histogram2d(conf_m[mask], conf_g[mask],
                                 bins=[edges, edges])
        hists.append(h.tolist())

    return {"config": {"d": D, "dt": DT, "p": P, "shots": SHOTS},
            "bin_db": BIN_DB, "g_max_view": G_MAX_VIEW,
            "spearman_rho": float(rho),
            "spearman_sample": SPEARMAN_SAMPLE,
            "hists": hists}


def get_pairing():
    os.makedirs(DERIVED_DIR, exist_ok=True)
    cache = os.path.join(DERIVED_DIR,
                         f"pairing_d{D}_dt{DT}_p{P}_shots{SHOTS}.json")
    if not RECOMPUTE and os.path.isfile(cache):
        with open(cache) as f:
            return json.load(f)
    res = compute_pairing()
    with open(cache, "w") as f:
        json.dump(res, f)
    return res


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    start = time.perf_counter()

    res = get_pairing()
    rho = res["spearman_rho"]
    print(f"Spearman rho (1e7 subsample): {rho:.4f}")

    edges = np.arange(0, res["g_max_view"] + res["bin_db"], res["bin_db"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True,
                             sharey=True, constrained_layout=True)

    for ax, title, h in zip(axes, PANEL_TITLES, res["hists"]):
        h = np.asarray(h, dtype=np.float64)
        pc = ax.pcolormesh(edges, edges, h.T, norm=LogNorm(vmin=1),
                           cmap="Blues", rasterized=True)
        ax.plot([0, G_MAX_VIEW], [0, G_MAX_VIEW], color="black",
                linestyle=":", linewidth=1.2)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(r"$g_{\rm MWPM}$ (dB)")
        ax.set_xlim(0, G_MAX_VIEW)
        ax.set_ylim(0, G_MAX_VIEW)
        fig.colorbar(pc, ax=ax, shrink=0.85, label="shots")

    axes[0].set_ylabel(r"$g_{\rm GNN}$ (dB)")
    axes[0].annotate(rf"Spearman $\rho = {rho:.2f}$", xy=(0.04, 0.93),
                     xycoords="axes fraction", fontsize=12)

    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"appC_score_pairing.{ext}")
        fig.savefig(path, dpi=300)
        print(f"saved {path}")
    plt.close(fig)

    print(f"Elapsed: {(time.perf_counter() - start) / 60:.1f} min")
