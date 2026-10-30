"""
Fig. 7 (Appendix D): shotwise pairing of the two confidence scores.

2D histograms of (|g_MWPM|, |g_GNN|) on the same shots for (d, r) = (9, 9)
at p = 0.005, split into three panels:

    (a) all shots
    (b) shots where the MWPM decoder fails
    (c) shots where the GNN decoder fails

The Spearman rank correlation (computed on all 10^8 shots) is annotated
on panel (a).

Output (in --out): appC_score_pairing.pdf/.png; the histograms and the rank
correlation are cached in derived/pairing_d9_dt9.json.

Options: --mwpm {mwpm, mwpmcorr} selects the MWPM decoder
(uncorrelated / correlated); --out the output folder.
"""

import argparse
import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import runs

FIG_DIR = os.path.join(runs.BUILD_DIR, "mwpmcorr")

D, DT = 9, 9
P = runs.P_MAIN
SHOTS = runs.SHOTS

G_MAX_VIEW = 150   # dB; axis cap for readability (tails extend far beyond)
BIN_DB = 2         # bin width in dB

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
})


VARIANTS = {None: ("mwpm", "MWPM", r"g_{\rm MWPM}"),
            "mwpm": ("mwpm", r"MWPM$_{\rm uncorr}$", r"g_{\rm uncorr}"),
            "mwpmcorr": ("mwpmcorr", r"MWPM$_{\rm corr}$", r"g_{\rm corr}")}


def compute(mwpm):
    """Spearman rho and the three 2D histograms, from the per-shot arrays."""
    runs.require_arrays(D, DT, P, SHOTS, decoders=[mwpm, "gnn"])
    dm = np.load(runs.array_path(mwpm, D, DT, P, SHOTS))
    dg = np.load(runs.array_path("gnn", D, DT, P, SHOTS))
    conf_m = np.abs(dm)
    conf_g = np.abs(dg)
    fail_m = np.signbit(dm)   # a wrongly decided exact tie is stored as -0.0
    fail_g = np.signbit(dg)
    del dm, dg

    from scipy.stats import spearmanr
    rho, _ = spearmanr(conf_m, conf_g)
    edges = np.arange(0, G_MAX_VIEW + BIN_DB, BIN_DB)
    hists = {}
    for name, mask in (("all", np.ones_like(fail_m)), ("mwpm_fails", fail_m), ("gnn_fails", fail_g)):
        h, _, _ = np.histogram2d(conf_m[mask], conf_g[mask], bins=[edges, edges])
        hists[name] = h.astype(np.int64).tolist()
    return {"d": D, "r": DT, "p": P, "shots": len(conf_m), "spearman_rho": float(rho),
            "edges_db": edges.tolist(), "hist": hists}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fig. 7: joint distribution of the two scores")
    ap.add_argument("--mwpm", choices=["mwpm", "mwpmcorr"], default=None)
    ap.add_argument("--out", default=FIG_DIR)
    args = ap.parse_args()
    MWPM_KEY, MWPM_NAME, MWPM_SCORE = VARIANTS[args.mwpm]
    FIG_DIR = args.out
    os.makedirs(os.path.join(FIG_DIR, "derived"), exist_ok=True)
    start = time.perf_counter()

    cache = os.path.join(FIG_DIR, "derived", f"pairing_d{D}_dt{DT}.json")
    if os.path.exists(cache):
        res = json.load(open(cache))
    else:
        res = compute(MWPM_KEY)
        json.dump(res, open(cache, "w"))
    rho = res["spearman_rho"]
    print(f"Spearman rho (all {res['shots']:,} shots): {rho:.4f}")

    edges = np.array(res["edges_db"])
    panels = [
        ("(a) All shots", "all"),
        (f"(b) {MWPM_NAME} decoder fails", "mwpm_fails"),
        ("(c) GNN decoder fails", "gnn_fails"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharex=True,
                             sharey=True, constrained_layout=True)

    for ax, (title, key) in zip(axes, panels):
        h = np.array(res["hist"][key], dtype=float)
        pc = ax.pcolormesh(edges, edges, h.T, norm=LogNorm(vmin=1),
                           cmap="Blues", rasterized=True)
        ax.plot([0, G_MAX_VIEW], [0, G_MAX_VIEW], color="black",
                linestyle=":", linewidth=1.2)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(rf"${MWPM_SCORE}$ (dB)")
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
