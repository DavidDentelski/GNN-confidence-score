"""
Fig. 3: distributions of the signed confidence Delta of the MWPM decoder and of
the GNN, at p = 0.005 and d = r = 5, 7, 9.

Delta > 0: correctly decoded shot; Delta < 0: logical failure; |Delta| is the
confidence in dB. Integer-dB bins; bins with fewer than MIN_COUNT shots are
suppressed before a Hann-cosine smoothing (radius 3 dB, mass preserving);
raw bins are shown as points, the smoothed curve as a line.

Outputs (in --out): fig3_distribution.pdf/.png (Fig. 3, one row, PRA width) and
one single-panel figure per configuration; the integer-dB counts are cached in
derived/distribution_d{d}_dt{r}.json.

Options: --mwpm {mwpm, mwpmcorr} selects the MWPM decoder (uncorrelated /
correlated, see correlated_gap.py); --out the output folder.
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import figure_style as fs
import runs
from make_postselection import FIG1_LW, FIG1_RC, _fig1_axes_style

FIG_DIR = os.path.join(runs.BUILD_DIR, "mwpmcorr")
P, SHOTS = runs.P_MAIN, runs.SHOTS
MIN_COUNT, SMOOTH_RADIUS = 3, 3
X_CAP = {7: 500, 9: 800}      # dB; axis truncation for readability (tails continue)
PANELS = [((9, 9), "distribution_d9_dt9"), ((5, 5), "distribution_d5_dt5"), ((7, 7), "distribution_d7_dt7")]

VARIANTS = {None: ("delta_mwpm", "MWPM"),
            "mwpm": ("delta_mwpm", r"MWPM$_{\rm uncorr}$"),
            "mwpmcorr": ("delta_mwpmcorr", r"MWPM$_{\rm corr}$")}

plt.rcParams.update({"font.size": 14, "axes.labelsize": 18, "legend.fontsize": 14,
                     "xtick.labelsize": 14, "ytick.labelsize": 14,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})


def integer_db_distribution(delta_db):
    rounded = np.rint(delta_db).astype(np.int64)
    lo = rounded.min()
    counts = np.bincount(rounded - lo)
    xs = np.arange(lo, lo + len(counts))
    return xs, counts / counts.sum(), counts


def cosine_smooth(y, radius=SMOOTH_RADIUS):
    t = np.arange(-radius, radius + 1)
    window = 1.0 + np.cos(np.pi * t / radius)
    window /= window.sum()
    ys = np.convolve(y, window, mode="same")
    return ys * (y.sum() / ys.sum()) if ys.sum() > 0 else ys


def binned_counts(d, r, decoder, derived_dir):
    """Integer-dB counts (first bin, counts) of Delta for one decoder, cached as JSON."""
    cache = os.path.join(derived_dir, f"distribution_d{d}_dt{r}.json")
    res = json.load(open(cache)) if os.path.exists(cache) else {}
    if decoder not in res:
        runs.require_arrays(d, r, P, SHOTS, decoders=[decoder])
        delta = np.load(runs.array_path(decoder, d, r, P, SHOTS))
        if d in X_CAP:
            delta = delta[np.abs(delta) <= X_CAP[d]]
        xs, _, counts = integer_db_distribution(delta)
        res[decoder] = {"first_bin_db": int(xs[0]), "counts": counts.tolist(), "x_cap_db": X_CAP.get(d)}
        os.makedirs(derived_dir, exist_ok=True)
        json.dump(res, open(cache, "w"))
    c = res[decoder]
    counts = np.array(c["counts"], dtype=np.int64)
    xs = np.arange(c["first_bin_db"], c["first_bin_db"] + len(counts))
    return xs, counts / counts.sum(), counts


def distributions(d, r, mwpm_key, mwpm_name, derived_dir):
    """[(name, style, xs, probs, smooth, sig)] for MWPM and the GNN at (d, r)."""
    out = []
    for name, key in ((mwpm_name, mwpm_key), ("GNN", "gnn")):
        xs, probs, counts = binned_counts(d, r, key, derived_dir)
        sig = counts >= MIN_COUNT
        out.append((name, fs.style(key), xs, probs, cosine_smooth(np.where(sig, probs, 0.0)), sig))
    return out


def draw_distributions(ax, curves, s=10, linewidth=2.0):
    for name, style, xs, probs, smooth, sig in curves:
        pts = sig & (probs > 0)
        ax.scatter(xs[pts], probs[pts], color=style["color"], marker=style["marker"],
                   s=s, alpha=0.55, linewidths=0)
        nz = smooth > 0
        ax.plot(xs[nz], smooth[nz], color=style["color"], linestyle=style["linestyle"],
                linewidth=linewidth, label=name)
    ax.axvline(0, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_yscale("log")


def plot_distribution(d, r, curves, save_stem):
    fig, ax = plt.subplots(figsize=(7, 5))
    draw_distributions(ax, curves)
    ax.set_xlabel(r"Signed confidence $\Delta$ (dB)")
    ax.set_ylabel("Probability")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title=rf"$d={d}$, $r={r}$, $p={P}$", framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{save_stem}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  saved {save_stem}.pdf/.png")


def plot_distribution_grid(panels, save_stem):
    """Revised Fig. 3: one column per distance, drawn at PRA two-column width."""
    with plt.rc_context(FIG1_RC):
        fig, axes = plt.subplots(1, len(panels), figsize=(7.1, 2.55), gridspec_kw=dict(wspace=0.12))
        for c, ((d, r), curves) in enumerate(panels):
            ax = axes[c]
            draw_distributions(ax, curves, s=2.5, linewidth=FIG1_LW)
            _fig1_axes_style(ax)
            ax.set_title(rf"$\mathbf{{({'abc'[c]})}}$   $d = r = {d}$", loc="left")
            ax.set_xlabel(r"Signed confidence $\Delta$ (dB)")
        lo = min(a.get_ylim()[0] for a in axes)
        hi = max(a.get_ylim()[1] for a in axes)
        for i, a in enumerate(axes):
            a.set_ylim(lo, hi)
            if i:
                a.tick_params(labelleft=False)
        axes[0].set_ylabel(rf"Probability at $p = {P}$")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.subplots_adjust(top=0.8)
        for ext in ("pdf", "png"):
            fig.savefig(f"{save_stem}.{ext}", dpi=300)
        plt.close(fig)
    print(f"  saved {save_stem}.pdf/.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mwpm", choices=["mwpm", "mwpmcorr"], default=None)
    ap.add_argument("--out", default=FIG_DIR)
    a = ap.parse_args()
    _, mwpm_name = VARIANTS[a.mwpm]
    os.makedirs(a.out, exist_ok=True)
    computed = {}
    for (d, r), stem in PANELS:
        computed[(d, r)] = distributions(d, r, a.mwpm or "mwpm", mwpm_name, os.path.join(a.out, "derived"))
        plot_distribution(d, r, computed[(d, r)], os.path.join(a.out, stem))
    grid = [((d, d), computed[(d, d)]) for d in (5, 7, 9) if (d, d) in computed]
    if grid:
        plot_distribution_grid(grid, os.path.join(a.out, "fig3_distribution"))
