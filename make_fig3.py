"""
Figure 3: signed confidence distributions, from the saved shot-aligned
signed-delta arrays (no re-decoding).

The integer-dB histograms are cached as JSON in figures3/derived/, so the
figures regenerate without the raw arrays. On a cache miss the script reads
the saved shot-aligned arrays:

    delta_mwpm_d{d}_dt{dt}_p{p}_shots{shots}.npy
    delta_gnn_d{d}_dt{dt}_p{p}_shots{shots}.npy

Outputs:
    figures3/fig3a_distribution_d9.pdf/.png     main-text Fig. 3(a), (9, 9)
    figures3/fig3b_distribution_d5.pdf/.png     main-text Fig. 3(b), (5, 5)
    figures3/appB_distribution_d7_dt7.*         Appendix B, (7, 7)
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Path detection
# ---------------------------------------------------------------------------

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_runs")

# Output folder for the paper figures.
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures3")
DERIVED_DIR = os.path.join(FIG_DIR, "derived")

RECOMPUTE = False

# Decoder-color convention shared by all figures in which the two
# decoders appear in one panel: blue = MWPM, red = GNN.
# Marker/linestyle are kept as a redundant encoding.
COLOR_MWPM = "#1f77b4"   # blue
COLOR_GNN = "#e60000"    # red

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw(d, dt, p, shots):
    """
    Load saved delta_db arrays.

    Returns:
        delta_mwpm, delta_gnn
    """
    tag = f"d{d}_dt{dt}_p{p}_shots{shots}"

    path_mwpm = os.path.join(SAVE_DIR, f"delta_mwpm_{tag}.npy")
    path_gnn = os.path.join(SAVE_DIR, f"delta_gnn_{tag}.npy")

    if not os.path.isfile(path_mwpm):
        raise FileNotFoundError(f"Missing MWPM file:\n{path_mwpm}")

    if not os.path.isfile(path_gnn):
        raise FileNotFoundError(f"Missing GNN file:\n{path_gnn}")

    return np.load(path_mwpm), np.load(path_gnn)


def get_distribution(d, dt, p, shots):
    """
    Cached integer-dB histograms of the signed confidence, per decoder.
    Returns {"MWPM": {"lo": ..., "counts": [...]}, "GNN": {...}}.
    """
    os.makedirs(DERIVED_DIR, exist_ok=True)
    cache = os.path.join(DERIVED_DIR,
                         f"fig3_dist_d{d}_dt{dt}_p{p}_shots{shots}.json")
    if not RECOMPUTE and os.path.isfile(cache):
        with open(cache) as f:
            return json.load(f)

    delta_mwpm, delta_gnn = load_raw(d=d, dt=dt, p=p, shots=shots)
    out = {}
    for name, delta_db in [("MWPM", delta_mwpm), ("GNN", delta_gnn)]:
        xs, _, counts = integer_db_distribution(delta_db)
        out[name] = {"lo": int(xs[0]), "counts": counts.tolist()}
    with open(cache, "w") as f:
        json.dump(out, f)
    return out


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def config_label(d, dt, decoder_name=None):
    """
    Legend/caption label with both distance and number of rounds.
    """
    base = rf"$d={d}$, $r={dt}$"
    if decoder_name is None:
        return base
    return rf"{base}, {decoder_name}"


def integer_db_distribution(delta_db):
    """
    Bin signed gaps by nearest integer dB.

    Returns:
        xs, probabilities, counts
    """
    rounded = np.rint(delta_db).astype(int)

    lo = rounded.min()
    hi = rounded.max()

    xs = np.arange(lo, hi + 1)

    counts = np.bincount(
        rounded - lo,
        minlength=len(xs),
    )

    probs = counts / counts.sum()

    return xs, probs, counts


def cosine_smooth(y, radius=3):
    """
    Hann-cosine smoothing for readability.
    Preserves total probability mass.
    """
    if radius <= 0:
        return y.copy()

    t = np.arange(-radius, radius + 1)

    window = 1.0 + np.cos(np.pi * t / radius)
    window /= window.sum()

    y_smooth = np.convolve(
        y,
        window,
        mode="same",
    )

    if y_smooth.sum() > 0:
        y_smooth *= y.sum() / y_smooth.sum()

    return y_smooth


# ---------------------------------------------------------------------------
# Plot 1: signed gap distribution
# ---------------------------------------------------------------------------

def plot_signed_gap_distribution_many(
    configs,
    p,
    shots,
    smooth_radius=3,
    min_count=20,
    save_stem=None,
):
    """
    Fig. 9-style signed-gap distribution:
        raw points + smoothed curves.

    Color encodes the decoder (MWPM blue, GNN vermillion); when a single
    config is plotted, the (d, r) values go into the legend title.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    decoder_styles = {
        "MWPM": {
            "marker": "s",
            "linestyle": "-",
            "color": COLOR_MWPM,
        },
        "GNN": {
            "marker": "o",
            "linestyle": "--",
            "color": COLOR_GNN,
        },
    }

    xcap_map  = {7: 500, 9: 800}

    for d, dt in configs:
        x_cap = xcap_map.get(d, None)

        dist = get_distribution(d=d, dt=dt, p=p, shots=shots)

        for decoder_name in ("MWPM", "GNN"):
            style = decoder_styles[decoder_name]

            counts = np.asarray(dist[decoder_name]["counts"], dtype=np.int64)
            xs = dist[decoder_name]["lo"] + np.arange(len(counts))
            if x_cap is not None:
                keep = np.abs(xs) <= x_cap
                xs, counts = xs[keep], counts[keep]
            probs = counts / counts.sum()

            # Suppress insignificant bins before smoothing so they don't
            # dominate the log-scale y-axis (e.g. 1-count bins at 1e-8).
            sig = counts >= min_count
            probs_sig = np.where(sig, probs, 0.0)

            probs_smooth = cosine_smooth(
                probs_sig,
                radius=smooth_radius,
            )

            plot_pts = sig & (probs > 0)

            # Raw binned probabilities as points
            ax.scatter(
                xs[plot_pts],
                probs[plot_pts],
                color=style["color"],
                marker=style["marker"],
                s=10,
                alpha=0.55,
                linewidths=0,
            )

            # Smoothed curve (only where at least one neighbour is significant)
            smooth_nonzero = probs_smooth > 0
            label = (decoder_name if len(configs) == 1
                     else config_label(d, dt, decoder_name))
            ax.plot(
                xs[smooth_nonzero],
                probs_smooth[smooth_nonzero],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                label=label,
            )

    ax.axvline(
        0,
        color="black",
        linestyle=":",
        linewidth=1,
        alpha=0.6,
    )

    ax.set_yscale("log")

    ax.set_xlabel(r"Signed confidence $\Delta$ (dB)")
    ax.set_ylabel("Probability")

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    legend_title = (rf"$d={configs[0][0]}$, $r={configs[0][1]}$, $p={p}$"
                    if len(configs) == 1 else None)
    ax.legend(
        ncol=1,
        title=legend_title,
        framealpha=0.9,
    )

    fig.tight_layout()

    if save_stem is not None:
        for ext in ("pdf", "png"):
            path = f"{save_stem}.{ext}"
            fig.savefig(path, dpi=300)
            print(f"Saved signed-gap figure to: {path}")

    plt.close(fig)

    return fig, ax


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Must match the names of your saved .npy files.
    p = 5e-3
    shots = 10**8

    # Paper Fig. 3: signed confidence distributions, (a) = (9, 9) and
    # (b) = (5, 5) in the main text (mirroring Fig. 2); (7, 7) in Appendix B.
    os.makedirs(FIG_DIR, exist_ok=True)

    plot_signed_gap_distribution_many(
        configs=[(9, 9)], p=p, shots=shots, smooth_radius=3, min_count=3,
        save_stem=os.path.join(FIG_DIR, "fig3a_distribution_d9"),
    )
    plot_signed_gap_distribution_many(
        configs=[(5, 5)], p=p, shots=shots, smooth_radius=3, min_count=3,
        save_stem=os.path.join(FIG_DIR, "fig3b_distribution_d5"),
    )
    plot_signed_gap_distribution_many(
        configs=[(7, 7)], p=p, shots=shots, smooth_radius=3, min_count=3,
        save_stem=os.path.join(FIG_DIR, "appB_distribution_d7_dt7"),
    )