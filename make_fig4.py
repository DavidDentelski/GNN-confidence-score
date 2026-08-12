"""
Figure 4 + Table II + Appendix (alpha vs p): empirical calibration of the
confidence scores.

Model (one-parameter temperature scaling, alpha = 1/T, in dB units):

    P(logical error | g) = 1 / (1 + 10^(alpha * g / 10))            (Eq. 3)

Fitting procedure:
  * alpha is fitted by binomial MAXIMUM LIKELIHOOD on sufficient counts
    in score bins of width 0.001 dB, over the PRE-SPECIFIED window
    g in [G_MIN, G_MAX] dB, fixed a priori.
    A refinement check refits with 10x coarser bins and reports the
    change in alpha; further refinement produces no visible change in
    the fitted parameters (the check is printed per configuration).
  * The quoted uncertainty is the UNSCALED statistical (Fisher) error.
  * Goodness of fit is reported separately as a reduced chi-square over
    the integer-dB display bins; a large chi2_nu is part of the result
    and is NOT absorbed into the error bar.
  * Integer-dB bins and the min-count rule are used for DISPLAY ONLY
    (reliability points, Wilson error bars, residuals).
  * Per-configuration fits are the primary evidence. The pooled panel is
    a summary across configurations (shot-pooled); Table II additionally
    reports the meta-analytic (inverse-variance weighted) mean of the
    per-configuration slopes.

The GNN trivial-syndrome sentinel value is removed per config before
binning (it never enters the fit window; see the robustness appendix).

Outputs:
    figures3/fig4a_calibration_d9.pdf/.png      (9, 9) + residual strip
    figures3/fig4b_calibration_pooled.pdf/.png  summary across configs
    figures3/appB_calibration_d{d}_dt{dt}.*     other configs
    figures3/appD_alpha_vs_p.pdf/.png           alpha vs p (d = 7, 9)
    figures3/tables/appB_alpha.tex              alpha, chi2_nu (Appendix B)

First press computes everything from saved_runs/ (~30 s per config) and
caches binned counts in figures3/derived/ (cache key: calib_v5_*).
"""

import glob
import json
import os
import re
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.optimize import minimize_scalar
from scipy.special import expit

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_runs")

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures3")
DERIVED_DIR = os.path.join(FIG_DIR, "derived")
TABLE_DIR = os.path.join(FIG_DIR, "tables")

P_MAIN = 0.005
SHOTS_MAIN = 100_000_000
CONFIGS = [(5, 5), (5, 7), (5, 9), (5, 11), (7, 7), (7, 9), (7, 11), (9, 9)]
MAIN_CONFIG = (9, 9)
APPB_CONFIGS = [(5, 5), (7, 7)]

RECOMPUTE = False
G_MIN, G_MAX = 10, 30      # pre-specified fit window (dB)
MIN_COUNT = 3              # display-bin validity threshold

FINE_BIN_DB = 0.001        # sufficient-count resolution for the ML fit
FINE_MAX_DB = 50.0         # fine bins cover [0, FINE_MAX_DB)

LN10 = np.log(10.0)

COLOR_MWPM = "#1f77b4"   # blue  (decoder colors shared by all figures)
COLOR_GNN = "#e60000"    # red
DECODER_STYLE = {
    "MWPM": dict(color=COLOR_MWPM, marker="s", linestyle="-"),
    "GNN": dict(color=COLOR_GNN, marker="o", linestyle="--"),
}

plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 18,
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Binned counts per run (cached)
# ---------------------------------------------------------------------------

def bin_deltas(delta_db):
    """Integer-dB display bins AND fine sufficient-count bins for the fit."""
    gap = np.abs(delta_db)
    is_err = delta_db < 0

    # Display bins (integer dB, full range).
    gi = np.rint(gap).astype(np.int64)
    length = int(gi.max()) + 1
    counts = np.bincount(gi, minlength=length)
    err_counts = np.bincount(gi[is_err], minlength=length)

    # Fine bins for the ML fit (0.01 dB, [0, FINE_MAX_DB)).
    n_fine = int(round(FINE_MAX_DB / FINE_BIN_DB))
    in_range = gap < FINE_MAX_DB
    fi = (gap[in_range] / FINE_BIN_DB).astype(np.int64)
    err_in_range = is_err[in_range]
    fine_counts = np.bincount(fi, minlength=n_fine)
    fine_err = np.bincount(fi[err_in_range], minlength=n_fine)

    return counts, err_counts, fine_counts, fine_err


def compute_binned(d, dt, p, shots):
    t0 = time.perf_counter()
    tag = f"d{d}_dt{dt}_p{p}_shots{shots}"
    dm = np.load(os.path.join(SAVE_DIR, f"delta_mwpm_{tag}.npy"))
    dg = np.load(os.path.join(SAVE_DIR, f"delta_gnn_{tag}.npy"))

    # Trivial-syndrome sentinels: shots with bit-identical signed deltas
    # in both arrays (the GNN sentinel is copied from the MWPM trivial-gap
    # computation) that share one |value| — accidental float coincidences
    # between a matching weight and a logit occur at scattered unique
    # values and are kept (they are genuine network outputs). Sentinel
    # confidences are excluded from the GNN calibration data, as they are
    # not produced by the network; they ARE genuine MWPM outputs and
    # remain included for MWPM. The sentinel lies far outside the fit
    # window in every configuration, so this cannot affect the fit.
    cand = np.flatnonzero(dm == dg)
    sentinel, removed = None, 0
    if len(cand) > 0:
        absv = np.abs(dm[cand])
        vals, counts = np.unique(absv, return_counts=True)
        if counts.max() >= 2:
            sentinel = float(vals[np.argmax(counts)])
            triv_idx = cand[absv == sentinel]
            removed = int(len(triv_idx))
            keep = np.ones(len(dg), dtype=bool)
            keep[triv_idx] = False
            dg = dg[keep]

    out = {"config": {"d": d, "dt": dt, "p": p, "shots": shots},
           "gnn_sentinel_db": sentinel, "gnn_sentinel_removed": removed}
    for name, delta in [("MWPM", dm), ("GNN", dg)]:
        c, e, fc, fe = bin_deltas(delta)
        out[name] = {"counts": c.tolist(), "err_counts": e.tolist(),
                     "fine_counts": fc.tolist(), "fine_err": fe.tolist()}
    print(f"  binned d={d}, dt={dt}, p={p} in {time.perf_counter() - t0:.0f} s"
          + (f" (GNN sentinel {sentinel:.2f} dB, {removed:,} removed)"
             if sentinel is not None else ""))
    return out


def get_binned(d, dt, p=P_MAIN, shots=SHOTS_MAIN):
    os.makedirs(DERIVED_DIR, exist_ok=True)
    cache = os.path.join(DERIVED_DIR, f"calib_v5_d{d}_dt{dt}_p{p}_shots{shots}.json")
    if not RECOMPUTE and os.path.isfile(cache):
        with open(cache) as f:
            return json.load(f)
    res = compute_binned(d, dt, p, shots)
    with open(cache, "w") as f:
        json.dump(res, f)
    return res


def pool_counts(binned_list, decoder):
    """Sum bin counts across configs (identical to concatenating shots)."""
    def _sum(key):
        length = max(len(b[decoder][key]) for b in binned_list)
        acc = np.zeros(length, dtype=np.int64)
        for b in binned_list:
            a = np.asarray(b[decoder][key], dtype=np.int64)
            acc[:len(a)] += a
        return acc
    return {"counts": _sum("counts"), "err_counts": _sum("err_counts"),
            "fine_counts": _sum("fine_counts"), "fine_err": _sum("fine_err")}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def wilson_ci(k, n, z=1.96):
    p_hat = k / n
    denom = 1.0 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def fitted_probability(g, alpha):
    return 1.0 / (1.0 + 10.0 ** (alpha * np.asarray(g, dtype=float) / 10.0))


def fit_alpha_ml(fine_counts, fine_err, g_min=G_MIN, g_max=G_MAX,
                 bin_db=FINE_BIN_DB):
    """
    Binomial maximum-likelihood fit of alpha on fine sufficient counts.

    Model: P(error | g) = sigmoid(-c * alpha * g), c = ln(10)/10.
    Returns (alpha, sigma_alpha, n_shots_in_window). sigma_alpha is the
    unscaled statistical error from the observed Fisher information.
    """
    n = np.asarray(fine_counts, dtype=np.float64)
    k = np.asarray(fine_err, dtype=np.float64)
    g = (np.arange(len(n)) + 0.5) * bin_db
    m = (g >= g_min) & (g <= g_max) & (n > 0)
    n, k, g = n[m], k[m], g[m]
    if n.sum() == 0 or k.sum() == 0:
        return None, None, 0

    c = LN10 / 10.0
    x = c * g

    def nll(alpha):
        # log p = log sigmoid(-a x); log(1-p) = log sigmoid(a x)
        return -(np.sum(k * np.log(expit(-alpha * x))
                        + (n - k) * np.log(expit(alpha * x))))

    res = minimize_scalar(nll, bounds=(0.1, 2.0), method="bounded",
                          options={"xatol": 1e-8})
    alpha = float(res.x)

    p = expit(-alpha * x)
    fisher = float(np.sum(n * p * (1.0 - p) * x**2))
    sigma = 1.0 / np.sqrt(fisher)
    return alpha, float(sigma), int(n.sum())


def display_arrays(counts, err_counts, min_count=MIN_COUNT):
    """Integer-dB reliability points (display only)."""
    g = np.arange(len(counts))
    counts = np.asarray(counts, dtype=np.float64)
    err_counts = np.asarray(err_counts, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_fail = err_counts / counts
    valid = ((counts >= min_count) & (err_counts >= min_count)
             & np.isfinite(p_fail) & (p_fail > 0))
    return g, p_fail, counts, err_counts, valid


def chi2_display(counts, err_counts, alpha, g_min=G_MIN, g_max=G_MAX,
                 min_count=MIN_COUNT):
    """Reduced chi-square of the ML fit over display bins in the window."""
    g, p_fail, _, err_arr, valid = display_arrays(counts, err_counts, min_count)
    m = valid & (g >= g_min) & (g <= g_max)
    if m.sum() < 2:
        return None, int(m.sum())
    sigma_ln = np.sqrt((1.0 - p_fail[m]) / err_arr[m])
    resid = (np.log(p_fail[m]) - np.log(fitted_probability(g[m], alpha))) / sigma_ln
    return float(np.sum(resid**2) / (m.sum() - 1)), int(m.sum())


def full_fit(binned, decoder):
    """ML alpha + unscaled sigma + display-bin chi2_nu for one decoder."""
    b = binned[decoder]
    alpha, sigma, n_win = fit_alpha_ml(b["fine_counts"], b["fine_err"])
    if alpha is None:
        return None
    chi2_nu, n_bins = chi2_display(b["counts"], b["err_counts"], alpha)

    # Binning-refinement check: refit with 10x coarser bins. The shift must
    # be negligible compared to sigma for the "further refinement produces
    # no visible change" statement in the paper.
    fc = np.asarray(b["fine_counts"], dtype=np.int64).reshape(-1, 10).sum(axis=1)
    fe = np.asarray(b["fine_err"], dtype=np.int64).reshape(-1, 10).sum(axis=1)
    alpha_coarse, _, _ = fit_alpha_ml(fc, fe, bin_db=FINE_BIN_DB * 10)

    return {"alpha": alpha, "sigma": sigma, "chi2_nu": chi2_nu,
            "n_bins": n_bins, "n_window_shots": n_win,
            "refinement_shift": abs(alpha - alpha_coarse)}


def window_deviation(binned, decoder, alpha, g_min=G_MIN, g_max=G_MAX):
    """
    Relative deviation |data/fit - 1| of the integer-dB display bins inside
    the fit window: the plain-language size of the residual structure that
    chi2_nu resolves (quoted in the Results of the paper). The
    failure-count-weighted mean is the headline (a bare max is dominated by
    the statistically noisiest bin near the window edge); max is secondary.
    """
    b = binned[decoder]
    g, p_fail, _, err_arr, valid = display_arrays(b["counts"], b["err_counts"])
    m = valid & (g >= g_min) & (g <= g_max)
    dev = np.abs(p_fail[m] / fitted_probability(g[m], alpha) - 1.0)
    w = err_arr[m]  # failures per bin ~ inverse variance of the rel. deviation
    return float(np.sum(w * dev) / np.sum(w)), float(dev.max())


# Window-robustness check (reported in an appendix): refit alpha over
# alternative confidence windows. Baseline first; "full" uses every
# populated score bin.
WINDOW_CHOICES = [("$[10, 30]$ (baseline)", 10, 30),
                  ("$[5, 30]$", 5, 30),
                  ("$[10, 40]$", 10, 40),
                  ("$[5, 40]$", 5, 40),
                  ("$[15, 25]$", 15, 25),
                  ("full range", 0, 10**9)]


def window_robustness(binned):
    """
    Alpha for every config x decoder x window. Returns
    {window_label: {decoder: {config_key: {"alpha":..., "sigma":...}}}}.
    """
    out = {}
    for label, g_lo, g_hi in WINDOW_CHOICES:
        out[label] = {}
        for name in ("MWPM", "GNN"):
            out[label][name] = {}
            for cfg in CONFIGS:
                b = binned[cfg][name]
                a, s, _ = fit_alpha_ml(b["fine_counts"], b["fine_err"],
                                       g_min=g_lo, g_max=g_hi)
                out[label][name][f"{cfg[0]},{cfg[1]}"] = {
                    "alpha": a, "sigma": s}
    return out


def write_window_table(wr):
    base_label = WINDOW_CHOICES[0][0]
    lines = [
        r"\begin{table}",
        r"\centering",
        r"\begin{tabular}{l|cc}",
        r"\toprule",
        r"Fit window (dB) & MWPM $\alpha$ & GNN $\alpha$ \\",
        r"\hline",
        r"\midrule",
    ]
    for label, _, _ in WINDOW_CHOICES:
        cells = []
        for name in ("MWPM", "GNN"):
            a = [wr[label][name][f"{d},{dt}"]["alpha"] for d, dt in CONFIGS]
            base = [wr[base_label][name][f"{d},{dt}"]["alpha"]
                    for d, dt in CONFIGS]
            shift = max(abs(x - y) for x, y in zip(a, base))
            rng = f"{min(a):.3f}--{max(a):.3f}"
            cells.append(rng if label == base_label
                         else f"{rng} ({shift:.3f})")
        lines.append(rf"{label} & {cells[0]} & {cells[1]} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Robustness of the fitted calibration slopes to the choice "
        r"of fit window: for each window, the range of $\alpha$ across the "
        r"eight configurations, with the largest per-configuration shift "
        r"relative to the pre-specified window $g \in [10, 30]$~dB of the "
        r"main text in parentheses (binomial maximum-likelihood fits as in "
        r"Table~\ref{alpha_table}; ``full range'' uses every populated score "
        r"bin).}",
        r"\label{window_table}",
        r"\end{table}",
    ]
    path = os.path.join(TABLE_DIR, "appD_window_robustness.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {path}")


def meta_mean(fits):
    """
    Inverse-variance weighted DESCRIPTIVE mean of per-config alphas,
    with heterogeneity statistics. The configurations have genuinely
    different slopes, so this is a summary, not an estimate of a shared
    true alpha; the per-configuration values remain the primary result.
    """
    a = np.array([f["alpha"] for f in fits])
    w = np.array([1.0 / f["sigma"]**2 for f in fits])
    mean = float(np.sum(w * a) / np.sum(w))
    sigma = float(1.0 / np.sqrt(np.sum(w)))
    q = float(np.sum(w * (a - mean) ** 2))
    k = len(a)
    i2 = float(max(0.0, (q - (k - 1)) / q)) if q > 0 else 0.0
    return {"mean": mean, "sigma": sigma, "Q": q, "I2": i2,
            "min": float(a.min()), "max": float(a.max())}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_calibration(ax, axr, data_per_decoder, fits):
    for name, b in data_per_decoder.items():
        style = DECODER_STYLE[name]
        fit = fits[name]
        g, p_fail, counts_arr, err_arr, valid = display_arrays(
            b["counts"], b["err_counts"])

        lo, hi = wilson_ci(err_arr[valid], counts_arr[valid])
        p_v = p_fail[valid]
        ax.errorbar(g[valid], p_v, yerr=[p_v - lo, hi - p_v],
                    color=style["color"], marker=style["marker"],
                    markersize=4, linestyle="none", capsize=2.5, alpha=0.8,
                    label=f"{name} data")

        g_fit = np.linspace(0, g[valid].max(), 400)
        ax.plot(g_fit, fitted_probability(g_fit, fit["alpha"]),
                color=style["color"], linestyle=style["linestyle"],
                linewidth=2.0,
                label=(rf"{name} fit, $\alpha = {fit['alpha']:.3f} \pm "
                       rf"{fit['sigma']:.3f}$ ($\chi^2_\nu = {fit['chi2_nu']:.1f}$)"))

        sigma_ln = np.sqrt((1.0 - p_v) / err_arr[valid])
        resid = (np.log(p_v)
                 - np.log(fitted_probability(g[valid], fit["alpha"]))) / sigma_ln
        axr.plot(g[valid], resid, color=style["color"],
                 marker=style["marker"], markersize=3.5, linestyle="none",
                 alpha=0.8)

    for a in (ax, axr):
        a.axvspan(G_MIN, G_MAX, color="0.55", alpha=0.15, linewidth=0)
        a.grid(True, which="both", linestyle="--", alpha=0.3)
    axr.axhline(0, color="black", linewidth=0.8)
    axr.axhspan(-2, 2, color="0.55", alpha=0.12, linewidth=0)

    ax.set_yscale("log")
    ax.set_ylabel(r"$P(\mathrm{logical\ error} \mid g)$")
    plt.setp(ax.get_xticklabels(), visible=False)
    axr.set_xlabel(r"Absolute confidence $g = |\Delta|$ (dB)")
    axr.set_ylabel(r"resid. / $\sigma$", fontsize=13)
    axr.set_ylim(-6, 6)


def calibration_figure(binned, title, save_stem):
    fits = {name: full_fit(binned, name) for name in ("MWPM", "GNN")}
    fig = plt.figure(figsize=(7, 6.2))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08, figure=fig)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    _draw_calibration(ax, axr,
                      {n: binned[n] for n in ("MWPM", "GNN")}, fits)
    ax.legend(title=title, framealpha=0.9, loc="lower left", fontsize=10)

    for ext in ("pdf", "png"):
        path = f"{save_stem}.{ext}"
        fig.savefig(path, dpi=300)
        print(f"  saved {path}")
    plt.close(fig)
    return fits


def pooled_figure(pooled, per_config_fits, save_stem):
    fits = {name: full_fit(pooled, name) for name in ("MWPM", "GNN")}
    fig = plt.figure(figsize=(7, 6.2))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08, figure=fig)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    _draw_calibration(ax, axr,
                      {n: pooled[n] for n in ("MWPM", "GNN")}, fits)
    ax.legend(title="Pooled shots", framealpha=0.9, loc="lower left",
              fontsize=10)

    # Inset: per-config alpha +- unscaled sigma; the primary evidence.
    axins = inset_axes(ax, width="40%", height="25%", loc="upper right",
                       borderpad=0.8)
    xs = np.arange(len(CONFIGS))
    for name in ("MWPM", "GNN"):
        style = DECODER_STYLE[name]
        vals = np.array([per_config_fits[cfg][name]["alpha"] for cfg in CONFIGS])
        errs = np.array([per_config_fits[cfg][name]["sigma"] for cfg in CONFIGS])
        axins.errorbar(xs, vals, yerr=errs, color=style["color"],
                       marker=style["marker"], markersize=4, capsize=2,
                       linestyle="none")
        meta = meta_mean([per_config_fits[cfg][name] for cfg in CONFIGS])
        axins.axhline(meta["mean"], color=style["color"],
                      linestyle=style["linestyle"], linewidth=1.0, alpha=0.7)
    axins.axhline(1.0, color="black", linestyle=":", linewidth=1.2)
    axins.set_xticks(xs)
    axins.set_xticklabels([f"({d},{dt})" for d, dt in CONFIGS],
                          rotation=60, fontsize=7)
    axins.set_ylabel(r"$\alpha$", fontsize=11)
    axins.tick_params(axis="y", labelsize=8)
    axins.grid(True, linestyle="--", alpha=0.3)

    for ext in ("pdf", "png"):
        path = f"{save_stem}.{ext}"
        fig.savefig(path, dpi=300)
        print(f"  saved {path}")
    plt.close(fig)
    return fits


# ---------------------------------------------------------------------------
# Appendix: alpha vs physical error rate
# ---------------------------------------------------------------------------

def discover_runs():
    """
    Every (d = r, p, shots) run available: from the raw arrays when
    present, otherwise from the binned-count caches in figures3/derived/,
    so the figure regenerates without the raw data.
    """
    runs = set()
    pat = re.compile(r"delta_mwpm_d(\d+)_dt(\d+)_p([0-9.]+)_shots(\d+)\.npy$")
    for path in sorted(glob.glob(os.path.join(SAVE_DIR, "delta_mwpm_*.npy"))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        d, dt, p, shots = int(m[1]), int(m[2]), float(m[3]), int(m[4])
        gnn = os.path.join(SAVE_DIR,
                           f"delta_gnn_d{d}_dt{dt}_p{m[3]}_shots{shots}.npy")
        if os.path.isfile(gnn) and d == dt:
            runs.add((d, dt, p, shots))

    pat_cache = re.compile(r"calib_v5_d(\d+)_dt(\d+)_p([0-9.]+)_shots(\d+)\.json$")
    for path in sorted(glob.glob(os.path.join(DERIVED_DIR, "calib_v5_*.json"))):
        m = pat_cache.search(os.path.basename(path))
        if not m:
            continue
        d, dt, p, shots = int(m[1]), int(m[2]), float(m[3]), int(m[4])
        if d == dt:
            runs.add((d, dt, p, shots))

    return sorted(runs)


def alpha_vs_p_figure(save_stem):
    """ML-fitted alpha at every (d = r, p) run found on disk."""
    per_d = {}
    for d, dt, p, shots in discover_runs():
        binned = get_binned(d, dt, p, shots)
        row = {"p": p, "shots": shots}
        for name in ("MWPM", "GNN"):
            row[name] = full_fit(binned, name)
        per_d.setdefault(d, []).append(row)
        print(f"  d={d}, p={p} ({shots:,} shots): "
              + "  ".join(f"{n}: alpha={row[n]['alpha']:.3f}"
                          f"+-{row[n]['sigma']:.3f}"
                          f" (chi2_nu={row[n]['chi2_nu']:.1f})"
                          for n in ("MWPM", "GNN")))

    ds = sorted(k for k, v in per_d.items() if len(v) >= 2)
    if not ds:
        print("  fewer than two p-values per distance on disk; skipping figure")
        return per_d

    fig, axes = plt.subplots(1, len(ds), figsize=(4.2 * len(ds), 4.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, ds):
        rows = sorted(per_d[d], key=lambda r: r["p"])
        ps = np.array([r["p"] for r in rows])
        for name in ("MWPM", "GNN"):
            style = DECODER_STYLE[name]
            vals = np.array([r[name]["alpha"] for r in rows])
            errs = np.array([r[name]["sigma"] for r in rows])
            ax.errorbar(ps, vals, yerr=errs, color=style["color"],
                        marker=style["marker"], markersize=6,
                        linestyle=style["linestyle"], linewidth=1.4,
                        capsize=3, label=name)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2)
        ax.set_xlabel(r"Physical error rate $p$")
        ax.set_title(rf"$d = r = {d}$", fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel(r"Calibration slope $\alpha$")
    axes[0].legend(framealpha=0.9)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = f"{save_stem}.{ext}"
        fig.savefig(path, dpi=300)
        print(f"  saved {path}")
    plt.close(fig)
    return per_d


# ---------------------------------------------------------------------------
# Table II
# ---------------------------------------------------------------------------

def write_alpha_table(per_config_fits, pooled_fits):
    os.makedirs(TABLE_DIR, exist_ok=True)
    header = (" & ".join(rf"$({d}, {dt})$" for d, dt in CONFIGS)
              + r" & Pooled & IVW mean")

    def row(name):
        cells = []
        for cfg in CONFIGS:
            f = per_config_fits[cfg][name]
            cells.append(f"{f['alpha']:.3f}")
        f = pooled_fits[name]
        cells.append(f"{f['alpha']:.3f}")
        meta = meta_mean([per_config_fits[cfg][name] for cfg in CONFIGS])
        cells.append(f"{meta['mean']:.3f}")
        return " & ".join(cells)

    def chi_row(name):
        cells = [f"{per_config_fits[cfg][name]['chi2_nu']:.1f}"
                 for cfg in CONFIGS]
        cells.append(f"{pooled_fits[name]['chi2_nu']:.1f}")
        cells.append("--")
        return " & ".join(cells)

    lines = [
        r"\begin{table*}",
        r"\centering",
        rf"\begin{{tabular}}{{l| {' '.join('c' for _ in CONFIGS)} |c|c}}",
        r"\toprule",
        rf" & {header} \\",
        r"\hline",
        r"\midrule",
        rf"MWPM $\alpha$ & {row('MWPM')} \\",
        rf"MWPM $\chi^2_\nu$ & {chi_row('MWPM')} \\",
        r"\hline",
        rf"GNN $\alpha$ & {row('GNN')} \\",
        rf"GNN $\chi^2_\nu$ & {chi_row('GNN')} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Calibration slope $\alpha$ from a binomial "
        r"maximum-likelihood fit over the pre-specified window "
        rf"$g \in [{G_MIN}, {G_MAX}]$~dB, at $p = {P_MAIN}$, per "
        r"configuration $(d, r)$, for shot-pooled data, and as an "
        r"inverse-variance-weighted descriptive mean of the "
        r"per-configuration slopes (the configurations correspond to "
        r"different trained networks and do not share a single true "
        r"$\alpha$; the per-configuration values are the primary result). "
        r"The unscaled statistical ($1\sigma$, Fisher) uncertainties are at "
        r"most $0.001$ and are omitted; the reduced $\chi^2$ over integer-dB "
        r"display bins quantifies the goodness of fit and is reported "
        r"separately. Values of $\chi^2_\nu \gg 1$ indicate that the "
        r"one-parameter form does not describe the data within statistics.}",
        r"\label{alpha_table}",
        r"\end{table*}",
    ]
    path = os.path.join(TABLE_DIR, "appB_alpha.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    start = time.perf_counter()

    print("Binning saved runs (p = 0.005):")
    binned = {}
    for d, dt in CONFIGS:
        binned[(d, dt)] = get_binned(d, dt)

    per_config_fits = {}
    print("\nPer-configuration ML fits (primary evidence):")
    for cfg in CONFIGS:
        per_config_fits[cfg] = {name: full_fit(binned[cfg], name)
                                for name in ("MWPM", "GNN")}
        for name in ("MWPM", "GNN"):
            f = per_config_fits[cfg][name]
            print(f"  ({cfg[0]}, {cfg[1]}) {name}: alpha = {f['alpha']:.4f} "
                  f"+- {f['sigma']:.4f} (unscaled), chi2_nu = {f['chi2_nu']:.1f} "
                  f"over {f['n_bins']} bins, "
                  f"refinement shift = {f['refinement_shift']:.2e}")

    for name in ("MWPM", "GNN"):
        meta = meta_mean([per_config_fits[cfg][name] for cfg in CONFIGS])
        print(f"  IVW descriptive mean [{name}]: alpha = {meta['mean']:.4f} "
              f"+- {meta['sigma']:.4f}, range [{meta['min']:.3f}, "
              f"{meta['max']:.3f}], Q = {meta['Q']:.0f}, I2 = {meta['I2']:.3f}")

    print("\nWindow-robustness check (appendix):")
    wr = window_robustness(binned)
    with open(os.path.join(DERIVED_DIR, "window_robustness.json"), "w") as f:
        json.dump(wr, f, indent=1)
    write_window_table(wr)
    base_label = WINDOW_CHOICES[0][0]
    for label, _, _ in WINDOW_CHOICES:
        parts = []
        for name in ("MWPM", "GNN"):
            a = [wr[label][name][f"{d},{dt}"]["alpha"] for d, dt in CONFIGS]
            base = [wr[base_label][name][f"{d},{dt}"]["alpha"]
                    for d, dt in CONFIGS]
            shift = max(abs(x - y) for x, y in zip(a, base))
            parts.append(f"{name}: {min(a):.3f}-{max(a):.3f} "
                         f"(max shift {shift:.3f})")
        print(f"  {label:22s} {parts[0]}  |  {parts[1]}")

    print("\nIn-window relative deviation of display bins from the fit, "
          "|data/fit - 1| (plain-language size of the residual structure):")
    for name in ("MWPM", "GNN"):
        devs = [window_deviation(binned[cfg], name,
                                 per_config_fits[cfg][name]["alpha"])
                for cfg in CONFIGS]
        means = [dv[0] for dv in devs]
        print(f"  {name}: weighted-mean per-config range "
              f"{min(means):.1%} - {max(means):.1%}"
              + "".join(f"\n      ({cfg[0]}, {cfg[1]}): weighted mean "
                        f"{dv[0]:.1%}, max {dv[1]:.1%}"
                        for cfg, dv in zip(CONFIGS, devs)))

    print("\nFig. 4a:")
    d, dt = MAIN_CONFIG
    calibration_figure(binned[MAIN_CONFIG],
                       title=rf"$d={d}$, $r={dt}$, $p={P_MAIN}$",
                       save_stem=os.path.join(FIG_DIR, f"fig4a_calibration_d{d}"))

    for d, dt in APPB_CONFIGS:
        print(f"\nAppendix B calibration ({d}, {dt}):")
        calibration_figure(binned[(d, dt)],
                           title=rf"$d={d}$, $r={dt}$, $p={P_MAIN}$",
                           save_stem=os.path.join(
                               FIG_DIR, f"appB_calibration_d{d}_dt{dt}"))

    print("\nFig. 4b (summary across configurations):")
    pooled = {"config": {"pooled": True}}
    for name in ("MWPM", "GNN"):
        pooled[name] = {k: v.tolist() for k, v in
                        pool_counts(list(binned.values()), name).items()}
    pooled_fits = pooled_figure(pooled, per_config_fits,
                                save_stem=os.path.join(
                                    FIG_DIR, "fig4b_calibration_pooled"))

    write_alpha_table(per_config_fits, pooled_fits)

    print("\nAppendix: alpha vs p (auto-discovered runs):")
    alpha_vs_p_figure(os.path.join(FIG_DIR, "appD_alpha_vs_p"))

    print(f"\nTotal elapsed: {(time.perf_counter() - start) / 60:.1f} min")
