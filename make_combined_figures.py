"""
Figures that show BOTH MWPM variants (correlated and uncorrelated) next to the
GNN, on the same shots:

  fig2_crossed.{pdf,png}   Fig. 2: exchanged confidence scores at p = 0.005, one column per
                           distance d = r (default 5, 7, 9); top row MWPM_corr vs GNN, bottom
                           row MWPM_uncorr vs GNN; solid = each decoder ranked by its own
                           score, dashed = by the other decoder's score
  fig4_calibration.{pdf,png}  Fig. 5: calibration of MWPM_corr and the GNN with residuals, at (9, 9) and
                           for the shots of all configurations pooled; inset: the slope alpha of every
                           configuration for the three scores
  appC_calibration_d5_d7.{pdf,png}  Fig. 6: the same at (5, 5) and (7, 7)
  appD_alpha_vs_p_combined.{pdf,png}  Fig. 9: calibration slope alpha vs p for MWPM_corr,
                           MWPM_uncorr and the GNN, one column per distance (full page width)

Inputs: the cached results of make_exchange.py and make_calibration.py run with
--mwpm mwpm / --mwpm mwpmcorr and --out <build>/<variant> (make_all.py).
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

import figure_style as fs
import make_exchange as mf2
import make_calibration as mf4
import runs
from make_postselection import FIG1_LW, FIG1_MS, FIG1_RC, _fig1_axes_style

ROWS = ("mwpmcorr", "mwpm")  # Fig. 2 rows: MWPM variant exchanged with the GNN
CURVES = (("decoder=mwpm,rank=g_mwpm", None, "-"), ("decoder=mwpm,rank=g_gnn", None, "--"),
          ("decoder=gnn,rank=g_gnn", "gnn", "-"), ("decoder=gnn,rank=g_mwpm", "gnn", "--"))


def crossed(build, variant, d):
    return json.load(open(os.path.join(build, variant, "derived", f"crossed_d{d}_dt{d}_p{mf2.P}.json")))


def plot_fig2(build, distances, save_stem):
    kappa_show = mf2.KAPPA_SHOW
    with plt.rc_context(FIG1_RC):
        fig, axes = plt.subplots(len(ROWS), len(distances), figsize=(7.1, 2.25 * len(ROWS) + 0.45),
                                 sharex=True, squeeze=False, gridspec_kw=dict(hspace=0.08, wspace=0.12))
        for r, variant in enumerate(ROWS):
            for c, d in enumerate(distances):
                ax, res = axes[r, c], crossed(build, variant, d)
                if res is None:
                    continue
                kappa = np.array(res["kappa"])
                idx = np.array([int(np.argmin(np.abs(kappa - k))) for k in kappa_show])
                # a kappa point is shown only if all four curves of the panel rest on enough failures
                ok = np.ones(len(kappa_show), dtype=bool)
                for key, _, _ in CURVES:
                    ok &= np.array(res["curves"][key]["n_fail_accepted"], dtype=float)[idx] >= mf2.MIN_ACCEPTED_FAILURES
                for key, dec, ls in CURVES:
                    cur = res["curves"][key]
                    ler = np.array(cur["ler"], dtype=float)[idx][ok]
                    n_acc = np.array(cur["n_accepted"], dtype=float)[idx][ok]
                    style = {**fs.style(dec or variant), "linestyle": ls}
                    ci = 1.96 * np.sqrt(ler * (1 - ler) / n_acc)
                    ax.fill_between(kappa_show[ok], np.maximum(ler - ci, 1e-12), ler + ci, color=style["color"],
                                    alpha=0.16, linewidth=0)
                    ax.plot(kappa_show[ok], ler, markersize=FIG1_MS, linewidth=FIG1_LW, **style)
                _fig1_axes_style(ax)
                ax.set_xlim(0.15, 1.05)
                if r == 0:
                    ax.set_title(rf"$\mathbf{{({'abcdef'[c]})}}$   $d = r = {d}$", loc="left")
                if r == len(ROWS) - 1:
                    ax.set_xlabel(r"Acceptance rate $\kappa$")
            axes[r, 0].set_ylabel(f"{fs.NAME[variant]} vs GNN\n" + rf"$L(\kappa)$ at $p = {mf2.P}$")
        lo = min(a.get_ylim()[0] for a in axes.flat)
        hi = max(a.get_ylim()[1] for a in axes.flat)
        for (r, c), a in np.ndenumerate(axes):
            a.set_ylim(lo, hi)
            if c:
                a.tick_params(labelleft=False)
        fig.align_ylabels(axes[:, 0])
        handles = [Line2D([], [], markersize=FIG1_MS + 0.6, linewidth=FIG1_LW, **{**fs.style(n), "linestyle": "-"})
                   for n in ("mwpmcorr", "mwpm", "gnn")]
        handles += [Line2D([], [], color="0.25", linewidth=FIG1_LW, linestyle=ls) for ls in ("-", "--")]
        labels = [fs.NAME[n] for n in ("mwpmcorr", "mwpm", "gnn")] + [
            "ranked by its own score", "ranked by the other decoder's score"]
        fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0),
                   handlelength=2.4, columnspacing=1.2)
        fig.subplots_adjust(top=0.87)
        for ext in ("pdf", "png"):
            fig.savefig(f"{save_stem}.{ext}", dpi=300)
        plt.close(fig)
    print(f"  saved {save_stem}.pdf/.png")


def alpha_vs_p(build, save_stem):
    """ML-fitted alpha at every (d = r, p) run, for both MWPM variants and the GNN."""
    per_d = {}
    for variant in ("mwpmcorr", "mwpm"):
        mf4.configure(variant, os.path.join(build, variant))
        for d, dt, p, shots in mf4.pscan_runs():
            b = mf4.get_binned(d, dt, p, shots)
            row = per_d.setdefault(d, {}).setdefault(p, {})
            row[variant] = mf4.full_fit(b, "MWPM")
            row["gnn"] = mf4.full_fit(b, "GNN")
    ds = sorted(d for d, v in per_d.items() if len(v) >= 2)
    # drawn like Figs. 1-3: one column per distance at PRA two-column width, p axis as in Fig. 1
    with plt.rc_context(FIG1_RC):
        fig, axes = plt.subplots(1, len(ds), figsize=(7.1, 2.55), sharey=True, squeeze=False,
                                 gridspec_kw=dict(wspace=0.12))
        for c, (ax, d) in enumerate(zip(axes[0], ds)):
            ps = sorted(per_d[d])
            for n in ("mwpmcorr", "mwpm", "gnn"):
                pts = [(p, per_d[d][p][n]) for p in ps if n in per_d[d][p]]
                ax.errorbar([x[0] for x in pts], [x[1]["alpha"] for x in pts], yerr=[x[1]["sigma"] for x in pts],
                            markersize=FIG1_MS, linewidth=FIG1_LW, elinewidth=0.8, capsize=1.5, label=fs.NAME[n],
                            **fs.style(n))
            ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8)
            ax.grid(True, which="major", linestyle="-", alpha=0.18, linewidth=0.5)
            ax.tick_params(which="both", direction="in", top=True, right=True)
            ax.set_xscale("log")
            ax.set_xlim(0.00293, 0.0051)
            ax.xaxis.set_major_locator(FixedLocator([0.003, 0.0035, 0.004, 0.0045, 0.005]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 1e3:g}"))
            ax.xaxis.set_minor_locator(FixedLocator([]))
            ax.set_xlabel(r"Physical error rate $p$ ($10^{-3}$)")
            ax.set_title(rf"$\mathbf{{({'abcdef'[c]})}}$   $d = r = {d}$", loc="left")
        axes[0, 0].set_ylabel(r"Calibration slope $\alpha$")
        handles = [Line2D([], [], markersize=FIG1_MS + 0.6, linewidth=FIG1_LW, **fs.style(n))
                   for n in ("mwpmcorr", "mwpm", "gnn")]
        fig.legend(handles, [fs.NAME[n] for n in ("mwpmcorr", "mwpm", "gnn")], loc="upper center", ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.subplots_adjust(top=0.8)
        for ext in ("pdf", "png"):
            fig.savefig(f"{save_stem}.{ext}", dpi=300)
        plt.close(fig)
    print(f"  saved {save_stem}.pdf/.png")


CALIB_NAMES = (("MWPM", "mwpmcorr"), ("GNN", "gnn"))
RESID_LIM = 6.0


def _calibration_panel(ax, axr, binned, fits):
    """Empirical P(fail | g) with Wilson intervals and the ML fit of Eq. (5); residuals beneath
    (points beyond +-RESID_LIM are drawn at the edge as open markers)."""
    for name, key in CALIB_NAMES:
        style = fs.style(key)
        g, p_fail, counts, errs, valid = mf4.display_arrays(binned[name]["counts"], binned[name]["err_counts"])
        lo, hi = mf4.wilson_ci(errs[valid], counts[valid])
        gv, pv = g[valid], p_fail[valid]
        ax.errorbar(gv, pv, yerr=[pv - lo, hi - pv], color=style["color"], marker=style["marker"],
                    markersize=FIG1_MS - 1.4, linestyle="none", elinewidth=0.6, capsize=0, alpha=0.85)
        g_fit = np.linspace(0, gv.max(), 400)
        ax.plot(g_fit, mf4.fitted_probability(g_fit, fits[name]["alpha"]), color=style["color"],
                linestyle=style["linestyle"], linewidth=FIG1_LW)
        resid = (np.log(pv) - np.log(mf4.fitted_probability(gv, fits[name]["alpha"]))) / np.sqrt((1 - pv) / errs[valid])
        inside = np.abs(resid) <= RESID_LIM
        axr.plot(gv[inside], resid[inside], linestyle="none", marker=style["marker"], markersize=2.2,
                 color=style["color"], alpha=0.85)
        axr.plot(gv[~inside], np.clip(resid[~inside], -RESID_LIM, RESID_LIM), linestyle="none",
                 marker=style["marker"], markersize=2.2, markerfacecolor="none", markeredgecolor=style["color"],
                 markeredgewidth=0.6)
    for a in (ax, axr):
        a.axvspan(mf4.G_MIN, mf4.G_MAX, color="0.55", alpha=0.12, linewidth=0)
        a.grid(True, which="major", linestyle="-", alpha=0.18, linewidth=0.5)
        a.tick_params(which="both", direction="in", top=True, right=True)
    axr.axhline(0, color="black", linewidth=0.6)
    axr.axhspan(-2, 2, color="0.55", alpha=0.12, linewidth=0)
    axr.set_ylim(-1.2 * RESID_LIM, 1.2 * RESID_LIM)
    ax.set_yscale("log")
    ax.tick_params(labelbottom=False)
    text = "\n".join(rf"{fs.NAME[key]}: $\alpha = {fits[name]['alpha']:.3f}$, $\chi^2_\nu = {fits[name]['chi2_nu']:.1f}$"
                     for name, key in CALIB_NAMES)
    ax.text(0.97, 0.96, text, transform=ax.transAxes, fontsize=7, va="top", ha="right",
            bbox=dict(facecolor="white", edgecolor="0.8", linewidth=0.5, alpha=0.9, boxstyle="round,pad=0.3"))


def _slope_inset(ax, per_config):
    """Per-configuration slope alpha of Eq. (5) for the three scores, as an inset in the empty lower-left
    corner of a calibration panel; per_config: {key: {(d, r): fit}}."""
    ins = ax.inset_axes([0.13, 0.14, 0.4, 0.25])
    configs = list(next(iter(per_config.values())))
    xs = np.arange(len(configs))
    for key, fits in per_config.items():
        style = fs.style(key)
        ins.plot(xs, [fits[c]["alpha"] for c in configs], linestyle="none", marker=style["marker"],
                 color=style["color"], markersize=2.4)
    ins.axhline(1.0, color="black", linestyle=":", linewidth=0.6)
    ins.set_xlim(-0.5, len(configs) - 0.5)
    ins.set_xticks(xs)
    ins.set_xticklabels([f"({d},{r})" for d, r in configs], rotation=90, fontsize=5)
    ins.tick_params(which="both", direction="in", labelsize=5, length=2, pad=1.5)
    ins.grid(True, which="major", linestyle="-", alpha=0.18, linewidth=0.4)
    ins.set_ylabel(r"$\alpha$ per config.", fontsize=6, labelpad=1)
    ins.patch.set_alpha(0.92)


def calibration_grid(panels, save_stem, slopes=None):
    """panels: [(title, binned)], one calibration column each (MWPM_corr and GNN); slopes: per-configuration
    fits {key: {(d, r): fit}} drawn as an inset in the last panel."""
    ncol = len(panels)
    with plt.rc_context(FIG1_RC):
        fig = plt.figure(figsize=(7.1, 3.7))
        gs = fig.add_gridspec(2, ncol, height_ratios=[3, 1.1], hspace=0.06, wspace=0.1)
        top = [fig.add_subplot(gs[0, c]) for c in range(len(panels))]
        bot = [fig.add_subplot(gs[1, c], sharex=top[c]) for c in range(len(panels))]
        for c, (title, binned) in enumerate(panels):
            fits = {name: mf4.full_fit(binned, name) for name, _ in CALIB_NAMES}
            _calibration_panel(top[c], bot[c], binned, fits)
            top[c].set_title(rf"$\mathbf{{({'abcd'[c]})}}$   {title}", loc="left")
            bot[c].set_xlabel(r"Confidence $g$ (dB)")
        for row in (top, bot):
            lo = min(a.get_ylim()[0] for a in row)
            hi = max(a.get_ylim()[1] for a in row)
            for i, a in enumerate(row):
                a.set_ylim(lo, hi)
                if i:
                    a.tick_params(labelleft=False)
        top[0].set_ylabel(r"$P(\mathrm{logical\ error} \mid g)$")
        bot[0].set_ylabel(r"Resid./$\sigma$")
        fig.align_ylabels([top[0], bot[0]])
        keys = [key for _, key in CALIB_NAMES]
        if slopes is not None:
            _slope_inset(top[-1], slopes)
            keys = ["mwpmcorr", "mwpm", "gnn"]
        handles = [Line2D([], [], markersize=FIG1_MS, linewidth=FIG1_LW, **fs.style(key)) for key in keys]
        labels = [fs.NAME[key] for key in keys]
        fig.legend(handles, labels, loc="upper center", ncol=len(keys), frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.subplots_adjust(top=0.87)
        for ext in ("pdf", "png"):
            fig.savefig(f"{save_stem}.{ext}", dpi=300)
        plt.close(fig)
    print(f"  saved {save_stem}.pdf/.png")


def _binned_main(build, variant):
    """Cached calibration counts of make_calibration.py at p = 0.005 for every configuration of `variant`."""
    mf4.configure(variant, os.path.join(build, variant))
    return {(d, r): json.load(open(os.path.join(
        mf4.DERIVED_DIR, f"calib_d{d}_dt{r}_p{mf4.P_MAIN}_shots{mf4.SHOTS_MAIN}.json"))) for d, r in mf4.CONFIGS}


def calibration_figures(build, out, suffix=""):
    """Fig. 5 [calibration at (9, 9); calibration of the shots of all configurations pooled, with the alpha of
    every configuration for the three scores as an inset] and Fig. 6 [(5, 5) and (7, 7)]."""
    corr, uncorr = _binned_main(build, "mwpmcorr"), _binned_main(build, "mwpm")
    slopes = {"mwpmcorr": {c: mf4.full_fit(b, "MWPM") for c, b in corr.items()},
              "mwpm": {c: mf4.full_fit(b, "MWPM") for c, b in uncorr.items()},
              "gnn": {c: mf4.full_fit(b, "GNN") for c, b in corr.items()}}
    pooled = {name: {k: v.tolist() for k, v in mf4.pool_counts(list(corr.values()), name).items()}
              for name, _ in CALIB_NAMES}   # summed counts = concatenated shots (as in make_calibration.py)
    for name, key in CALIB_NAMES:
        f = mf4.full_fit(pooled, name)
        print(f"  pooled {key}: alpha = {f['alpha']:.4f}, chi2_nu = {f['chi2_nu']:.1f} ({len(corr)} configurations)")
    calibration_grid([(r"$(d, r) = (9, 9)$", corr[(9, 9)]), ("all configurations, pooled", pooled)],
                     os.path.join(out, f"fig4_calibration{suffix}"), slopes=slopes)
    calibration_grid([(r"$(d, r) = (5, 5)$", corr[(5, 5)]), (r"$(d, r) = (7, 7)$", corr[(7, 7)])],
                     os.path.join(out, f"appC_calibration_d5_d7{suffix}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=runs.BUILD_DIR)
    ap.add_argument("--out", default=os.path.join(runs.BUILD_DIR, "combined"))
    ap.add_argument("--distances", type=int, nargs="+", default=[5, 7, 9], help="Fig. 2 columns (d = r)")
    ap.add_argument("--steps", nargs="+", choices=["fig2", "alpha", "calib"], default=["fig2", "alpha", "calib"])
    ap.add_argument("--suffix", default="", help="appended to the output file names (previews)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "savefig.bbox": "tight"})
    if "fig2" in a.steps:
        plot_fig2(a.build, a.distances, os.path.join(a.out, f"fig2_crossed{a.suffix}"))
    if "alpha" in a.steps:
        alpha_vs_p(a.build, os.path.join(a.out, f"appD_alpha_vs_p_combined{a.suffix}"))
    if "calib" in a.steps:
        calibration_figures(a.build, a.out, a.suffix)


if __name__ == "__main__":
    main()
