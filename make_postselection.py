"""
Fig. 1 and Table III: post-selection performance of the three native
decoder-confidence pairs, all evaluated on the SAME shots:

    MWPM_uncorr + g_uncorr (complementary gap)          (delta_mwpm_*)
    MWPM_corr   + g_corr   (gap on the shared            (delta_mwpmcorr_*, correlated_gap.py)
                            correlation-reweighted graph)
    GNN         + g_GNN    (logit)                       (delta_gnn_*)

Fig. 1, columns (a)-(c) for d = r = 5, 7, 9; colour = pair:
  top     post-selected LER vs acceptance rate kappa at p = 0.005, with a strip
          L_pair / L_GNN with PAIRED 95% bootstrap bands (same shots, tie-averaged);
  bottom  post-selected LER vs p at kappa = 0.9 (the p-scan of runs.py).

A point is shown when its curve rests on >= 20 accepted failures. Post-selection
is tie-averaged: when the acceptance threshold falls inside a block of shots with
equal confidence, the block's failures are accepted in proportion to the accepted
fraction of the block (Appendix A).

Outputs (in --out, default build/fig1/): fig1_postselection.pdf/.png,
tables/fig1a_counts.tex/.csv (Table III), tables/fig1a_eq3_decomposition.csv
(Eq. (4) factors vs kappa), tables/fig1b_points.csv; cached curves in derived/.
"""

import argparse
import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

import figure_style as fs
import runs

FIG_DIR = os.path.join(runs.BUILD_DIR, "fig1")
DERIVED_DIR = os.path.join(FIG_DIR, "derived")
TABLE_DIR = os.path.join(FIG_DIR, "tables")
RECOMPUTE = False

P_MAIN = runs.P_MAIN
SHOTS_MAIN = runs.SHOTS
DISTANCES = runs.PSCAN_DISTANCES
KAPPA_GRID = np.round(np.linspace(0.2, 1.0, 81), 4)
KAPPA_SHOW = np.round(np.arange(0.2, 1.0001, 0.1), 4)
KAPPA_1B = 0.9
MIN_FAIL = 20
N_BOOT, BOOT_SEED = 3000, 7

PAIRS = {n: (f"delta_{n}", fs.pair_label(n), fs.style(n))  # name -> (array prefix, label, style)
         for n in ("mwpm", "mwpmcorr", "gnn")}
plt.rcParams.update({"pdf.fonttype": 42, "savefig.bbox": "tight"})
# Fig. 1 is drawn at PRA two-column width (7.1 in), so its fonts are the printed sizes
FIG1_RC = {"font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9, "xtick.labelsize": 7.5,
           "ytick.labelsize": 7.5, "legend.fontsize": 8, "axes.linewidth": 0.6,
           "xtick.major.width": 0.6, "ytick.major.width": 0.6}
FIG1_MS, FIG1_LW = 3.6, 1.1


def load(name, d, r, p, shots):
    runs.require_arrays(d, r, p, shots, decoders=[name])
    return np.load(runs.array_path(name, d, r, p, shots))


# --------------------------------------------------------------------------
# Tie-averaged post-selection and the paired bootstrap
# --------------------------------------------------------------------------

def native_curve(delta_db, kappas):
    """
    Post-selected LER of one decoder ranked by its own |signed delta|,
    with FRACTIONAL TIE HANDLING: when the acceptance threshold falls
    inside a block of equal-confidence shots, the block's failures are
    accepted proportionally to the accepted fraction of the block
    (i.e., the expected LER under uniform random ordering of tied shots).
    Failure counts are therefore floats.
    """
    conf = np.abs(delta_db)
    order = np.argsort(-conf, kind="stable")
    conf_desc = conf[order]
    fail_desc = np.signbit(delta_db[order])   # a wrongly decided exact tie is stored as -0.0
    n = len(delta_db)

    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(conf_desc[1:], conf_desc[:-1], out=new[1:])
    block_start = np.flatnonzero(new)
    block_end = np.append(block_start[1:], n)
    cum = np.cumsum(fail_desc, dtype=np.float64)

    lers, fails, accepted = [], [], []
    for k in kappas:
        n_acc = int(np.floor(k * n))
        if n_acc < 1:
            lers.append(np.nan)
            fails.append(0.0)
            accepted.append(0)
            continue
        b = int(np.searchsorted(block_start, n_acc, side="right")) - 1
        s, e = int(block_start[b]), int(block_end[b])
        if n_acc >= e:
            f_acc = float(cum[n_acc - 1])
        else:
            f_before = float(cum[s - 1]) if s > 0 else 0.0
            f_acc = f_before + (float(cum[e - 1]) - f_before) \
                * (n_acc - s) / (e - s)
        lers.append(f_acc / n_acc)
        fails.append(f_acc)
        accepted.append(n_acc)
    return np.array(lers), np.array(fails), np.array(accepted, dtype=np.int64)


def _block_structure(conf):
    """Per-shot tie-block index in descending-confidence order."""
    n = len(conf)
    order = np.argsort(conf, kind="stable")
    conf_desc = conf[order][::-1].copy()
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(conf_desc[1:], conf_desc[:-1], out=new[1:])
    block_start = np.flatnonzero(new)
    block_end = np.append(block_start[1:], n)
    bid_desc = (np.cumsum(new) - 1).astype(np.int32)
    block_of_shot = np.empty(n, dtype=np.int32)
    block_of_shot[order[::-1]] = bid_desc
    return block_of_shot, block_start, block_end


def paired_ratio_ci(dm, dg, kappas, n_boot, seed):
    """
    Paired percentile CI for LER_MWPM(kappa) / LER_GNN(kappa) (native
    decoder-score pairs) via a multinomial bootstrap over 36 joint
    per-shot categories: (fail_M, fail_G, tie-zone under g_MWPM,
    tie-zone under g_GNN), where the tie-zone is strictly-above /
    inside / strictly-below the threshold tie block. The tie-averaging
    rule of the point estimate is reproduced inside every replicate, so
    the interval refers to the tie-averaged estimand.
    """
    n = len(dm)
    rng = np.random.default_rng(seed)
    fail_m = np.signbit(dm)   # a wrongly decided exact tie is stored as -0.0
    fail_g = np.signbit(dg)
    bm = _block_structure(np.abs(dm))
    bg = _block_structure(np.abs(dg))

    idx = np.arange(36)
    fm_mask = (idx // 18) == 1
    fg_mask = ((idx // 9) % 2) == 1
    zm_idx = (idx // 3) % 3
    zg_idx = idx % 3

    lo_list, hi_list = [], []
    for k in kappas:
        n_acc = int(np.floor(k * n))
        zones = []
        for b_of_shot, b_start, b_end in (bm, bg):
            tb = int(np.searchsorted(b_start, n_acc, side="right")) - 1
            if n_acc >= b_end[tb]:
                tb += 1
            zones.append(
                (np.clip(b_of_shot - tb, -1, 1) + 1).astype(np.int8))

        cat = (fail_m.astype(np.int8) * 18 + fail_g.astype(np.int8) * 9
               + zones[0] * 3 + zones[1])
        counts = np.bincount(cat, minlength=36).astype(np.float64)
        draws = rng.multinomial(n, counts / n, size=n_boot).astype(np.float64)

        def acc_fails(f_mask, z_idx):
            above = f_mask & (z_idx == 0)
            inblk = f_mask & (z_idx == 1)
            n_above = draws[:, z_idx == 0].sum(axis=1)
            m_blk = draws[:, z_idx == 1].sum(axis=1)
            frac = np.clip((n_acc - n_above) / np.maximum(m_blk, 1), 0.0, 1.0)
            return (draws[:, above].sum(axis=1)
                    + frac * draws[:, inblk].sum(axis=1))

        f_m = acc_fails(fm_mask, zm_idx)
        f_g = acc_fails(fg_mask, zg_idx)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = f_m / f_g
        ratios = ratios[np.isfinite(ratios)]
        if len(ratios) < n_boot // 2:
            lo_list.append(np.nan)
            hi_list.append(np.nan)
        else:
            lo_list.append(float(np.percentile(ratios, 2.5)))
            hi_list.append(float(np.percentile(ratios, 97.5)))
        del cat
    return np.array(lo_list), np.array(hi_list)


def ler_at_kappa(delta_db, kappa):
    """
    Exact tie-averaged post-selected LER at one kappa; O(N), no full sort.
    The threshold value v is found by partial sort; shots strictly above v
    are accepted, and the tie block at v is accepted fractionally.
    """
    n = len(delta_db)
    n_acc = int(np.floor(kappa * n))
    conf = np.abs(delta_db)
    v = np.partition(conf, n - n_acc)[n - n_acc]
    fail = np.signbit(delta_db)   # a wrongly decided exact tie is stored as -0.0
    above = conf > v
    n_above = int(above.sum())
    f_above = float((fail & above).sum())
    at = conf == v
    m_v = int(at.sum())
    f_v = float((fail & at).sum())
    f_acc = f_above + f_v * (n_acc - n_above) / m_v
    return f_acc / n_acc, f_acc, n_acc


# --------------------------------------------------------------------------
# Fig. 1, panels (a)-(c): L(kappa) at p = 0.005
# --------------------------------------------------------------------------

def compute_1a(d):
    arrays = {n: load(n, d, d, P_MAIN, SHOTS_MAIN) for n in PAIRS}
    out = {"config": {"d": d, "r": d, "p": P_MAIN, "shots": SHOTS_MAIN}, "kappa": KAPPA_GRID.tolist(),
           "pairs": {}, "ratio_vs_gnn": {}}
    for n, a in arrays.items():
        ler, fails, acc = native_curve(a, KAPPA_GRID)
        out["pairs"][n] = {"ler": ler.tolist(), "n_fail": fails.tolist(), "n_accepted": acc.tolist()}
    if "gnn" in arrays:
        for n in ("mwpm", "mwpmcorr"):
            if n in arrays:
                lo, hi = paired_ratio_ci(arrays[n], arrays["gnn"], KAPPA_SHOW, N_BOOT, BOOT_SEED)
                out["ratio_vs_gnn"][n] = {"kappa": KAPPA_SHOW.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
    return out


def cached(name, fn, key):
    os.makedirs(DERIVED_DIR, exist_ok=True)
    path = os.path.join(DERIVED_DIR, f"{name}.json")
    if not RECOMPUTE and os.path.exists(path):
        res = json.load(open(path))
        if res.get("_key") == key:
            return res
    res = fn()
    res["_key"] = key
    json.dump(res, open(path, "w"))
    return res


def at(res, name, kappas):
    k = np.array(res["kappa"])
    idx = [int(np.argmin(np.abs(k - x))) for x in kappas]
    c = res["pairs"][name]
    return (np.array(c["ler"])[idx], np.array(c["n_fail"])[idx], np.array(c["n_accepted"])[idx])


def _band_line(ax, x, y, nacc, n):
    """Curve of pair n with its binomial 95% band."""
    style = PAIRS[n][2]
    ci = 1.96 * np.sqrt(y * (1 - y) / nacc)
    ax.fill_between(x, np.maximum(y - ci, 1e-12), y + ci, color=style["color"], alpha=0.16, linewidth=0)
    ax.plot(x, y, markersize=FIG1_MS, linewidth=FIG1_LW, **style)


def _fig1_axes_style(ax):
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="-", alpha=0.18, linewidth=0.5)
    ax.grid(True, which="minor", linestyle="-", alpha=0.07, linewidth=0.4)
    ax.tick_params(which="both", direction="in", top=True, right=True)


def plot_fig1(results, res_1b):
    """Panels (a)-(c), one column per distance: L(kappa) with an L_pair / L_GNN strip, and L(p) below."""
    with plt.rc_context(FIG1_RC):
        fig = plt.figure(figsize=(7.1, 5.6))
        gs = fig.add_gridspec(4, len(results), height_ratios=[3, 1.1, 0.55, 3], hspace=0.07, wspace=0.12)
        top = [fig.add_subplot(gs[0, c]) for c in range(len(results))]
        rat = [fig.add_subplot(gs[1, c], sharex=top[c]) for c in range(len(results))]
        bot = [fig.add_subplot(gs[3, c]) for c in range(len(results))]
        for c, res in enumerate(results):
            d = res["config"]["d"]
            ax, axr, axb = top[c], rat[c], bot[c]
            for n in PAIRS:
                if n in res["pairs"]:
                    ler, nf, nacc = at(res, n, KAPPA_SHOW)
                    ok = nf >= MIN_FAIL
                    _band_line(ax, KAPPA_SHOW[ok], ler[ok], nacc[ok], n)
            _fig1_axes_style(ax)
            ax.set_title(rf"$\mathbf{{({'abc'[c]})}}$   $d = r = {d}$", loc="left")
            ax.set_xlim(0.15, 1.05)
            ax.tick_params(labelbottom=False)
            for n_, rr in res["ratio_vs_gnn"].items():
                style = PAIRS[n_][2]
                ler_n, nf_n, _ = at(res, n_, KAPPA_SHOW)
                ler_g, nf_g, _ = at(res, "gnn", KAPPA_SHOW)
                ok = (nf_n >= MIN_FAIL) & (nf_g >= MIN_FAIL)
                lo, hi = np.array(rr["lo"], dtype=float), np.array(rr["hi"], dtype=float)
                band = ok & np.isfinite(lo) & np.isfinite(hi)
                axr.fill_between(KAPPA_SHOW[band], lo[band], hi[band], color=style["color"], alpha=0.2, linewidth=0)
                axr.plot(KAPPA_SHOW[ok], (ler_n / ler_g)[ok], markersize=2.8, linewidth=1.0, **style)
            axr.axhline(1.0, color="black", linewidth=0.7)
            _fig1_axes_style(axr)
            axr.set_ylim(0.12, 8)
            axr.set_xlabel(r"Acceptance rate $\kappa$")
            rows = sorted([r for r in res_1b["rows"] if r["d"] == d], key=lambda r: r["p"])
            for n in PAIRS:
                pts = [(r["p"], r[n]) for r in rows if n in r and r[n]["n_fail"] >= MIN_FAIL]
                if pts:
                    _band_line(axb, np.array([x[0] for x in pts]), np.array([x[1]["ler"] for x in pts]),
                               np.array([x[1]["n_accepted"] for x in pts]), n)
            _fig1_axes_style(axb)
            axb.set_xscale("log")
            axb.set_xlim(0.00293, 0.0051)
            axb.xaxis.set_major_locator(FixedLocator([0.003, 0.0035, 0.004, 0.0045, 0.005]))
            axb.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 1e3:g}"))
            axb.xaxis.set_minor_locator(FixedLocator([]))
            axb.set_xlabel(r"Physical error rate $p$ ($10^{-3}$)")
        for row in (top, rat, bot):  # one y-range per row
            lo = min(a.get_ylim()[0] for a in row)
            hi = max(a.get_ylim()[1] for a in row)
            for i, a in enumerate(row):
                a.set_ylim(lo, hi)
                if i:
                    a.tick_params(labelleft=False)
        for a in rat:
            a.yaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5]))
            a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            a.yaxis.set_minor_formatter(NullFormatter())
        top[0].set_ylabel(rf"$L(\kappa)$ at $p = {P_MAIN}$")
        rat[0].set_ylabel(r"$L_{\rm pair}/L_{\rm GNN}$")
        bot[0].set_ylabel(rf"$L$ at $\kappa = {KAPPA_1B}$")
        fig.align_ylabels([top[0], rat[0], bot[0]])
        handles = [Line2D([], [], markersize=FIG1_MS + 0.6, linewidth=FIG1_LW, **PAIRS[n][2]) for n in PAIRS]
        fig.legend(handles, [PAIRS[n][1] for n in PAIRS], loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
        fig.subplots_adjust(top=0.9)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIG_DIR, f"fig1_postselection.{ext}"), dpi=300)
        plt.close(fig)


# --------------------------------------------------------------------------
# Fig. 1, bottom row: L(p) at kappa = 0.9
# --------------------------------------------------------------------------

def runs_1b():
    """The p-scan runs (d, p, shots) of Fig. 1, bottom row."""
    return [(d, p, shots) for d in DISTANCES for _, _, p, shots in runs.pscan_runs(d)]


def compute_1b():
    rows = []
    for d, p, shots in runs_1b():
        row = {"d": d, "p": p, "shots": shots}
        for n in PAIRS:
            a = load(n, d, d, p, shots)
            ler, nf, nacc = ler_at_kappa(a, KAPPA_1B)
            row[n] = {"ler": ler, "n_fail": nf, "n_accepted": nacc}
        rows.append(row)
    return {"rows": rows}


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def write_tables(results, res_1b):
    os.makedirs(TABLE_DIR, exist_ok=True)
    with open(os.path.join(TABLE_DIR, "fig1a_counts.csv"), "w") as f:
        f.write("d,decoder," + ",".join(f"k{k:.1f}" for k in KAPPA_SHOW) + "\n")
        for res in results:
            for n in PAIRS:
                if n in res["pairs"]:
                    _, nf, _ = at(res, n, KAPPA_SHOW)
                    f.write(f"{res['config']['d']},{n}," + ",".join(f"{v:.1f}" for v in nf) + "\n")
    with open(os.path.join(TABLE_DIR, "fig1a_eq3_decomposition.csv"), "w") as f:
        f.write("d,decoder,kappa,pair_ratio_vs_gnn,hard_factor,selection_gain_ratio,ci_lo,ci_hi\n")
        for res in results:
            if "gnn" not in res["pairs"]:
                continue
            lg, _, _ = at(res, "gnn", KAPPA_SHOW)
            for n in ("mwpm", "mwpmcorr"):
                if n not in res["pairs"]:
                    continue
                ln, _, _ = at(res, n, KAPPA_SHOW)
                rr = res["ratio_vs_gnn"].get(n, {})
                for i, k in enumerate(KAPPA_SHOW):
                    hard = ln[-1] / lg[-1]
                    pair = ln[i] / lg[i]
                    f.write(f"{res['config']['d']},{n},{k:.1f},{pair:.4f},{hard:.4f},{pair / hard:.4f},"
                            f"{rr.get('lo', [np.nan] * 9)[i]:.4f},{rr.get('hi', [np.nan] * 9)[i]:.4f}\n")
    # LaTeX: accepted failures behind every point of Fig. 1(a)-(c), three pairs
    lines = [r"\begin{table*}", r"\centering",
             rf"\begin{{tabular}}{{ll| {' '.join('c' for _ in KAPPA_SHOW)}}}", r"\toprule",
             rf" & & \multicolumn{{{len(KAPPA_SHOW)}}}{{c}}{{Accepted failures at acceptance rate $\kappa$}} \\",
             "Config & Pair & " + " & ".join(f"{k:.1f}" for k in KAPPA_SHOW) + r" \\", r"\hline", r"\midrule"]
    for i, res in enumerate(results):
        d = res["config"]["d"]
        for n in PAIRS:
            if n in res["pairs"]:
                _, nf, _ = at(res, n, KAPPA_SHOW)
                lines.append(rf"$({d}, {d})$ & {PAIRS[n][1]} & " + " & ".join(f"{v:.0f}" for v in nf) + r" \\")
        if i < len(results) - 1:
            lines.append(r"\hline")
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{Number of accepted logical failures behind each $L(\kappa)$ point of Fig.~1(a)--(c), at $p = {P_MAIN}$ "
              r"with $10^8$ shots per configuration (the same shots for all three pairs). Acceptance-threshold "
              r"ties are averaged uniformly (fractionally accepted), so the entries are tie-averaged "
              rf"expectations, rounded to integers. Points resting on fewer than {MIN_FAIL} failures are "
              r"suppressed in the figure.}", r"\label{counts_table}", r"\end{table*}"]
    with open(os.path.join(TABLE_DIR, "fig1a_counts.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(TABLE_DIR, "fig1b_points.csv"), "w") as f:
        f.write("d,p,shots,decoder,ler,n_fail,n_accepted\n")
        for r in res_1b["rows"]:
            for n in PAIRS:
                if n in r:
                    f.write(f"{r['d']},{r['p']},{r['shots']},{n},{r[n]['ler']:.6e},{r[n]['n_fail']:.1f},"
                            f"{r[n]['n_accepted']}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fig. 1 and Table III: three native pairs")
    ap.add_argument("--out", default=FIG_DIR)
    FIG_DIR = ap.parse_args().out
    DERIVED_DIR = os.path.join(FIG_DIR, "derived")
    TABLE_DIR = os.path.join(FIG_DIR, "tables")
    os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.perf_counter()
    results = []
    for d in DISTANCES:
        key = [*PAIRS, runs.tag(d, d)]
        results.append(cached(f"fig1a_d{d}", lambda d=d: compute_1a(d), key=key))
    key_1b = [f"d{d}_p{p}_shots{shots}" for d, p, shots in runs_1b()]
    res_1b = cached("fig1b", compute_1b, key=key_1b)
    plot_fig1(results, res_1b)
    write_tables(results, res_1b)
    for res in results:
        if "gnn" in res["pairs"] and "mwpmcorr" in res["pairs"]:
            lg, _, _ = at(res, "gnn", [0.9, 1.0])
            lc, _, _ = at(res, "mwpmcorr", [0.9, 1.0])
            lu, _, _ = at(res, "mwpm", [0.9, 1.0])
            print(f"  d={res['config']['d']} kappa=0.9: L_corr/L_GNN {lc[0] / lg[0]:.3f} "
                  f"(hard {lc[1] / lg[1]:.3f} x selection {(lc[0] / lc[1]) / (lg[0] / lg[1]):.3f}); "
                  f"L_uncorr/L_GNN {lu[0] / lg[0]:.3f}")
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min -> {FIG_DIR}")
