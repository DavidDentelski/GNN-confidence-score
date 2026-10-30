"""
Table VI (Appendix D) and Table VII (Appendix E): anatomy and robustness of the
discrimination results.

Two checks, computed from the shot-aligned per-shot arrays:

1. TRIVIAL-SYNDROME ROBUSTNESS. Trivial syndromes bypass the GNN and are
   assigned, as a sentinel confidence, the MWPM gap of the trivial
   syndrome (the maximal matching gap). This script identifies the
   trivial shots exactly (they are the only shots whose signed deltas are
   bit-identical between the two arrays, since the sentinel is copied
   from the MWPM computation), reports per config:
       - the trivial fraction,
       - the number of trivial shots carrying a logical error,
       - the assigned hard decision (0) and confidence value,
   and recomputes the AUC table and (for the d = 5 family, where the
   trivial fraction is macroscopic) the crossed anchor-kappa LERs with
   the trivial shots EXCLUDED.

2. PAIRED AUC DIFFERENCES. For each decoder D, the difference
       dAUC_D = AUC(g_own -> D) - AUC(g_other -> D)
   with a paired DeLong standard error (the two scores are evaluated on
   the same shots, so the covariance of the per-shot placement values is
   accounted for). AUC ties receive half credit throughout.

Outputs (in --out):
    tables/appD_trivial_robustness.tex   Table VII (+ console report)
    tables/appC_mechanism.tex            two-bin anatomy
    tables/appC_mechanism_full.tex/.csv  Table VI: full decomposition of the same-decoder
                                         AUC difference over all gap bins, split into
                                         failures shared with the GNN and MWPM-only failures
    derived/robustness_d{d}_dt{r}.json   cache (also holds the paired dAUC and its standard error)

Options: --mwpm {mwpm, mwpmcorr} selects the MWPM decoder (uncorrelated /
correlated); --out the output folder. Trivial shots are always identified
against the uncorrelated array (the GNN sentinel is copied from the
uncorrelated trivial gap).
"""

import argparse
import json
import os
import time

import numpy as np

import runs

FIG_DIR = os.path.join(runs.BUILD_DIR, "mwpmcorr")
DERIVED_DIR = os.path.join(FIG_DIR, "derived")
TABLE_DIR = os.path.join(FIG_DIR, "tables")

P = runs.P_MAIN
SHOTS = runs.SHOTS
CONFIGS = runs.CONFIGS
CROSSED_EXCLUDED_CONFIGS = [(5, 5), (5, 7), (5, 9), (5, 11)]
KAPPA_ANCHORS = [0.5, 0.7, 0.9]

RECOMPUTE = False

MWPM_ARRAY = "mwpm"
MWPM_NAME, MWPM_SCORE = "MWPM", r"g_{\rm MWPM}"
VARIANTS = {"mwpm": ("mwpm", r"MWPM$_{\rm uncorr}$", r"g_{\rm uncorr}"),
            "mwpmcorr": ("mwpmcorr", r"MWPM$_{\rm corr}$", r"g_{\rm corr}")}


def configure(variant=None, out_dir=None):
    global MWPM_ARRAY, MWPM_NAME, MWPM_SCORE, FIG_DIR, DERIVED_DIR, TABLE_DIR
    if variant is not None:
        MWPM_ARRAY, MWPM_NAME, MWPM_SCORE = VARIANTS[variant]
    if out_dir is not None:
        FIG_DIR = out_dir
        DERIVED_DIR = os.path.join(FIG_DIR, "derived")
        TABLE_DIR = os.path.join(FIG_DIR, "tables")


# ---------------------------------------------------------------------------
# Loading and trivial-shot identification
# ---------------------------------------------------------------------------

def load_deltas(d, dt):
    runs.require_arrays(d, dt, P, SHOTS, decoders=[MWPM_ARRAY, "gnn"])
    dm = np.load(runs.array_path(MWPM_ARRAY, d, dt, P, SHOTS))
    dg = np.load(runs.array_path("gnn", d, dt, P, SHOTS))
    return dm, dg


def load_uncorrelated(d, dt):
    runs.require_arrays(d, dt, P, SHOTS, decoders=["mwpm"])
    return np.load(runs.array_path("mwpm", d, dt, P, SHOTS))


def trivial_mask(dm, dg):
    """
    Trivial-syndrome shots are identified as the shots with bit-identical
    signed deltas in both arrays (the GNN sentinel is copied from the MWPM
    trivial-gap computation) that additionally SHARE one |value|: all
    trivial shots carry the same sentinel confidence, whereas accidental
    float coincidences between a matching weight and a network logit
    (a handful are expected in 1e8 trials) occur at scattered unique
    values. Candidates whose |value| appears only once are therefore
    counted as collisions, not as trivial syndromes.
    """
    cand = dm == dg
    idx = np.flatnonzero(cand)
    info = {"n_candidates": int(len(idx))}
    if len(idx) > 0:
        absv = np.abs(dm[idx])
        vals, counts = np.unique(absv, return_counts=True)
        if counts.max() >= 2:
            sentinel = float(vals[np.argmax(counts)])
            triv_local = absv == sentinel
        else:
            sentinel = None
            triv_local = np.zeros(len(idx), dtype=bool)
    else:
        sentinel = None
        triv_local = np.zeros(0, dtype=bool)

    mask = np.zeros(len(dm), dtype=bool)
    if triv_local.any():
        mask[idx[triv_local]] = True
    n_triv = int(mask.sum())

    info.update({
        "n_trivial": n_triv,
        "n_collisions": int(len(idx)) - n_triv,
        "fraction": n_triv / len(dm),
        "confidence_db": sentinel,
        "is_max_mwpm_gap": (bool(np.isclose(sentinel,
                                            float(np.abs(dm).max())))
                            if sentinel is not None else None),
        "n_logical_error": int(np.signbit(dm[mask]).sum()),
        "hard_decision": 0,  # trivial syndromes are predicted as 0
    })
    return mask, info


# ---------------------------------------------------------------------------
# AUC with midranks (ties get half credit) + DeLong placements
# ---------------------------------------------------------------------------

def auc_and_placements(conf, fail):
    """
    Tie-aware AUC = P(conf_success > conf_fail) + 0.5 P(tie), plus the
    DeLong placement arrays:
        phi_f[i] = (fraction of successes strictly above failure i) + ties/2
        phi_s[j] = (fraction of failures strictly below success j) + ties/2
    AUC = mean(phi_f) = mean(phi_s).
    """
    c_fail = np.sort(conf[fail])
    c_all_sorted = np.sort(conf)
    n = len(conf)
    n_f = len(c_fail)
    n_s = n - n_f

    # Placements of successes among failures (one searchsorted pass).
    c_succ = conf[~fail]
    below = np.searchsorted(c_fail, c_succ, side="left")
    above_incl = np.searchsorted(c_fail, c_succ, side="right")
    phi_s = (below + 0.5 * (above_incl - below)) / n_f

    # Placements of failures among successes, via all-minus-failures.
    c_f = conf[fail]
    all_below = np.searchsorted(c_all_sorted, c_f, side="left")
    all_upto = np.searchsorted(c_all_sorted, c_f, side="right")
    f_below = np.searchsorted(c_fail, c_f, side="left")
    f_upto = np.searchsorted(c_fail, c_f, side="right")
    s_below = all_below - f_below
    s_ties = (all_upto - all_below) - (f_upto - f_below)
    s_above = n_s - s_below - s_ties
    phi_f = (s_above + 0.5 * s_ties) / n_s

    auc = float(phi_f.mean())
    return auc, phi_f.astype(np.float64), phi_s.astype(np.float32)


def paired_delta_auc(conf_a, conf_b, fail):
    """
    dAUC = AUC(conf_a -> fail) - AUC(conf_b -> fail) with paired DeLong SE:
        var = var(phi_f_a - phi_f_b)/n_f + var(phi_s_a - phi_s_b)/n_s.
    """
    auc_a, pf_a, ps_a = auc_and_placements(conf_a, fail)
    auc_b, pf_b, ps_b = auc_and_placements(conf_b, fail)
    n_f = len(pf_a)
    n_s = len(ps_a)
    var = (np.var(pf_a - pf_b, ddof=1) / n_f
           + np.var(ps_a.astype(np.float64) - ps_b.astype(np.float64),
                    ddof=1) / n_s)
    return {"auc_a": auc_a, "auc_b": auc_b,
            "delta": auc_a - auc_b, "se": float(np.sqrt(var))}


def midrank_auc(conf, fail):
    """Tie-aware AUC only (no placements) — cheaper for the exclusion table."""
    order = np.argsort(conf, kind="stable")
    sc = conf[order]
    new = np.empty(len(sc), dtype=bool)
    new[0] = True
    np.not_equal(sc[1:], sc[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    counts = np.diff(np.append(starts, len(sc)))
    mid = starts + (counts + 1) / 2.0
    ranks_sorted = mid[np.cumsum(new) - 1]
    f_sorted = fail[order]
    n = len(sc)
    n_f = int(f_sorted.sum())
    n_s = n - n_f
    r_fail = float(ranks_sorted[f_sorted].sum())
    u_fail = r_fail - n_f * (n_f + 1) / 2.0
    return 1.0 - u_fail / (n_f * n_s)


# ---------------------------------------------------------------------------
# Fractional-tie post-selection (same rule as make_fig1/make_fig2)
# ---------------------------------------------------------------------------

def ler_at_kappa(conf, fail, kappa):
    n = len(conf)
    n_acc = int(np.floor(kappa * n))
    v = np.partition(conf, n - n_acc)[n - n_acc]
    above = conf > v
    n_above = int(above.sum())
    f_above = float((fail & above).sum())
    at = conf == v
    m_v = int(at.sum())
    f_v = float((fail & at).sum())
    f_acc = f_above + f_v * (n_acc - n_above) / m_v
    return f_acc / n_acc


# ---------------------------------------------------------------------------
# Per-config analysis
# ---------------------------------------------------------------------------

MECHANISM_CONFIGS = [(5, 5), (9, 9)]
MECH_BINS = [(0.0, 5.0), (30.0, 40.0)]


def mechanism_stats(d, dt):
    """
    Anatomy of the crossed-ranking verdict (numbers quoted in Results, the
    signed-distribution paragraph). For MWPM's failures binned by their gap
    value: the mean exceedance (fraction of correct shots ranked above the
    failure, ties half credit) under each score. Plus the failure-set
    overlap and the gap's overconfident-failure fraction.
    """
    dm, dg = load_deltas(d, dt)
    fail = np.signbit(dm)   # a wrongly decided exact tie is stored as -0.0
    gm, gl = np.abs(dm), np.abs(dg)
    both = int((fail & np.signbit(dg)).sum())
    del dm, dg
    gm_f, gl_f = gm[fail], gl[fail]
    gm_s, gl_s = np.sort(gm[~fail]), np.sort(gl[~fail])
    del gm, gl

    def exceed(sorted_s, vals):
        lo = np.searchsorted(sorted_s, vals, side="left")
        hi = np.searchsorted(sorted_s, vals, side="right")
        return (len(sorted_s) - hi + 0.5 * (hi - lo)) / len(sorted_s)

    n_fail = len(gm_f)
    out = {"n_fail_mwpm": int(n_fail),
           "both_fail": both,
           "gnn_correct_on_mwpm_failures": 1.0 - both / n_fail,
           "overconfident_fraction_gap_gt_15db": float((gm_f > 15).mean()),
           "exceedance_by_gap_bin": {}}
    for lo_db, hi_db in MECH_BINS:
        m = (gm_f >= lo_db) & (gm_f < hi_db)
        out["exceedance_by_gap_bin"][f"{lo_db:g}-{hi_db:g}"] = {
            "n_failures": int(m.sum()),
            "gap": float(exceed(gm_s, gm_f[m]).mean()),
            "logit": float(exceed(gl_s, gl_f[m]).mean()),
        }
    return out


def saturation_stats(d, dt, n_trivial):
    """
    Gap-saturation block: shots whose MWPM gap sits exactly at its maximal
    value (the weighted distance across the code), where the gap cannot
    order the shots any further. Trivial-syndrome shots carry the same
    value (the sentinel), so the non-trivial fraction subtracts them.
    """
    dm, _ = load_deltas(d, dt)
    g = np.abs(dm)
    gmax = float(g.max())
    n_sat = int(np.count_nonzero(g == gmax))
    return {"max_gap_db": gmax,
            "n_saturated": n_sat,
            "fraction": n_sat / len(g),
            "fraction_nontrivial": (n_sat - n_trivial) / len(g)}


def mechanism_full(d, dt):
    """
    Exact decomposition of the same-decoder AUC difference for MWPM's failures,
        dAUC = AUC(g_MWPM -> MWPM) - AUC(g_GNN -> MWPM)
             = sum over gap bins b of (1/n_f) sum_{i in b} [exceed_gap(i) - exceed_logit(i)],
    where exceed(i) is the fraction of correctly decoded shots ranked above
    failure i (ties half credit). Each bin is split into failures shared with
    the GNN and MWPM-only failures (on which the GNN logit is the GNN's
    confidence in the OPPOSITE label).
    """
    dm, dg = load_deltas(d, dt)
    fail = np.signbit(dm)   # a wrongly decided exact tie is stored as -0.0
    shared = np.signbit(dg)[fail]
    gm, gl = np.abs(dm), np.abs(dg)
    del dm, dg
    gm_f, gl_f = gm[fail], gl[fail]
    gm_s, gl_s = np.sort(gm[~fail]), np.sort(gl[~fail])
    del gm, gl

    def exceed(sorted_s, vals):
        lo = np.searchsorted(sorted_s, vals, side="left")
        hi = np.searchsorted(sorted_s, vals, side="right")
        return (len(sorted_s) - hi + 0.5 * (hi - lo)) / len(sorted_s)

    e_gap, e_log = exceed(gm_s, gm_f), exceed(gl_s, gl_f)
    n_f = len(gm_f)
    contrib = (e_gap - e_log) / n_f
    edges = [0, 5, 10, 15, 20, 30, 40, 60, np.inf]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (gm_f >= lo) & (gm_f < hi)
        if not m.any():
            continue
        bins.append({"lo": lo, "hi": None if np.isinf(hi) else hi, "n_fail": int(m.sum()),
                     "frac": float(m.mean()),
                     "outranked_gap": float(1 - e_gap[m].mean()), "outranked_logit": float(1 - e_log[m].mean()),
                     "contrib": float(contrib[m].sum()), "contrib_shared": float(contrib[m & shared].sum()),
                     "contrib_mwpm_only": float(contrib[m & ~shared].sum())})
    return {"n_fail": n_f, "shared_fraction": float(shared.mean()), "dauc": float(contrib.sum()),
            "dauc_shared": float(contrib[shared].sum()), "dauc_mwpm_only": float(contrib[~shared].sum()),
            "median_logit_shared": float(np.median(gl_f[shared])),
            "median_logit_mwpm_only": float(np.median(gl_f[~shared])),
            "bins": bins}


def analyze_config(d, dt):
    t0 = time.perf_counter()
    dm, dg = load_deltas(d, dt)
    du = dm if MWPM_ARRAY == "delta_mwpm" else load_uncorrelated(d, dt)
    mask, triv = trivial_mask(du, dg)
    del du
    triv["n_logical_error"] = int(np.signbit(dm[mask]).sum())   # errors of the decoder analysed

    fail = {"mwpm": np.signbit(dm), "gnn": np.signbit(dg)}   # a wrongly decided exact tie is stored as -0.0
    conf = {"g_mwpm": np.abs(dm), "g_gnn": np.abs(dg)}
    del dm, dg

    out = {"config": {"d": d, "dt": dt, "p": P, "shots": SHOTS},
           "trivial": triv, "auc": {}, "auc_excl_trivial": {},
           "delta_auc": {}, "crossed_excl": {}}

    # Paired dAUC (all shots), per decoder: own score minus other score.
    for dec, own, other in [("mwpm", "g_mwpm", "g_gnn"),
                            ("gnn", "g_gnn", "g_mwpm")]:
        res = paired_delta_auc(conf[own], conf[other], fail[dec])
        out["delta_auc"][dec] = res
        out["auc"][f"{own}->{dec}"] = res["auc_a"]
        out["auc"][f"{other}->{dec}"] = res["auc_b"]

    # AUC with trivial shots excluded.
    if triv["n_trivial"] > 0:
        keep = ~mask
        for s_name in conf:
            for dec in fail:
                out["auc_excl_trivial"][f"{s_name}->{dec}"] = midrank_auc(
                    conf[s_name][keep], fail[dec][keep])
    else:
        out["auc_excl_trivial"] = dict(out["auc"])

    # Crossed anchor-kappa LERs with trivial shots excluded (d=5 family).
    if (d, dt) in CROSSED_EXCLUDED_CONFIGS and triv["n_trivial"] > 0:
        keep = ~mask
        for s_name in conf:
            for dec in fail:
                key = f"decoder={dec},rank={s_name}"
                out["crossed_excl"][key] = {
                    f"{k:g}": ler_at_kappa(conf[s_name][keep],
                                           fail[dec][keep], k)
                    for k in KAPPA_ANCHORS
                }

    print(f"  d={d}, dt={dt}: {time.perf_counter() - t0:.0f} s | "
          f"trivial: {triv['n_trivial']:,} ({triv['fraction']:.2%}), "
          f"{triv['n_logical_error']} with logical error")
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_table(results):
    os.makedirs(TABLE_DIR, exist_ok=True)
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\begin{tabular}{l|cc|cc|cc}",
        r"\toprule",
        r"Config & trivial fraction & conf.\ (dB) & "
        rf"$\Delta$AUC$_{{\rm {MWPM_NAME.split('$')[0]}}}$ & $\Delta$AUC$_{{\rm GNN}}$ & "
        rf"AUC$^{{\rm excl}}_{{g_{{\rm GNN}}\to{{\rm MWPM}}}}$ & "
        rf"AUC$^{{\rm excl}}_{{{MWPM_SCORE}\to{{\rm MWPM}}}}$ \\",
        r"\hline",
        r"\midrule",
    ]
    def pct(f):
        # Trivial fraction as a percentage; LaTeX scientific for tiny values.
        if f == 0:
            return "$0$"
        v = 100.0 * f
        if v >= 1e-4:
            return rf"${v:.2g}\%$"
        exp = int(np.floor(np.log10(v)))
        return rf"${v / 10**exp:.1f} \times 10^{{{exp}}}\%$"

    for (d, dt), res in results.items():
        t = res["trivial"]
        da_m = res["delta_auc"]["mwpm"]
        da_g = res["delta_auc"]["gnn"]
        conf_db = (f"{t['confidence_db']:.2f}"
                   if t["confidence_db"] is not None else "--")
        lines.append(
            rf"$({d}, {dt})$ & {pct(t['fraction'])} & {conf_db} & "
            rf"${da_m['delta']:+.4f}$ & ${da_g['delta']:+.4f}$ & "
            rf"{res['auc_excl_trivial']['g_gnn->mwpm']:.4f} & "
            rf"{res['auc_excl_trivial']['g_mwpm->mwpm']:.4f} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Robustness of the discrimination results at "
        rf"$p = {P}$. Trivial-syndrome shots (bypassing the GNN; assigned "
        r"hard decision $0$ and, as confidence, the maximal MWPM gap) are "
        r"identified exactly and characterized. ${\rm AUC}^{\rm excl}$ "
        r"denotes the tie-aware AUC recomputed with the trivial shots "
        r"excluded (both scores ranking MWPM's failures). "
        r"$\Delta{\rm AUC}_D$ is the own-score minus other-score AUC for "
        r"decoder $D$ (all shots); the paired DeLong standard errors are "
        r"below $10^{-4}$ throughout and are omitted. AUC ties receive half "
        r"credit throughout.}",
        r"\label{robustness_table}",
        r"\end{table*}",
    ]
    path = os.path.join(TABLE_DIR, "appD_trivial_robustness.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {path}")


def write_mechanism_table(results):
    """
    Appendix C table: anatomy of the crossed ranking of MWPM's failures.
    For failures binned by their gap value, the mean fraction of correctly
    decoded shots ranked BELOW the failure (ties half credit) under each
    score — the plain-language content of the exceedance analysis.
    """
    os.makedirs(TABLE_DIR, exist_ok=True)
    lines = [
        r"\begin{table}",
        r"\centering",
        r"\begin{tabular}{ll|cc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{correct shots outranked} \\",
        rf"Config & {MWPM_NAME} failures with gap in & by ${MWPM_SCORE}$ & "
        r"by $g_{\rm GNN}$ \\",
        r"\hline",
        r"\midrule",
    ]
    for d, dt in MECHANISM_CONFIGS:
        m = results[(d, dt)]["mechanism"]
        for rng, e in m["exceedance_by_gap_bin"].items():
            lo_db, hi_db = rng.split("-")
            lines.append(
                rf"$({d}, {dt})$ & ${lo_db}$--${hi_db}$ dB "
                rf"({e['n_failures']:,} failures) & "
                rf"{1 - e['gap']:.1%} & {1 - e['logit']:.1%} \\"
                .replace("%", r"\%"))
        if (d, dt) != MECHANISM_CONFIGS[-1]:
            lines.append(r"\hline")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Anatomy of the crossed ranking of MWPM's failures at "
        rf"$p = {P}$: for MWPM failures grouped by their own gap value, the "
        r"mean fraction of correctly decoded shots that the failure outranks "
        r"(i.e., that receive a lower confidence than the failure; ties half "
        r"credit) under each score (two example bins; the full decomposition "
        r"is in Table~\ref{mechanism_full_table}).}",
        r"\label{mechanism_table}",
        r"\end{table}",
    ]
    path = os.path.join(TABLE_DIR, "appC_mechanism.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {path}")


FULL_MECHANISM_CONFIGS = [(5, 5), (7, 7), (9, 9)]


def write_mechanism_full_table(results):
    """Full bin decomposition of dAUC = AUC(own gap) - AUC(GNN logit) on MWPM's failures."""
    os.makedirs(TABLE_DIR, exist_ok=True)
    cfgs = [c for c in FULL_MECHANISM_CONFIGS if c in results and "mechanism_full" in results[c]]
    with open(os.path.join(TABLE_DIR, "appC_mechanism_full.csv"), "w") as f:
        f.write("d,r,bin_lo_db,bin_hi_db,n_fail,frac,outranked_by_gap,outranked_by_logit,"
                "contrib,contrib_shared,contrib_mwpm_only\n")
        for d, dt in cfgs:
            for b in results[(d, dt)]["mechanism_full"]["bins"]:
                f.write(f"{d},{dt},{b['lo']},{b['hi']},{b['n_fail']},{b['frac']:.5f},{b['outranked_gap']:.5f},"
                        f"{b['outranked_logit']:.5f},{b['contrib']:.6f},{b['contrib_shared']:.6f},"
                        f"{b['contrib_mwpm_only']:.6f}\n")
    lines = [r"\begin{table*}", r"\centering", r"\begin{tabular}{ll|rr|cc|rrr}", r"\toprule",
             r" & & & & \multicolumn{2}{c|}{correct shots outranked} & \multicolumn{3}{c}{contribution to "
             r"$\Delta$AUC ($\times 10^{3}$)} \\",
             rf"Config & ${MWPM_SCORE}$ (dB) & failures & fraction & by ${MWPM_SCORE}$ & by $g_{{\rm GNN}}$ & "
             r"all & shared & only \\", r"\hline", r"\midrule"]
    for i, (d, dt) in enumerate(cfgs):
        mf = results[(d, dt)]["mechanism_full"]
        for b in mf["bins"]:
            rng = f"{b['lo']:g}--{b['hi']:g}" if b["hi"] is not None else rf"$\geq {b['lo']:g}$"
            lines.append(rf"$({d}, {dt})$ & {rng} & {b['n_fail']:,} & {100 * b['frac']:.1f}\% & "
                         rf"{100 * b['outranked_gap']:.1f}\% & {100 * b['outranked_logit']:.1f}\% & "
                         rf"${1e3 * b['contrib']:+.2f}$ & ${1e3 * b['contrib_shared']:+.2f}$ & "
                         rf"${1e3 * b['contrib_mwpm_only']:+.2f}$ \\")
        lines.append(rf"$({d}, {dt})$ & total & {mf['n_fail']:,} & 100\% & & & "
                     rf"${1e3 * mf['dauc']:+.2f}$ & ${1e3 * mf['dauc_shared']:+.2f}$ & "
                     rf"${1e3 * mf['dauc_mwpm_only']:+.2f}$ \\")
        if i < len(cfgs) - 1:
            lines.append(r"\hline")
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{Exact decomposition of the same-decoder AUC difference "
              rf"$\Delta{{\rm AUC}} = {{\rm AUC}}({MWPM_SCORE}) - {{\rm AUC}}(g_{{\rm GNN}})$ for the failures of "
              rf"{MWPM_NAME} at $p = {P}$, over bins of the failure's own gap (positive contributions favour "
              r"the gap). Each contribution is split into failures shared with the GNN and failures of MWPM "
              r"only, on which the GNN logit expresses the GNN's confidence in the opposite (correct) label. "
              r"Correct shots outranked: mean fraction of correctly decoded shots receiving a lower confidence "
              r"than the failure (ties half credit).}", r"\label{mechanism_full_table}", r"\end{table*}"]
    path = os.path.join(TABLE_DIR, "appC_mechanism_full.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tables VI and VII: anatomy and robustness")
    ap.add_argument("--mwpm", choices=sorted(VARIANTS), default="mwpmcorr")
    ap.add_argument("--out", default=None, help="output folder (default build/<mwpm>/)")
    args = ap.parse_args()
    configure(args.mwpm, args.out or os.path.join(runs.BUILD_DIR, args.mwpm))
    os.makedirs(DERIVED_DIR, exist_ok=True)
    start = time.perf_counter()

    results = {}
    for d, dt in CONFIGS:
        cache = os.path.join(DERIVED_DIR, f"robustness_d{d}_dt{dt}.json")
        if not RECOMPUTE and os.path.isfile(cache):
            with open(cache) as f:
                results[(d, dt)] = json.load(f)
            print(f"  d={d}, dt={dt}: loaded cache")
            continue
        res = analyze_config(d, dt)
        res["saturation"] = saturation_stats(d, dt, res["trivial"]["n_trivial"])
        if (d, dt) in MECHANISM_CONFIGS:      # the two configs quoted in the signed-distribution paragraph
            res["mechanism"] = mechanism_stats(d, dt)
        if (d, dt) in FULL_MECHANISM_CONFIGS:  # full bin decomposition of the AUC difference
            res["mechanism_full"] = mechanism_full(d, dt)
        results[(d, dt)] = res
        with open(cache, "w") as f:
            json.dump(res, f)

    write_table(results)
    if all(c in results for c in MECHANISM_CONFIGS):
        write_mechanism_table(results)
    write_mechanism_full_table(results)

    print("\nMechanism anatomy (MWPM failures; exceedance = fraction of "
          "correct shots ranked above the failure):")
    for d, dt in [c for c in MECHANISM_CONFIGS if c in results]:
        m = results[(d, dt)]["mechanism"]
        print(f"  (d={d}, r={dt}):  GNN decodes "
              f"{m['gnn_correct_on_mwpm_failures']:.0%} of MWPM's failures "
              f"correctly; {m['overconfident_fraction_gap_gt_15db']:.1%} of "
              f"MWPM failures carry gap > 15 dB")
        for rng, e in m["exceedance_by_gap_bin"].items():
            print(f"      gap {rng} dB ({e['n_failures']:,} failures): "
                  f"exceedance gap {e['gap']:.4f} vs logit {e['logit']:.4f}")

    print("\nGap saturation (shots at the maximal MWPM gap; the gap cannot "
          "rank inside this block):")
    for (d, dt), res in results.items():
        s = res["saturation"]
        print(f"  (d={d}, r={dt}):  max gap {s['max_gap_db']:.2f} dB, "
              f"saturated {s['n_saturated']:,} shots = {s['fraction']:.4%} "
              f"({s['fraction_nontrivial']:.4%} excluding trivial)")

    print("\nPaired dAUC (own - other), positive = own score ranks its "
          "decoder's failures better:")
    for (d, dt), res in results.items():
        m = res["delta_auc"]["mwpm"]
        g = res["delta_auc"]["gnn"]
        print(f"  (d={d}, r={dt}):  MWPM: {m['delta']:+.4f} +- {m['se']:.4f}"
              f"   GNN: {g['delta']:+.4f} +- {g['se']:.4f}")

    print("\nCross-decoder AUC at d=5 family, trivial shots excluded "
          "(does the small-d result survive?):")
    for (d, dt) in [c for c in CROSSED_EXCLUDED_CONFIGS if c in results]:
        res = results[(d, dt)]
        a = res["auc_excl_trivial"]
        print(f"  (d={d}, r={dt}): gG->M={a['g_gnn->mwpm']:.4f} vs "
              f"gM->M={a['g_mwpm->mwpm']:.4f}  |  "
              f"gG->G={a['g_gnn->gnn']:.4f} vs gM->G={a['g_mwpm->gnn']:.4f}")

    print(f"\nTotal elapsed: {(time.perf_counter() - start) / 60:.1f} min")
