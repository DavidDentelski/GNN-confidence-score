"""
Appendix: robustness of the discrimination results.

Robustness checks, computed from the saved shot-aligned arrays:

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

Outputs:
    figures3/tables/appD_trivial_robustness.tex  (+ console report)
    figures3/derived/robustness_d{d}_dt{dt}.json
"""

import json
import os
import time

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_runs")

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures3")
DERIVED_DIR = os.path.join(FIG_DIR, "derived")
TABLE_DIR = os.path.join(FIG_DIR, "tables")

P = 0.005
SHOTS = 100_000_000
CONFIGS = [(5, 5), (5, 7), (5, 9), (5, 11), (7, 7), (7, 9), (7, 11), (9, 9)]
CROSSED_EXCLUDED_CONFIGS = [(5, 5), (5, 7), (5, 9), (5, 11)]
KAPPA_ANCHORS = [0.5, 0.7, 0.9]

RECOMPUTE = False


# ---------------------------------------------------------------------------
# Loading and trivial-shot identification
# ---------------------------------------------------------------------------

def load_deltas(d, dt):
    tag = f"d{d}_dt{dt}_p{P}_shots{SHOTS}"
    dm = np.load(os.path.join(SAVE_DIR, f"delta_mwpm_{tag}.npy"))
    dg = np.load(os.path.join(SAVE_DIR, f"delta_gnn_{tag}.npy"))
    return dm, dg


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
        "n_logical_error": int((dm[mask] < 0).sum()),
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
    Anatomy of the crossed-ranking verdict (numbers quoted in the Results
    of the paper). For MWPM's failures binned by their gap
    value: the mean exceedance (fraction of correct shots ranked above the
    failure, ties half credit) under each score. Plus the failure-set
    overlap and the gap's overconfident-failure fraction.
    """
    dm, dg = load_deltas(d, dt)
    fail = dm < 0
    gm, gl = np.abs(dm), np.abs(dg)
    both = int((fail & (dg < 0)).sum())
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


def analyze_config(d, dt):
    t0 = time.perf_counter()
    dm, dg = load_deltas(d, dt)
    mask, triv = trivial_mask(dm, dg)

    fail = {"mwpm": dm < 0, "gnn": dg < 0}
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
        r"$\Delta$AUC$_{\rm MWPM}$ & $\Delta$AUC$_{\rm GNN}$ & "
        r"AUC$^{\rm excl}_{g_{\rm GNN}\to{\rm MWPM}}$ & "
        r"AUC$^{\rm excl}_{g_{\rm MWPM}\to{\rm MWPM}}$ \\",
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
        r"identified exactly and characterized; no trivial shot in any "
        r"configuration carries a logical error. ${\rm AUC}^{\rm excl}$ "
        r"denotes the tie-aware AUC recomputed with the trivial shots "
        r"excluded; the two columns show the cross-decoder comparison of "
        r"the $d = 5$ reversal (both scores ranking MWPM's failures). "
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
        r"Config & MWPM failures with gap in & by $g_{\rm MWPM}$ & "
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
        r"credit) under each score. The gap is the sharper flag for its "
        r"near-tied failures; the logit demotes the overconfident matching "
        r"failures far more effectively at both distances. The balance of "
        r"the two populations sets the sign of the same-decoder AUC "
        r"difference in Table~\ref{auc_table}.}",
        r"\label{mechanism_table}",
        r"\end{table}",
    ]
    path = os.path.join(TABLE_DIR, "appC_mechanism.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {path}")


if __name__ == "__main__":
    os.makedirs(DERIVED_DIR, exist_ok=True)
    start = time.perf_counter()

    results = {}
    for d, dt in CONFIGS:
        cache = os.path.join(DERIVED_DIR, f"robustness_v2_d{d}_dt{dt}.json")
        if not RECOMPUTE and os.path.isfile(cache):
            with open(cache) as f:
                results[(d, dt)] = json.load(f)
            print(f"  d={d}, dt={dt}: loaded cache")
        else:
            results[(d, dt)] = analyze_config(d, dt)
            with open(cache, "w") as f:
                json.dump(results[(d, dt)], f)

        # Saturation stats augment existing caches in place (cheap: only
        # the MWPM array is loaded) instead of forcing a full recompute.
        if "saturation" not in results[(d, dt)]:
            results[(d, dt)]["saturation"] = saturation_stats(
                d, dt, results[(d, dt)]["trivial"]["n_trivial"])
            with open(cache, "w") as f:
                json.dump(results[(d, dt)], f)

        # Mechanism anatomy, for the two configurations quoted in the
        # Results of the paper.
        if (d, dt) in MECHANISM_CONFIGS and "mechanism" not in results[(d, dt)]:
            results[(d, dt)]["mechanism"] = mechanism_stats(d, dt)
            with open(cache, "w") as f:
                json.dump(results[(d, dt)], f)

    write_table(results)
    write_mechanism_table(results)

    print("\nMechanism anatomy (MWPM failures; exceedance = fraction of "
          "correct shots ranked above the failure):")
    for d, dt in MECHANISM_CONFIGS:
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
    for (d, dt) in CROSSED_EXCLUDED_CONFIGS:
        res = results[(d, dt)]
        a = res["auc_excl_trivial"]
        print(f"  (d={d}, r={dt}): gG->M={a['g_gnn->mwpm']:.4f} vs "
              f"gM->M={a['g_mwpm->mwpm']:.4f}  |  "
              f"gG->G={a['g_gnn->gnn']:.4f} vs gM->G={a['g_mwpm->gnn']:.4f}")

    print(f"\nTotal elapsed: {(time.perf_counter() - start) / 60:.1f} min")
