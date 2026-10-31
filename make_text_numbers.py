"""
Every number quoted in the text that is not printed in a table or read off a
figure, collected by section of the paper in text_numbers.md.

Computed here from the per-shot arrays (runs.py) and the run provenance JSONs,
and cached in text_numbers.json:
  - decision mismatches of the shared-graph gap vs PyMatching's correlated decoder (Sec. II, App. B);

  at p = 0.005, d = r = 5, 7, 9 (1e8 shots, the same shots for all decoders):
    - fraction of shots on which the decisions of MWPM_corr and the GNN differ (Sec. III B),
    - fraction of each matching decoder's failures that the GNN shares (Sec. III B, App. D),
    - tails of the joint score distribution (App. D): MWPM_corr failures with a large logit and
      GNN failures with a large correlated gap, and how often the other decoder is correct there,
    - the weighted-distance bound of each gap (= gap of the trivial syndrome) and the fraction of
      shots sitting at it (App. D);
  over the p-scan (d = r = 5, 7, 9):
    - hard-decision LERs, their slopes d ln L / d ln p (weighted least squares, Poisson errors), and
      the ratios L_corr/L_uncorr and L_corr/L_GNN (Sec. III A).

Read from the cached results of the other steps of make_all.py (build/): the Eq. (4)
factors and pair ratios at kappa = 0.9 (Sec. III A), the decision disagreement of every
configuration and the AUC recovered by the signed exchange (Sec. III B, App. D), the
size of the d = 3 model and the KL reduction (Sec. III C, App. F), the log-loss and
calibration-slope statements (Sec. III D, App. C), the comparison of correlated-gap
constructions (App. B), the rank correlation and the GNN's high-confidence failures
(App. D), the local slopes and alpha vs p (App. E).

Outputs (in --out): text_numbers.json (cache of the array-based part, plus all values) and text_numbers.md.
"""

import argparse
import json
import os

import numpy as np

import runs

NAMES = runs.DECODERS
P_MAIN, SHOTS_MAIN = runs.P_MAIN, runs.SHOTS
TAIL_DB = (20, 30)
BLOCK = 10_000_000


def path(name, d, r, p, shots):
    return runs.array_path(name, d, r, p, shots)


def shot_level(d):
    """Overlaps, disagreements, tails and gap bounds at p = 0.005 (arrays read in blocks)."""
    runs.require_arrays(d, d, P_MAIN, SHOTS_MAIN)
    a = {n: np.load(path(n, d, d, P_MAIN, SHOTS_MAIN), mmap_mode="r") for n in NAMES}
    bound = {n: max(float(np.max(np.abs(a[n][i:i + BLOCK]))) for i in range(0, SHOTS_MAIN, BLOCK))
             for n in ("mwpm", "mwpmcorr")}
    c = {k: 0 for k in ("disagree", "fail_mwpm", "fail_mwpmcorr", "fail_gnn", "shared_mwpm", "shared_mwpmcorr",
                        "at_bound_mwpm", "at_bound_mwpmcorr")}
    for t in TAIL_DB:
        c.update({f"corr_fail_logit_gt{t}": 0, f"corr_fail_logit_gt{t}_gnn_correct": 0,
                  f"gnn_fail_gap_gt{t}": 0, f"gnn_fail_gap_gt{t}_corr_correct": 0})
    for i in range(0, SHOTS_MAIN, BLOCK):
        x = {n: np.asarray(a[n][i:i + BLOCK]) for n in NAMES}
        f = {n: np.signbit(x[n]) for n in NAMES}   # a wrongly decided exact tie is stored as -0.0
        c["disagree"] += int(np.count_nonzero(f["mwpmcorr"] ^ f["gnn"]))
        for n in NAMES:
            c[f"fail_{n}"] += int(np.count_nonzero(f[n]))
        for n in ("mwpm", "mwpmcorr"):
            c[f"shared_{n}"] += int(np.count_nonzero(f[n] & f["gnn"]))
            c[f"at_bound_{n}"] += int(np.count_nonzero(np.abs(np.abs(x[n]) - bound[n]) < 1e-3))
        for t in TAIL_DB:
            m = f["mwpmcorr"] & (np.abs(x["gnn"]) > t)
            c[f"corr_fail_logit_gt{t}"] += int(np.count_nonzero(m))
            c[f"corr_fail_logit_gt{t}_gnn_correct"] += int(np.count_nonzero(m & ~f["gnn"]))
            m = f["gnn"] & (np.abs(x["mwpmcorr"]) > t)
            c[f"gnn_fail_gap_gt{t}"] += int(np.count_nonzero(m))
            c[f"gnn_fail_gap_gt{t}_corr_correct"] += int(np.count_nonzero(m & ~f["mwpmcorr"]))
    out = {"d": d, "shots": SHOTS_MAIN, "counts": c, "bound_db": bound,
           "disagree_frac": c["disagree"] / SHOTS_MAIN,
           "shared_frac_of_mwpm_failures": {n: c[f"shared_{n}"] / c[f"fail_{n}"] for n in ("mwpm", "mwpmcorr")},
           "at_bound_frac": {n: c[f"at_bound_{n}"] / SHOTS_MAIN for n in ("mwpm", "mwpmcorr")}}
    for t in TAIL_DB:
        out[f"tail_{t}dB"] = {
            "corr_failures_with_large_logit_frac": c[f"corr_fail_logit_gt{t}"] / c["fail_mwpmcorr"],
            "gnn_correct_among_them": c[f"corr_fail_logit_gt{t}_gnn_correct"] / max(c[f"corr_fail_logit_gt{t}"], 1),
            "gnn_failures_with_large_gap_frac": c[f"gnn_fail_gap_gt{t}"] / c["fail_gnn"],
            "corr_correct_among_them": c[f"gnn_fail_gap_gt{t}_corr_correct"] / max(c[f"gnn_fail_gap_gt{t}"], 1)}
    return out


def hard_ler(d, p, shots):
    runs.require_arrays(d, d, p, shots)
    out = {}
    for n in NAMES:
        a = np.load(path(n, d, d, p, shots), mmap_mode="r")
        nf = sum(int(np.count_nonzero(np.signbit(a[i:i + BLOCK]))) for i in range(0, shots, BLOCK))
        out[n] = {"n_fail": nf, "ler": nf / shots}
    return out


def slope(ps, ls, ns):
    """Weighted least-squares slope of ln L vs ln p, with Poisson errors 1/sqrt(n_fail)."""
    x, y, w = np.log(ps), np.log(ls), np.asarray(ns, dtype=float)
    X = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    beta = cov @ (X.T @ (w * y))
    return float(beta[1]), float(np.sqrt(cov[1, 1]))


def run_checks():
    """Decisions of the shared-graph correlated gap vs PyMatching's correlated decoder, on all shots of each
    configuration at p = 0.005 (recorded by generate_shot_data.py in the run provenance JSONs)."""
    out = {}
    for d, r in runs.CONFIGS:
        path = os.path.join(runs.DATA_DIR, f"run_{runs.tag(d, r)}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"no cached result and {path} missing; download the data (README.md)")
        c = json.load(open(path))["arrays"]["mwpmcorr"]["consistency_with_pymatching_correlated"]
        out[f"{d},{r}"] = {"shots": c["shots"], "mismatch": c["decision_vs_pymatching_mismatch"],
                           "mismatch_at_exact_ties": c["mismatch_exact_ties"]}
    return out


def cached_result(*parts):
    return json.load(open(os.path.join(runs.BUILD_DIR, *parts)))


def read_csv_rows(path):
    with open(path) as f:
        head = f.readline().strip().split(",")
        return head, [line.strip().split(",") for line in f if line.strip() and not line.startswith("#")]


def derived_numbers():
    """Numbers computed from the cached results of the other steps of make_all.py (build/)."""
    import make_calibration as mf4
    num = {}
    key = lambda c: f"{c[0]},{c[1]}"

    # Sec. III A: Eq. (4) at kappa = 0.9, and MWPM_uncorr vs MWPM_corr
    num["eq4_kappa_0.9"], num["L_uncorr_over_L_corr_kappa_0.9"] = {}, {}
    for d in runs.PSCAN_DISTANCES:
        res = cached_result("fig1", "derived", f"fig1a_d{d}.json")
        kappa = np.array(res["kappa"])
        i09, i10 = int(np.argmin(np.abs(kappa - 0.9))), int(np.argmin(np.abs(kappa - 1.0)))
        ler = {n: np.array(res["pairs"][n]["ler"]) for n in runs.DECODERS}
        for n in ("mwpmcorr", "mwpm"):
            pair, hard = ler[n][i09] / ler["gnn"][i09], ler[n][i10] / ler["gnn"][i10]
            num["eq4_kappa_0.9"].setdefault(n, {})[str(d)] = {"pair_ratio": pair, "hard_factor": hard,
                                                              "selection_factor": pair / hard}
        num["L_uncorr_over_L_corr_kappa_0.9"][str(d)] = ler["mwpm"][i09] / ler["mwpmcorr"][i09]

    # Sec. III B and App. D: decisions and AUCs of MWPM_corr and the GNN
    num["decisions_differ"], num["sign_recovery"] = {}, {}
    for c in runs.CONFIGS:
        name = f"d{c[0]}_dt{c[1]}_p{runs.P_MAIN}.json"
        da = cached_result("mwpmcorr", "derived", "decision_aware_" + name)
        auc = cached_result("mwpmcorr", "derived", "crossed_" + name)["auc"]
        num["decisions_differ"][key(c)] = 1 - da["agreement"]
        # share of the exchange penalty AUC(own) - AUC(|other|) recovered by signing the other score
        num["sign_recovery"][key(c)] = {
            dec: (da["auc_signed_other"][dec] - auc[f"{oth}->{dec}"]) / (auc[f"{own}->{dec}"] - auc[f"{oth}->{dec}"])
            for dec, own, oth in (("mwpm", "g_mwpm", "g_gnn"), ("gnn", "g_gnn", "g_mwpm"))}

    # Sec. III C and App. F: exact posterior at (3, 3)
    ex = cached_result("exact_d3", "derived", "exact_d3.json")
    num["exact_d3"] = {"shots": ex["shots"], "nontrivial": ex["nontrivial"], **ex["dem"],
                       "kl_reduction_by_correlations": 1 - ex["scores"]["mwpmcorr"]["kl"] / ex["scores"]["mwpm"]["kl"]}

    # Sec. III D and App. C: calibration fits (Table V) and log-loss
    mf4.configure("mwpmcorr", os.path.join(runs.BUILD_DIR, "mwpmcorr"))
    gnn_alpha = {key(c): mf4.full_fit(mf4.get_binned(*c), "GNN")["alpha"] for c in runs.CONFIGS}
    num["gnn_alpha_below_1"] = sum(a < 1 for a in gnn_alpha.values())
    ll = cached_result("tables", "derived", "logloss.json")
    num["logloss_gnn_minus_corr_mbit"] = {key(c): ll[f"gnn|{key(c)}"][0] - ll[f"mwpmcorr|{key(c)}"][0]
                                          for c in runs.CONFIGS}
    num["logloss_rescaling_gain_max_mbit"] = {n: max(ll[f"{n}|{key(c)}"][0] - ll[f"{n}|{key(c)}"][3]
                                                     for c in runs.CONFIGS) for n in runs.DECODERS}
    drift = {v: cached_result(v, "derived", "alpha_drift.json") for v in ("mwpmcorr", "mwpm")}
    num["alpha_sigma_max"] = max(max(f["sigma_fisher"], f["sigma_boot"]) for v in drift.values()
                                 for fits in v["fits"].values() for f in fits.values())
    t = drift["mwpmcorr"]["trends"]["MWPM"]
    num["corr_alpha_trend"] = {k: t[k] for k in ("b_d", "se_b_d", "b_r", "se_b_r")}
    steps = drift["mwpmcorr"]["trends"]["GNN"]["steps"]
    num["gnn_alpha_neighbours"] = {"max_abs_delta": max(abs(s["delta"]) for s in steps),
                                   "min_abs_sigmas": min(abs(s["sigmas"]) for s in steps),
                                   "max_abs_sigmas": max(abs(s["sigmas"]) for s in steps)}
    g2 = drift["mwpmcorr"]["fits"]["GNN"].values()
    num["gnn_two_parameter"] = {"chi2nu_min": min(f["chi2nu_2p"] for f in g2), "chi2nu_max": max(f["chi2nu_2p"] for f in g2),
                                "offset_db_min": min(f["offset_db"] for f in g2),
                                "offset_db_max": max(f["offset_db"] for f in g2)}

    # App. D: rank correlation of the two scores at (9, 9)
    num["spearman_rho_d9"] = cached_result("mwpmcorr", "derived", "pairing_d9_dt9.json")["spearman_rho"]

    # App. D: the GNN's high-confidence failures behind its deep-cut floor at (9, 9)
    hc = cached_result("high_confidence", "derived", "high_confidence_arrays.json")
    sy = cached_result("high_confidence", "derived", "high_confidence_syndromes.json")
    c9, deep = hc["configs"]["9,9"], hc["deep_failures_9_9"]
    top = [x["g_gnn"] for x in deep if x["g_gnn"] >= c9["gnn"]["0.05"]["threshold_db"]]
    med = lambda g, k: sy["comparison"][g][k]["median"]
    num["high_confidence_d9"] = {
        "gnn_failures_kappa_0.4": len(deep), "gnn_failures_kappa_0.2": c9["gnn"]["0.2"]["failures"],
        "predicted_kappa_0.2": c9["gnn"]["0.2"]["expected"], "in_top_5pct": len(top),
        "top_5pct_logit_db": [min(top), max(top)],
        "corr_fails": sum(x["corr_fails"] for x in deep),
        "both_matching_fail": sum(x["corr_fails"] and x["uncorr_fails"] for x in deep),
        "g_corr_median_db": float(np.median([x["g_corr"] for x in deep])),
        "corr_threshold_kappa_0.4_db": c9["mwpmcorr"]["0.4"]["threshold_db"],
        "detection_events_median": {g: med(g, "n_events") for g in sy["comparison"]},
        "min_matching_weight_median_nats": {g: med(g, "min_matching_weight_nats") for g in sy["comparison"]}}
    num["gnn_deep_cut"] = {c: {"observed_kappa_0.1": r["gnn"]["0.1"]["failures"],
                               "predicted_kappa_0.1": r["gnn"]["0.1"]["expected"],
                               "L0.2_over_L0.4": r["gnn"]["0.2"]["ler"] / r["gnn"]["0.4"]["ler"]}
                           for c, r in hc["configs"].items()}

    # App. E: local slopes (Fig. 8) and alpha vs p (Fig. 9)
    tables = os.path.join(runs.BUILD_DIR, "mwpmcorr", "tables")
    head, rows = read_csv_rows(os.path.join(tables, "appD_local_slopes.csv"))
    _, srows = read_csv_rows(os.path.join(tables, "appD_local_slopes_sigma.csv"))
    num["local_slopes"] = {f"{r[0]} {r[1]},{r[2]}": dict(zip(head[3:], map(float, r[3:]))) for r in rows}
    num["local_slopes_sigma_max"] = max(float(v) for r in srows for v in r[3:])
    num["alpha_vs_p"] = {}
    for variant, name, label in (("mwpmcorr", "MWPM", "mwpmcorr"), ("mwpm", "MWPM", "mwpm"), ("mwpmcorr", "GNN", "gnn")):
        mf4.configure(variant, os.path.join(runs.BUILD_DIR, variant))
        for d in runs.PSCAN_DISTANCES:
            alphas = {str(p): mf4.full_fit(mf4.get_binned(d, d, p, shots), name)["alpha"]
                      for _, _, p, shots in runs.pscan_runs(d)}
            num["alpha_vs_p"].setdefault(label, {})[str(d)] = alphas

    # App. B: shared-graph vs per-sector correlated gap
    num["gap_constructions"] = {}
    for d in (5, 7):
        g = cached_result("gap_constructions", f"gap_constructions_d{d}.json")
        k9 = g["kappa"]["0.9"]
        num["gap_constructions"][str(d)] = {
            "shots": g["shots"], "auc_shared": g["auc"]["shared"], "auc_per_sector": g["auc"]["per_sector"],
            "ratio_L09_per_sector_over_shared": k9["ratio_per_sector_over_shared"], "ratio_ci95": g["ratio_ci95"]["0.9"],
            "L09_shared": k9["shared"][0], "L09_per_sector": k9["per_sector"][0], "L09_gnn": k9["gnn"][0],
            "sign_inconsistent_fraction": g["per_sector_sign_inconsistent"]["fraction"],
            "sign_inconsistent_failure_rate": g["per_sector_sign_inconsistent"]["failure_rate"], "checks": g["checks"]}
    return num


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(runs.BUILD_DIR, "text"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, "text_numbers.json")
    res = json.load(open(cache)) if os.path.exists(cache) else {}
    if "shot_level" not in res:   # the parts computed from the per-shot arrays and the run JSONs are cached
        res["shot_level"] = [shot_level(d) for d in runs.PSCAN_DISTANCES]
        res["pscan"] = {}
        for d in runs.PSCAN_DISTANCES:
            for _, _, p, shots in runs.pscan_runs(d):
                res["pscan"].setdefault(str(d), []).append({"p": p, "shots": shots, **hard_ler(d, p, shots)})
        res["slopes"] = {}
        for d, rows in res["pscan"].items():
            ps = np.array([r["p"] for r in rows])
            res["slopes"][d] = {n: slope(ps, [r[n]["ler"] for r in rows], [r[n]["n_fail"] for r in rows])
                                for n in NAMES}
    if "run_checks" not in res:
        res["run_checks"] = run_checks()
    res["derived"] = derived_numbers()
    json.dump(res, open(cache, "w"), indent=1)
    num = res["derived"]

    md = ["# Numbers quoted in the text", "",
          "Generated by make_text_numbers.py from the per-shot arrays (cached in build/text/text_numbers.json)",
          "and from the cached results of the other steps of make_all.py; the paper rounds these values.",
          "Numbers that appear in a table or are read off a figure are in tables/ and figures/.", "",
          "## Sec. II and Appendix B: decisions of the shared-graph gap vs PyMatching's correlated decoder", "",
          "| (d, r) | shots | decision mismatches | of which at exact ties (g_corr = 0) |", "|---|---|---|---|"]
    md += [f"| ({c}) | {v['shots']:.0e} | {v['mismatch']} | {v['mismatch_at_exact_ties']} |"
           for c, v in res["run_checks"].items()]
    md += ["", "## Sec. III A (Fig. 1)", "", "Eq. (4) at kappa = 0.9: L_pair / L_GNN = hard-decision factor x selection factor", ""]
    for n, lab in (("mwpmcorr", "MWPM_corr"), ("mwpm", "MWPM_uncorr")):
        md += [f"- {lab}, d = {d}: {v['pair_ratio']:.3f} = {v['hard_factor']:.3f} x {v['selection_factor']:.3f}"
               for d, v in num["eq4_kappa_0.9"][n].items()]
    md += ["", "L_uncorr / L_corr at kappa = 0.9: " + ", ".join(
        f"d = {d}: {v:.2f}" for d, v in num["L_uncorr_over_L_corr_kappa_0.9"].items())]
    md += ["", "## Sec. III B (Figs. 2 and 3), Appendix D: p = 0.005, 1e8 shots per configuration", "",
           "Shots on which the decisions of MWPM_corr and the GNN differ: " + ", ".join(
               f"({c}) {100 * v:.2f}%" for c, v in num["decisions_differ"].items()), "",
           "| quantity | (5, 5) | (7, 7) | (9, 9) |", "|---|---|---|---|"]
    sl = res["shot_level"]
    rows = [("decisions of MWPM_corr and GNN differ", lambda s: f"{100 * s['disagree_frac']:.2f}%"),
            ("MWPM_uncorr failures shared by GNN", lambda s: f"{100 * s['shared_frac_of_mwpm_failures']['mwpm']:.1f}%"),
            ("MWPM_corr failures shared by GNN",
             lambda s: f"{100 * s['shared_frac_of_mwpm_failures']['mwpmcorr']:.1f}%")]
    sl = res["shot_level"]
    rows = [("decisions of MWPM_corr and GNN differ", lambda s: f"{100 * s['disagree_frac']:.2f}%"),
            ("MWPM_uncorr failures shared by GNN", lambda s: f"{100 * s['shared_frac_of_mwpm_failures']['mwpm']:.1f}%"),
            ("MWPM_corr failures shared by GNN",
             lambda s: f"{100 * s['shared_frac_of_mwpm_failures']['mwpmcorr']:.1f}%")]
    for t in TAIL_DB:
        rows += [(f"MWPM_corr failures with g_GNN > {t} dB",
                  lambda s, t=t: f"{100 * s[f'tail_{t}dB']['corr_failures_with_large_logit_frac']:.1f}%"),
                 ("  of these, GNN correct", lambda s, t=t: f"{100 * s[f'tail_{t}dB']['gnn_correct_among_them']:.1f}%"),
                 (f"GNN failures with g_corr > {t} dB",
                  lambda s, t=t: f"{100 * s[f'tail_{t}dB']['gnn_failures_with_large_gap_frac']:.1f}%"),
                 ("  of these, MWPM_corr correct",
                  lambda s, t=t: f"{100 * s[f'tail_{t}dB']['corr_correct_among_them']:.1f}%")]
    rows += [("bound of g_uncorr (dB)", lambda s: f"{s['bound_db']['mwpm']:.2f}"),
             ("shots at the bound of g_uncorr", lambda s: f"{100 * s['at_bound_frac']['mwpm']:.2f}%"),
             ("shots at the maximum of g_corr", lambda s: f"{100 * s['at_bound_frac']['mwpmcorr']:.2f}%")]
    md += [f"| {name} | " + " | ".join(fmt(s) for s in sl) + " |" for name, fmt in rows]
    md += ["", "Share of the exchange penalty AUC(own) - AUC(|other|) recovered by the other score signed by "
           "agreement (Table I): " + ", ".join(
               f"({c}) MWPM_corr {100 * v['mwpm']:.1f}%, GNN {100 * v['gnn']:.1f}%"
               for c, v in num["sign_recovery"].items() if int(c.split(",")[0]) >= 7)]
    md += ["", "## Sec. III A (Fig. 1, bottom): p-scan, hard-decision LER (kappa = 1)", "",
           "| d | p | shots | L_uncorr | L_corr | L_GNN | L_corr/L_uncorr | L_corr/L_GNN |",
           "|---|---|---|---|---|---|---|---|"]
    for d, rws in res["pscan"].items():
        for r in rws:
            md.append(f"| {d} | {r['p']} | {r['shots']:.0e} | {r['mwpm']['ler']:.3e} | {r['mwpmcorr']['ler']:.3e} | "
                      f"{r['gnn']['ler']:.3e} | {r['mwpmcorr']['ler'] / r['mwpm']['ler']:.3f} | "
                      f"{r['mwpmcorr']['ler'] / r['gnn']['ler']:.3f} |")
    md += ["", "Slopes d ln L / d ln p (weighted fit over the p-scan):", ""]
    md += [f"- d = {d}: " + ", ".join(f"{n} {b:.2f} +- {e:.2f}" for n, (b, e) in v.items())
           for d, v in res["slopes"].items()]

    ex = num["exact_d3"]
    md += ["", "## Sec. III C (Fig. 4, Table II) and Appendix F: exact posterior at (d, r) = (3, 3)", "",
           f"- detector error model without decomposition: {ex['detectors']} detectors, "
           f"{ex['error_mechanisms']} independent error mechanisms",
           f"- non-trivial shots: {ex['nontrivial']:,} of {ex['shots']:.0e}",
           f"- reduction of the gap's KL divergence by the correlations: {100 * ex['kl_reduction_by_correlations']:.1f}%"]
    md += ["", "## Sec. III D (Fig. 5, Table V)", "",
           f"- configurations with GNN alpha < 1 at p = 0.005: {num['gnn_alpha_below_1']} of {len(runs.CONFIGS)}",
           "- native log-loss GNN - MWPM_corr (mbit/shot): " + ", ".join(
               f"({c}) {v:+.2f}" for c, v in num["logloss_gnn_minus_corr_mbit"].items())]
    md += ["", "## Appendix B: shared-graph vs per-sector correlated gap (same shots, same decisions)", ""]
    for d, g in num["gap_constructions"].items():
        md += [f"- d = r = {d} ({g['shots']:.0e} shots): AUC {g['auc_shared']:.4f} (shared) -> {g['auc_per_sector']:.4f} "
               f"(per sector); L(0.9) per sector / shared = {g['ratio_L09_per_sector_over_shared']:.2f} "
               f"(95% {g['ratio_ci95'][0]:.2f}-{g['ratio_ci95'][1]:.2f}); L(0.9) shared {g['L09_shared']:.2e}, "
               f"per sector {g['L09_per_sector']:.2e}, GNN {g['L09_gnn']:.2e}; sign-inconsistent "
               f"{100 * g['sign_inconsistent_fraction']:.2f}% of shots, {100 * g['sign_inconsistent_failure_rate']:.1f}% "
               f"of them failures; checks {g['checks']}"]
    t, nb, g2 = num["corr_alpha_trend"], num["gnn_alpha_neighbours"], num["gnn_two_parameter"]
    md += ["", "## Appendix C: calibration of all configurations", "",
           f"- largest statistical uncertainty of a slope (Fisher or binomial bootstrap): {num['alpha_sigma_max']:.4f}",
           f"- MWPM_corr slopes, weighted linear model: {t['b_d']:+.4f} +- {t['se_b_d']:.4f} per unit of d, "
           f"{t['b_r']:+.4f} +- {t['se_b_r']:.4f} per round",
           f"- GNN slopes of neighbouring configurations differ by up to {nb['max_abs_delta']:.3f}, i.e. "
           f"{nb['min_abs_sigmas']:.0f}-{nb['max_abs_sigmas']:.0f} standard deviations",
           f"- GNN two-parameter fit: chi2_nu {g2['chi2nu_min']:.1f}-{g2['chi2nu_max']:.1f}, offsets "
           f"{g2['offset_db_min']:+.2f} to {g2['offset_db_max']:+.2f} dB",
           "- largest log-loss reduction by a global rescaling (mbit/shot): " + ", ".join(
               f"{n} {v:.2f}" for n, v in num["logloss_rescaling_gain_max_mbit"].items())]
    md += ["", "## Appendix D", "", f"- Spearman rank correlation of g_corr and g_GNN at (9, 9): {num['spearman_rho_d9']:.4f}",
           "- tails and shared failures: Sec. III B table above"]
    h = num["high_confidence_d9"]
    md += [f"- (9, 9), GNN failures accepted at kappa <= 0.4: {h['gnn_failures_kappa_0.4']}; at kappa = 0.2: "
           f"{h['gnn_failures_kappa_0.2']:.0f}, predicted by the GNN's calibration (Eq. (5), alpha of Table V): "
           f"{h['predicted_kappa_0.2']:.2f}; among the 5% most confident shots: {h['in_top_5pct']}, logits "
           f"{h['top_5pct_logit_db'][0]:.1f}-{h['top_5pct_logit_db'][1]:.1f} dB",
           f"- of these {h['gnn_failures_kappa_0.4']}: MWPM_corr fails on {h['corr_fails']} (both matching decoders on "
           f"{h['both_matching_fail']}); median g_corr {h['g_corr_median_db']:.1f} dB, below the MWPM_corr threshold at "
           f"kappa = 0.4 ({h['corr_threshold_kappa_0.4_db']:.1f} dB)",
           "- median number of detection events: " + ", ".join(f"{g} {v:.0f}" for g, v in h["detection_events_median"].items())
           + "; median minimum matching weight (nats): " + ", ".join(
               f"{g} {v:.0f}" for g, v in h["min_matching_weight_median_nats"].items())
           + " (groups and all features: build/high_confidence/high_confidence_failures.md)",
           "- GNN failures accepted at kappa = 0.1, observed / predicted: " + ", ".join(
               f"({c}) {v['observed_kappa_0.1']:.0f} / {v['predicted_kappa_0.1']:.1f}" for c, v in num["gnn_deep_cut"].items()),
           "- GNN L(0.2)/L(0.4): " + ", ".join(f"({c}) {v['L0.2_over_L0.4']:.2f}" for c, v in num["gnn_deep_cut"].items())]
    md += ["", "## Appendix E: local slopes (Fig. 8) and alpha vs p (Fig. 9)", "",
           f"- largest Fisher standard error of a local slope: {num['local_slopes_sigma_max']:.4f}"]
    for k, v in num["local_slopes"].items():
        md.append(f"- local slopes {k}: " + ", ".join(f"[{w}] {a:.3f}" for w, a in v.items()))
    for n, per_d in num["alpha_vs_p"].items():
        for d, alphas in per_d.items():
            vals = list(alphas.values())
            md.append(f"- alpha vs p, {n}, d = {d}: " + ", ".join(f"{p}: {x:.3f}" for p, x in alphas.items())
                      + f"  (range {max(vals) - min(vals):.3f})")
    open(os.path.join(a.out, "text_numbers.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
