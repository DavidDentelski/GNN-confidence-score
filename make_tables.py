"""
Tables I, V and VIII of the paper, and summary tables of the three native
decoder-confidence pairs
    MWPM_uncorr + gap,  MWPM_corr + correlated gap,  GNN + logit
on the SAME shots (p = 0.005, 1e8 shots per configuration (d, r)).

  summary_auc.tex/.csv          tie-aware AUC of each decoder's failures ranked by its own score
  summary_calibration.tex/.csv  calibration slope alpha (binomial ML fit, g in [10, 30] dB),
                                reduced chi^2 and failure-weighted mean |data/fit - 1| in the window
  summary_logloss.tex/.csv      mean log-loss (proper scoring rule) of the predictive posterior
                                q(L | s) = sigmoid(a * lambda(s)), lambda = the decoder's signed
                                log-odds, native (a = 1) and with the best global a; differences
                                between pairs equal differences in mean KL divergence to the true
                                posterior P(L | s), since the entropy H(L | S) cancels.

Manuscript tables (in --paper-out; make_all.py copies them to tables/ under the
paper's numbering):

  table1_auc.tex      Table I: own score, magnitude of the other score, other score signed
                      by agreement (MWPM_corr vs GNN); last rows MWPM_uncorr's own gap and the
                      logit. Bold: the better of own score and |other score| where their paired
                      difference is resolved (its paired DeLong 95% interval excludes zero)
  appB_alpha.tex      Table V: alpha, chi2_nu (pooled, IVW mean), two-parameter fit,
                      log-loss, for MWPM_corr, GNN and MWPM_uncorr
  appD_window.tex     Table VIII: fit-window robustness, alpha range (max shift) and median chi2_nu

Inputs: the cached intermediate results of make_exchange.py, make_calibration.py,
make_calibration_extras.py and make_robustness.py run with --mwpm mwpm / --mwpm mwpmcorr
and --out <build>/<variant> (as done by make_all.py); the log-loss is computed from the
per-shot arrays once and cached in <build>/tables/derived/logloss.json.
"""

import argparse
import json
import os

import numpy as np
from scipy.optimize import minimize_scalar

import make_calibration as mf4
import runs

P, SHOTS = runs.P_MAIN, runs.SHOTS
CONFIGS = runs.CONFIGS
ROWS = [("mwpm", r"MWPM$_{\rm uncorr}$"), ("mwpmcorr", r"MWPM$_{\rm corr}$"), ("gnn", "GNN")]
DB2NAT, LN2 = np.log(10) / 10, np.log(2)
Z_RESOLVED = 1.96  # Table I bolds a difference whose paired 95% interval excludes zero (Appendix A)


def native_auc(build):
    """{pair: {(d, r): auc}} from the crossed caches of both variants."""
    out = {n: {} for n, _ in ROWS}
    for v in ("mwpm", "mwpmcorr"):
        for d, r in CONFIGS:
            auc = derived(build, v, f"crossed_d{d}_dt{r}_p{P}.json")["auc"]
            out[v][(d, r)] = auc["g_mwpm->mwpm"]
            out["gnn"][(d, r)] = auc["g_gnn->gnn"]
    return out


def calibration(build):
    """{pair: {(d, r) or 'pooled': (alpha, chi2_nu, mean_dev, sigma)}} via make_calibration's fits."""
    out = {n: {} for n, _ in ROWS}
    for v in ("mwpm", "mwpmcorr"):
        mf4.configure(v, os.path.join(build, v))
        binned = {(d, r): derived(build, v, f"calib_d{d}_dt{r}_p{P}_shots{SHOTS}.json") for d, r in CONFIGS}
        for key, b in list(binned.items()) + [("pooled", {
                n: {k: x.tolist() for k, x in mf4.pool_counts(list(binned.values()), n).items()}
                for n in ("MWPM", "GNN")})]:
            for name, row in (("MWPM", v), ("GNN", "gnn")):
                f = mf4.full_fit(b, name)
                dev = mf4.window_deviation(b, name, f["alpha"])[0]
                out[row][key] = (f["alpha"], f["chi2_nu"], dev, f["sigma"])
    return out


def logloss(build):
    """{pair: {(d, r): (native, se, best_a, rescaled, se)}} in mbit/shot (cached)."""
    cache = os.path.join(build, "tables", "derived", "logloss.json")   # a fitted on all shots
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    res = json.load(open(cache)) if os.path.exists(cache) else {}
    for d, r in CONFIGS:
        for n, _ in ROWS:
            key = f"{n}|{d},{r}"
            if key in res:
                continue
            runs.require_arrays(d, r, P, SHOTS, decoders=[n])
            x = np.load(runs.array_path(n, d, r, P, SHOTS)).astype(np.float64) * DB2NAT
            a = minimize_scalar(lambda a, x=x: np.mean(np.logaddexp(0, -a * x)), bounds=(0.3, 2.0),
                                method="bounded").x
            l1 = np.logaddexp(0, -x) / LN2 * 1e3
            la = np.logaddexp(0, -a * x) / LN2 * 1e3
            res[key] = [float(l1.mean()), float(l1.std() / np.sqrt(len(x))), float(a),
                        float(la.mean()), float(la.std() / np.sqrt(len(x)))]
            del x, l1, la
            json.dump(res, open(cache, "w"))
            print(f"  log-loss {key}: {res[key][0]:.3f} mbit/shot (a=1), {res[key][3]:.3f} (a={a:.3f})")
    out = {n: {} for n, _ in ROWS}
    for key, v in res.items():
        n, cfg = key.split("|")
        out[n][tuple(int(c) for c in cfg.split(","))] = v
    return out


def latex_table(header_cols, body_lines, caption, label):
    return "\n".join([r"\begin{table*}", r"\centering",
                      rf"\begin{{tabular}}{{l|{'c' * len(header_cols)}}}", r"\toprule",
                      " & " + " & ".join(header_cols) + r" \\", r"\hline", r"\midrule", *body_lines,
                      r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                      r"\end{table*}"]) + "\n"


def derived(build, variant, name):
    path = os.path.join(build, variant, "derived", name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run the {variant} steps of make_all.py first")
    return json.load(open(path))


def paper_table(spec, header_lines, body_lines, caption, label):
    return "\n".join([r"\begin{table*}", r"\centering", rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
                      *header_lines, r"\hline", r"\midrule", *body_lines, r"\bottomrule", r"\end{tabular}",
                      rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table*}"]) + "\n"


def write_paper_auc(build, out):
    """Table I: each decoder's failures ranked by its own score, by the magnitude of the other
    score and by the other score signed by agreement; last rows MWPM_uncorr ranked by its own gap and by the logit."""
    name = lambda c: f"d{c[0]}_dt{c[1]}_p{P}.json"
    cr = {v: {c: derived(build, v, "crossed_" + name(c)) for c in CONFIGS} for v in ("mwpmcorr", "mwpm")}
    da = {c: derived(build, "mwpmcorr", "decision_aware_" + name(c)) for c in CONFIGS}
    rob = {v: {c: derived(build, v, f"robustness_d{c[0]}_dt{c[1]}.json")["delta_auc"] for c in CONFIGS}
           for v in ("mwpmcorr", "mwpm")}
    configs = CONFIGS

    def bold_rows(v_own, v_oth, dauc):
        """Own and |other| rows; bold marks the better score where the paired difference is resolved."""
        own, oth = [], []
        for a, b, dd in zip(v_own, v_oth, dauc):
            resolved = abs(dd["delta"]) >= Z_RESOLVED * dd["se"]
            own.append(rf"\textbf{{{a:.4f}}}" if resolved and a > b else f"{a:.4f}")
            oth.append(rf"\textbf{{{b:.4f}}}" if resolved and b > a else f"{b:.4f}")
        return " & ".join(own), " & ".join(oth)

    body = []
    for dec, own, other, own_lab, other_lab, lab in (
            ("mwpm", "g_mwpm", "g_gnn", r"g_{\rm corr}", r"g_{\rm GNN}", ROWS[1][1]),
            ("gnn", "g_gnn", "g_mwpm", r"g_{\rm GNN}", r"g_{\rm corr}", "GNN")):
        v_own = [cr["mwpmcorr"][c]["auc"][f"{own}->{dec}"] for c in configs]
        v_oth = [cr["mwpmcorr"][c]["auc"][f"{other}->{dec}"] for c in configs]
        row_own, row_oth = bold_rows(v_own, v_oth, [rob["mwpmcorr"][c][dec] for c in configs])
        body += [rf"{lab} & ${own_lab}$ & {row_own} \\",
                 rf"{lab} & $|{other_lab}|$ & {row_oth} \\",
                 rf"{lab} & $\pm {other_lab}$ (signed) & "
                 + " & ".join(f"{da[c]['auc_signed_other'][dec]:.4f}" for c in configs) + r" \\", r"\hline"]
    v_own = [cr["mwpm"][c]["auc"]["g_mwpm->mwpm"] for c in configs]
    v_oth = [cr["mwpm"][c]["auc"]["g_gnn->mwpm"] for c in configs]
    row_own, row_oth = bold_rows(v_own, v_oth, [rob["mwpm"][c]["mwpm"] for c in configs])
    body += [rf"{ROWS[0][1]} & $g_{{\rm uncorr}}$ & {row_own} \\",
             rf"{ROWS[0][1]} & $|g_{{\rm GNN}}|$ & {row_oth} \\"]
    open(os.path.join(out, "table1_auc.tex"), "w").write(paper_table(
        "ll|" + "c" * len(configs),
        ["Decoder & Ranked by & " + " & ".join(rf"$({d}, {r})$" for d, r in configs) + r" \\"], body,
        rf"Tie-aware AUC of each decoder's logical failures at $p = {P}$ ($10^8$ shots per configuration, "
        r"the same shots for all decoders; ties receive half credit), ranked by its own score, by the "
        r"magnitude of the other decoder's score (bold: the better of these two), and by the other "
        r"decoder's score signed by agreement ($+|g|$ where the two decoders agree, $-|g|$ where they "
        r"disagree). Last rows: the failures of uncorrelated MWPM, ranked by its own gap and by the logit.",
        "auc_table"))


def write_paper_calibration(build, out, cal, ll):
    """Calibration table: one-parameter fit (per configuration, pooled, IVW mean), two-parameter
    fit and native log-loss for MWPM_corr and GNN; one-parameter fit and log-loss for MWPM_uncorr."""
    drift = {v: derived(build, v, "alpha_drift.json")["fits"] for v in ("mwpmcorr", "mwpm")}
    key = lambda c: f"{c[0]},{c[1]}"
    body = []
    for row, src, two_par in (("mwpmcorr", drift["mwpmcorr"]["MWPM"], True),
                              ("gnn", drift["mwpmcorr"]["GNN"], True), ("mwpm", None, False)):
        fits = cal[row]
        ivw = mf4.meta_mean([{"alpha": fits[c][0], "sigma": fits[c][3]} for c in CONFIGS])["mean"]
        lab = dict(ROWS)[row]
        body.append(rf"{lab} & $\alpha$ & " + " & ".join(f"{fits[c][0]:.3f}" for c in CONFIGS)
                    + f" & {fits['pooled'][0]:.3f} & {ivw:.3f}" + r" \\")
        body.append(r" & $\chi^2_\nu$ & " + " & ".join(f"{fits[c][1]:.1f}" for c in CONFIGS)
                    + f" & {fits['pooled'][1]:.1f} & --" + r" \\")
        if two_par:
            body.append(r" & $\alpha_2$ (2-par.) & " + " & ".join(f"{src[key(c)]['alpha2']:.3f}" for c in CONFIGS)
                        + r" & -- & -- \\")
            body.append(r" & $\chi^2_\nu$ (2-par.) & "
                        + " & ".join(f"{src[key(c)]['chi2nu_2p']:.1f}" for c in CONFIGS) + r" & -- & -- \\")
        body.append(r" & log-loss & " + " & ".join(f"{ll[row][c][0]:.2f}" for c in CONFIGS) + r" & -- & -- \\")
        body.append(r"\hline")
    sig = max(f[k]["sigma_boot"] for f in (drift["mwpmcorr"]["MWPM"], drift["mwpmcorr"]["GNN"],
                                            drift["mwpm"]["MWPM"]) for k in f)
    open(os.path.join(out, "appB_alpha.tex"), "w").write(paper_table(
        "ll|" + "c" * len(CONFIGS) + "|c|c",
        [" & & " + " & ".join(rf"$({d}, {r})$" for d, r in CONFIGS) + r" & Pooled & IVW mean \\"], body[:-1],
        rf"Calibration of the three confidence scores at $p = {P}$: slope $\alpha$ of the binomial "
        r"maximum-likelihood fit of Eq.~(4) over $g \in [10, 30]$~dB per configuration, for shot-pooled data "
        r"and as an inverse-variance-weighted descriptive mean, with the reduced $\chi^2$ over integer-dB bins; "
        r"slope and $\chi^2_\nu$ of the two-parameter fit $P = 1/(1+10^{(\alpha_2 g - c)/10})$; native "
        rf"log-loss (mbit/shot) of $q(L|s) = \sigma(\lambda(s))$. Statistical uncertainties of $\alpha$ "
        rf"(binomial bootstrap) are at most ${sig:.4f}$.", "alpha_table"))


def write_paper_window(build, out):
    """Fit-window robustness: range of alpha over configurations (max shift from the baseline window)
    and median chi2_nu, for MWPM_corr and GNN."""
    wr = derived(build, "mwpmcorr", "window_robustness.json")
    gof = {}
    with open(os.path.join(build, "mwpmcorr", "tables", "appD_window_gof.csv")) as f:
        head = f.readline().strip().split(",")
        for line in f:
            vals = line.strip().split(",")
            gof[vals[0].replace(";", ",")] = dict(zip(head[1:], map(float, vals[1:])))
    labels = list(wr)
    body = []
    for lab in labels:
        cells = []
        for dec in ("MWPM", "GNN"):
            a = [wr[lab][dec][f"{d},{r}"]["alpha"] for d, r in CONFIGS]
            b = [wr[labels[0]][dec][f"{d},{r}"]["alpha"] for d, r in CONFIGS]
            rng = f"{min(a):.3f}--{max(a):.3f}"
            cells += [rng if lab == labels[0] else f"{rng} ({max(abs(x - y) for x, y in zip(a, b)):.3f})",
                      f"{gof[lab][dec + '_chi2_median']:.1f}"]
        body.append(f"{lab} & " + " & ".join(cells) + r" \\")
    open(os.path.join(out, "appD_window.tex"), "w").write(paper_table(
        "l|cc|cc",
        [rf" & \multicolumn{{2}}{{c|}}{{{ROWS[1][1]}}} & \multicolumn{{2}}{{c}}{{GNN}} \\",
         r"Fit window (dB) & $\alpha$ range & median $\chi^2_\nu$ & $\alpha$ range & median $\chi^2_\nu$ \\"],
        body,
        r"Robustness of the calibration slopes to the fit window: for each window, the range of $\alpha$ "
        r"across the eight configurations, with the largest per-configuration shift relative to the "
        r"pre-specified window $g \in [10, 30]$~dB in parentheses, and the median over configurations of "
        r"the reduced $\chi^2$ of the one-parameter fit refitted in that window.", "window_table"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default=runs.BUILD_DIR)
    ap.add_argument("--out", default=os.path.join(runs.BUILD_DIR, "tables"))
    ap.add_argument("--paper-out", default=os.path.join(runs.BUILD_DIR, "tables", "paper"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.paper_out, exist_ok=True)
    cols = [rf"$({d}, {r})$" for d, r in CONFIGS]
    fmt = lambda v, f: (f % v) if v is not None else "--"

    auc = native_auc(a.build)
    with open(os.path.join(a.out, "summary_auc.csv"), "w") as f:
        f.write("pair," + ",".join(f"d{d}_r{r}" for d, r in CONFIGS) + "\n")
        for n, _ in ROWS:
            f.write(n + "," + ",".join(fmt(auc[n].get(c), "%.5f") for c in CONFIGS) + "\n")
    body = [lab + " & " + " & ".join(fmt(auc[n].get(c), "%.4f") for c in CONFIGS) + r" \\" for n, lab in ROWS]
    open(os.path.join(a.out, "summary_auc.tex"), "w").write(latex_table(
        cols, body, rf"Tie-aware AUC of each decoder's logical failures ranked by its own confidence score, "
                    rf"at $p = {P}$ ($10^8$ shots per configuration, the same shots for all three pairs; "
                    r"ties receive half credit).", "summary_auc_table"))

    cal = calibration(a.build)
    ccols = cols + ["Pooled"]
    keys = CONFIGS + ["pooled"]
    with open(os.path.join(a.out, "summary_calibration.csv"), "w") as f:
        f.write("pair,quantity," + ",".join([f"d{d}_r{r}" for d, r in CONFIGS] + ["pooled"]) + "\n")
        for n, _ in ROWS:
            for qi, q in enumerate(("alpha", "chi2_nu", "mean_rel_dev")):
                f.write(f"{n},{q}," + ",".join(fmt(cal[n][k][qi] if k in cal[n] else None, "%.5f")
                                               for k in keys) + "\n")
    body = []
    for qi, (q, ff) in enumerate(((r"$\alpha$", "%.3f"), (r"$\chi^2_\nu$", "%.1f"),
                                  (r"$\langle|{\rm data}/{\rm fit}-1|\rangle$", "%.3f"))):
        for n, lab in ROWS:
            body.append(f"{lab} {q} & " + " & ".join(fmt(cal[n][k][qi] if k in cal[n] else None, ff)
                                                    for k in keys) + r" \\")
        if qi < 2:
            body.append(r"\hline")
    open(os.path.join(a.out, "summary_calibration.tex"), "w").write(latex_table(
        ccols, body, r"Calibration of the three confidence scores at $p = 0.005$: slope $\alpha$ of the "
                     r"binomial maximum-likelihood fit of Eq.~(4) over $g \in [10, 30]$~dB, reduced $\chi^2$ "
                     r"over integer-dB bins, and failure-weighted mean relative deviation of the bins from the "
                     r"fit in the window (an effect size independent of the sample size). All pairs are "
                     r"evaluated on the same $10^8$ shots per configuration.", "summary_calibration_table"))

    ll = logloss(a.build)
    with open(os.path.join(a.out, "summary_logloss.csv"), "w") as f:
        f.write("pair,d,r,native_mbit,native_se,best_a,rescaled_mbit,rescaled_se\n")
        for n, _ in ROWS:
            for c in CONFIGS:
                if c in ll[n]:
                    f.write(f"{n},{c[0]},{c[1]}," + ",".join(f"{v:.5f}" for v in ll[n][c]) + "\n")
    body = []
    for n, lab in ROWS:
        body.append(f"{lab} ($a = 1$) & " + " & ".join(
            (rf"{ll[n][c][0]:.2f}$\pm${ll[n][c][1]:.2f}" if c in ll[n] else "--") for c in CONFIGS) + r" \\")
    body.append(r"\hline")
    for n, lab in ROWS:
        body.append(f"{lab} (best $a$) & " + " & ".join(
            (rf"{ll[n][c][3]:.2f} ({ll[n][c][2]:.3f})" if c in ll[n] else "--") for c in CONFIGS) + r" \\")
    open(os.path.join(a.out, "summary_logloss.tex"), "w").write(latex_table(
        cols, body, r"Mean log-loss (mbit/shot) of the predictive posterior $q(L|s) = \sigma(a\lambda(s))$ "
                    r"implied by each confidence score ($\lambda$: the decoder's signed log-odds), natively "
                    r"($a = 1$) and with the best global rescaling $a$ (in parentheses), at $p = 0.005$. "
                    r"Differences between rows equal differences in the mean Kullback-Leibler divergence to "
                    r"the true posterior.", "summary_logloss_table"))
    print(f"summary tables -> {a.out}")

    write_paper_auc(a.build, a.paper_out)
    write_paper_calibration(a.build, a.paper_out, cal, ll)
    write_paper_window(a.build, a.paper_out)
    print(f"manuscript tables -> {a.paper_out}")


if __name__ == "__main__":
    main()
