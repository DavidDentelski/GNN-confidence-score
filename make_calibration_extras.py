"""
Calibration diagnostics: Fig. 8, the two-parameter fit of Table V and the
goodness of fit of Table VIII (Appendices C and E).

  Is the variation of alpha across (d, r) real, and systematic?
       - statistical uncertainty of the ML slope by a binomial bootstrap of the
         sufficient counts (compared with the Fisher error),
       - constancy test (Cochran Q), weighted linear trend alpha = b0 + bd d + br r,
         consecutive steps at fixed d and at fixed r in units of sigma,
       - two-parameter (Platt: slope + offset) fit, P = 1/(1 + 10^((a g - c)/10)):
         does the drift survive a free intercept, and how much of the misfit does
         an offset absorb?
  Goodness of fit in every fit window (chi2_nu, and the sample-size-free
  failure-weighted mean |data/fit - 1|), and local slopes in 5-dB windows.

Reads the binned sufficient counts cached by make_calibration.py (run with the same
--mwpm / --out). Outputs (OUT/tables, OUT):
    appD_alpha_drift.tex/.csv     per-configuration fits + trend tests (derived/alpha_drift.json)
    appD_window_gof.tex/.csv      goodness of fit per fit window
    appD_local_slopes.tex/.csv    alpha in 5-dB sub-windows
    appD_local_slopes_sigma.csv   their Fisher standard errors
    appD_local_slopes.pdf/.png    Fig. 8: local slopes with Fisher error bars
"""

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from scipy.stats import chi2 as chi2dist

import make_calibration as mf4
import runs

B_BOOT = 100
LOCAL = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 50)]
C = mf4.LN10 / 10.0
plt.rcParams.update({"pdf.fonttype": 42, "savefig.bbox": "tight"})


def window(b, name, lo, hi):
    n = np.asarray(b[name]["fine_counts"], float)
    k = np.asarray(b[name]["fine_err"], float)
    g = (np.arange(len(n)) + 0.5) * mf4.FINE_BIN_DB
    m = (g >= lo) & (g <= hi) & (n > 0)
    return n[m], k[m], C * g[m]


def fit1(n, k, x):
    nll = lambda a: -np.sum(k * np.log(expit(-a * x)) + (n - k) * np.log(expit(a * x)))
    return minimize_scalar(nll, bounds=(0.05, 3.0), method="bounded", options={"xatol": 1e-9}).x


def fisher_sigma(n, x, a):
    """Unscaled Fisher standard error of the one-parameter ML slope a."""
    pp = expit(-a * x)
    return 1 / np.sqrt(np.sum(n * pp * (1 - pp) * x ** 2))


def fit2(n, k, x):
    def nll(t):
        z = t[0] * x - t[1]
        return -np.sum(k * np.log(expit(-z)) + (n - k) * np.log(expit(z)))
    a, c = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead",
                    options={"xatol": 1e-9, "fatol": 1e-7, "maxiter": 4000}).x
    return a, c / C  # offset in dB


def gof(b, name, prob, lo, hi, n_par):
    """chi2_nu over integer-dB display bins and failure-weighted mean |data/fit - 1|."""
    g, ph, _, k, valid = mf4.display_arrays(b[name]["counts"], b[name]["err_counts"])
    m = valid & (g >= lo) & (g <= hi)
    pf = prob(g[m])
    r = (np.log(ph[m]) - np.log(pf)) / np.sqrt((1 - ph[m]) / k[m])
    dev = np.sum(k[m] * np.abs(ph[m] / pf - 1)) / np.sum(k[m])
    return float(np.sum(r ** 2) / max(m.sum() - n_par, 1)), float(dev)


def per_config(b, name, rng):
    n, k, x = window(b, name, mf4.G_MIN, mf4.G_MAX)
    a1 = fit1(n, k, x)
    s_f = fisher_sigma(n, x, a1)
    boots = [fit1(n, rng.binomial(n.astype(np.int64), k / n).astype(float), x) for _ in range(B_BOOT)]
    a2, c_db = fit2(n, k, x)
    chi1, dev1 = gof(b, name, lambda g: 1 / (1 + 10 ** (a1 * g / 10)), mf4.G_MIN, mf4.G_MAX, 1)
    chi2_, dev2 = gof(b, name, lambda g: 1 / (1 + 10 ** ((a2 * g - c_db) / 10)), mf4.G_MIN, mf4.G_MAX, 2)
    return {"alpha": a1, "sigma_fisher": s_f, "sigma_boot": float(np.std(boots, ddof=1)),
            "alpha2": float(a2), "offset_db": float(c_db), "chi2nu_1p": chi1, "chi2nu_2p": chi2_,
            "dev_1p": dev1, "dev_2p": dev2}


def trend(configs, fits):
    a = np.array([fits[c]["alpha"] for c in configs])
    s = np.array([fits[c]["sigma_boot"] for c in configs])
    w = 1 / s ** 2
    mean = np.sum(w * a) / np.sum(w)
    q = float(np.sum(w * (a - mean) ** 2))
    out = {"Q": q, "dof": len(a) - 1, "p_constant": float(chi2dist.sf(q, len(a) - 1))}
    if len(configs) >= 4:
        X = np.column_stack([np.ones(len(a)), [c[0] for c in configs], [c[1] for c in configs]])
        cov = np.linalg.inv(X.T @ (w[:, None] * X))
        beta = cov @ X.T @ (w * a)
        res = a - X @ beta
        chi = float(np.sum(w * res ** 2))
        se = np.sqrt(np.diag(cov))
        out.update({"b_d": float(beta[1]), "se_b_d": float(se[1]), "b_r": float(beta[2]),
                    "se_b_r": float(se[2]), "trend_chi2": chi, "trend_dof": len(a) - 3})
    steps = []
    for key, fixed in (("r at fixed d", 0), ("d at fixed r", 1)):
        for v in sorted({c[fixed] for c in configs}):
            grp = sorted([c for c in configs if c[fixed] == v], key=lambda c: c[1 - fixed])
            for c0, c1 in zip(grp[:-1], grp[1:]):
                da = fits[c1]["alpha"] - fits[c0]["alpha"]
                steps.append({"from": c0, "to": c1, "delta": da,
                              "sigmas": da / np.hypot(fits[c0]["sigma_boot"], fits[c1]["sigma_boot"])})
    out["steps"] = steps
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mwpm", choices=sorted(mf4.VARIANTS), default="mwpmcorr")
    ap.add_argument("--out", default=None, help="output folder (default build/<mwpm>/)")
    args = ap.parse_args()
    mf4.configure(args.mwpm, args.out or os.path.join(runs.BUILD_DIR, args.mwpm))
    os.makedirs(mf4.TABLE_DIR, exist_ok=True)
    rng = np.random.default_rng(2026)
    names = {"MWPM": mf4.DISPLAY["MWPM"], "GNN": "GNN"}

    configs = mf4.CONFIGS
    binned = {c: mf4.get_binned(*c) for c in configs}

    # ---- alpha drift ------------------------------------------------------------------
    fits = {n: {c: per_config(binned[c], n, rng) for c in configs} for n in names}
    trends = {n: trend(configs, fits[n]) for n in names}
    json.dump({"fits": {n: {f"{c[0]},{c[1]}": v for c, v in f.items()} for n, f in fits.items()},
               "trends": {n: {**t, "steps": [{**s, "from": list(s["from"]), "to": list(s["to"])}
                                             for s in t["steps"]]} for n, t in trends.items()}},
              open(os.path.join(mf4.DERIVED_DIR, "alpha_drift.json"), "w"), indent=1)
    cols = ["alpha", "sigma_fisher", "sigma_boot", "alpha2", "offset_db", "chi2nu_1p", "chi2nu_2p", "dev_1p", "dev_2p"]
    with open(os.path.join(mf4.TABLE_DIR, "appD_alpha_drift.csv"), "w") as f:
        f.write("decoder,d,r," + ",".join(cols) + "\n")
        for n in names:
            for c in configs:
                f.write(f"{n},{c[0]},{c[1]}," + ",".join(f"{fits[n][c][k]:.6g}" for k in cols) + "\n")
        for n in names:
            t = trends[n]
            f.write(f"# {n}: Q={t['Q']:.1f} on {t['dof']} dof, p={t['p_constant']:.2e}"
                    + (f"; b_d={t['b_d']:+.5f}+-{t['se_b_d']:.5f}, b_r={t['b_r']:+.5f}+-{t['se_b_r']:.5f}, "
                       f"residual chi2={t['trend_chi2']:.1f}/{t['trend_dof']}" if "b_d" in t else "") + "\n")
            for s in t["steps"]:
                f.write(f"#   {tuple(s['from'])}->{tuple(s['to'])}: {s['delta']:+.4f} ({s['sigmas']:+.1f} sigma)\n")
    lines = [r"\begin{table*}", r"\centering", r"\begin{tabular}{ll|ccc|cc|cc|cc}", r"\toprule",
             r"Decoder & $(d, r)$ & $\alpha$ & $\sigma_{\rm Fisher}$ & $\sigma_{\rm boot}$ & $\alpha_2$ & "
             r"$c$ (dB) & $\chi^2_\nu$ (1p) & $\chi^2_\nu$ (2p) & $\langle|{\rm dev}|\rangle$ (1p) & "
             r"$\langle|{\rm dev}|\rangle$ (2p) \\", r"\hline", r"\midrule"]
    for i, n in enumerate(names):
        for c in configs:
            v = fits[n][c]
            lines.append(rf"{names[n]} & $({c[0]}, {c[1]})$ & {v['alpha']:.4f} & {v['sigma_fisher']:.4f} & "
                         rf"{v['sigma_boot']:.4f} & {v['alpha2']:.4f} & {v['offset_db']:+.2f} & "
                         rf"{v['chi2nu_1p']:.1f} & {v['chi2nu_2p']:.1f} & {100 * v['dev_1p']:.1f}\% & "
                         rf"{100 * v['dev_2p']:.1f}\% \\")
        t = trends[n]
        summ = rf"constancy: $Q = {t['Q']:.0f}$ ({t['dof']} dof)"
        if "b_d" in t:
            summ += (rf"; $\partial\alpha/\partial d = {t['b_d']:+.4f}\pm{t['se_b_d']:.4f}$, "
                     rf"$\partial\alpha/\partial r = {t['b_r']:+.4f}\pm{t['se_b_r']:.4f}$")
        lines.append(rf"\multicolumn{{11}}{{l}}{{\footnotesize {names[n]} {summ}}} \\")
        if i == 0:
            lines.append(r"\hline")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Calibration slope across configurations ($p = 0.005$, window $g \in [10, 30]$~dB): "
              r"one-parameter ML slope $\alpha$ with its Fisher and binomial-bootstrap uncertainties; "
              r"two-parameter fit $P = 1/(1+10^{(\alpha_2 g - c)/10})$; reduced $\chi^2$ and failure-weighted "
              r"mean relative deviation for both models. The constancy test and the weighted linear trend use "
              r"the bootstrap uncertainties.}", r"\label{alpha_drift_table}", r"\end{table*}"]
    open(os.path.join(mf4.TABLE_DIR, "appD_alpha_drift.tex"), "w").write("\n".join(lines) + "\n")

    # ---- goodness of fit per window ---------------------------------------------------
    rows = []
    for label, lo, hi in mf4.WINDOW_CHOICES:
        hi = min(hi, mf4.FINE_MAX_DB)
        r = {"window": label}
        for n in names:
            chis, devs = [], []
            for c in configs:
                a = fit1(*window(binned[c], n, lo, hi))
                chi, dev = gof(binned[c], n, lambda g, a=a: 1 / (1 + 10 ** (a * g / 10)), lo, hi, 1)
                chis.append(chi)
                devs.append(dev)
            r[n] = (min(chis), max(chis), float(np.median(chis)), float(np.median(devs)))
        rows.append(r)
    with open(os.path.join(mf4.TABLE_DIR, "appD_window_gof.csv"), "w") as f:
        f.write("window," + ",".join(f"{n}_{q}" for n in names for q in ("chi2_min", "chi2_max", "chi2_median",
                                                                         "dev_median")) + "\n")
        for r in rows:
            # full precision: make_tables.py rounds these once for Table VIII
            f.write(r["window"].replace(",", ";") + "," + ",".join(f"{v:.10g}" for n in names for v in r[n]) + "\n")
    lines = [r"\begin{table}", r"\centering", r"\begin{tabular}{l|cc|cc}", r"\toprule",
             rf" & \multicolumn{{2}}{{c|}}{{{names['MWPM']}}} & \multicolumn{{2}}{{c}}{{GNN}} \\",
             r"Window (dB) & $\chi^2_\nu$ (median) & $\langle|{\rm dev}|\rangle$ & $\chi^2_\nu$ (median) & "
             r"$\langle|{\rm dev}|\rangle$ \\", r"\hline", r"\midrule"]
    for r in rows:
        m, g = r["MWPM"], r["GNN"]
        lines.append(rf"{r['window']} & {m[0]:.1f}--{m[1]:.1f} ({m[2]:.1f}) & {100 * m[3]:.1f}\% & "
                     rf"{g[0]:.1f}--{g[1]:.1f} ({g[2]:.1f}) & {100 * g[3]:.1f}\% \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Goodness of fit of the one-parameter form, Eq.~(4), for each fit window "
              r"(ML slope refitted per window and configuration): range and median over configurations of the "
              r"reduced $\chi^2$ over integer-dB bins, and median failure-weighted mean relative deviation "
              r"$\langle|{\rm data}/{\rm fit}-1|\rangle$, which does not grow with the sample size.}",
              r"\label{window_gof_table}", r"\end{table}"]
    open(os.path.join(mf4.TABLE_DIR, "appD_window_gof.tex"), "w").write("\n".join(lines) + "\n")

    # ---- local slopes (Fig. 8) ----------------------------------------------------------
    local, local_sf = {n: {} for n in names}, {n: {} for n in names}
    for n in names:
        for c in configs:
            ws = [window(binned[c], n, lo, hi) for lo, hi in LOCAL]
            local[n][c] = [fit1(*w) for w in ws]
            local_sf[n][c] = [fisher_sigma(w[0], w[2], a) for w, a in zip(ws, local[n][c])]
    for fname, vals, fmt in (("appD_local_slopes.csv", local, "{:.4f}"), ("appD_local_slopes_sigma.csv", local_sf, "{:.5f}")):
        with open(os.path.join(mf4.TABLE_DIR, fname), "w") as f:
            f.write("decoder,d,r," + ",".join(f"{lo}-{hi}" for lo, hi in LOCAL) + "\n")
            for n in names:
                for c in configs:
                    f.write(f"{n},{c[0]},{c[1]}," + ",".join(fmt.format(v) for v in vals[n][c]) + "\n")
    lines = [r"\begin{table*}", r"\centering", rf"\begin{{tabular}}{{ll|{'c' * len(LOCAL)}}}", r"\toprule",
             r"Decoder & $(d, r)$ & " + " & ".join(f"{lo}--{hi}" for lo, hi in LOCAL) + r" \\", r"\hline",
             r"\midrule"]
    for i, n in enumerate(names):
        for c in configs:
            lines.append(rf"{names[n]} & $({c[0]}, {c[1]})$ & " + " & ".join(f"{v:.3f}" for v in local[n][c]) + r" \\")
        if i == 0:
            lines.append(r"\hline")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Local calibration slopes: ML fit of Eq.~(4) restricted to each confidence window (dB).}",
              r"\label{local_slopes_table}", r"\end{table*}"]
    open(os.path.join(mf4.TABLE_DIR, "appD_local_slopes.tex"), "w").write("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    centers = [(lo + hi) / 2 for lo, hi in LOCAL]
    cmap = plt.get_cmap("viridis")
    for ax, n in zip(axes, names):
        for i, c in enumerate(configs):
            ax.errorbar(centers, local[n][c], yerr=local_sf[n][c], marker="o", markersize=4, capsize=2,
                        elinewidth=0.8, color=cmap(i / max(len(configs) - 1, 1)), label=f"({c[0]}, {c[1]})")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2)
        ax.axvspan(mf4.G_MIN, mf4.G_MAX, color="0.55", alpha=0.12, linewidth=0)
        ax.set_title(names[n], fontsize=13)
        ax.set_xlabel("Confidence window centre (dB)", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].set_ylabel(r"Local slope $\alpha$", fontsize=12)
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(mf4.FIG_DIR, f"appD_local_slopes.{ext}"), dpi=300)
    plt.close(fig)
    for n in names:
        t = trends[n]
        print(f"{names[n]}: Q={t['Q']:.0f}/{t['dof']}, " + (f"b_d={t['b_d']:+.4f}+-{t['se_b_d']:.4f}, "
              f"b_r={t['b_r']:+.4f}+-{t['se_b_r']:.4f}" if "b_d" in t else ""))
    print(f"calibration extras -> {mf4.TABLE_DIR}")


if __name__ == "__main__":
    main()
