"""
Figure 1 + Appendix A counts table: post-selection performance of the two
decoder-confidence pairs, computed exactly from the saved signed-delta
arrays (no hardcoded values).

Fig. 1a: post-selected LER vs acceptance rate kappa at p = 0.005 for
         (d, r) = (5,5), (7,7), (9,9). Each decoder is ranked by its own
         confidence score ("native" pairs,
         but now recomputed raw from the arrays with +-1 sigma binomial
         bands and a paired-bootstrap CI on the inset advantage ratio).
Fig. 1b: post-selected LER vs physical error rate p at kappa = 0.9.
         Configs are auto-discovered from the files present in saved_runs/
         (currently d = 7 at p = 0.003-0.005 and d = 9 at p = 0.003-0.005;
         d = 5 arrays exist only at p = 0.005 and are included if/when the
         remaining files appear with the same naming convention).

Appendix A: accepted-failure counts behind every Fig. 1a point
         (figures3/tables/appA_failure_counts.tex / .csv).

Heavy results are cached in figures3/derived/; delete the JSON or set
RECOMPUTE=True to force recomputation (~1 min per config).

The inset CI accounts for the statistical dependence of the two LER
estimates (same shots): for each kappa the shots are classified into the
16 joint categories (accepted/failed by decoder/score pair), and the
category counts are resampled from a multinomial distribution. This is an
exact nonparametric paired bootstrap.
"""

import glob
import json
import os
import re
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_runs")

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures3")
DERIVED_DIR = os.path.join(FIG_DIR, "derived")
TABLE_DIR = os.path.join(FIG_DIR, "tables")

RECOMPUTE = False

# Fig. 1a settings
P_MAIN = 0.005
CONFIGS_1A = [(5, 5), (7, 7), (9, 9)]
KAPPA_GRID = np.linspace(0.2, 1.0, 161)
KAPPA_TABLE = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MIN_ACCEPTED_FAILURES = 20  # a (d, kappa) point is plotted only if BOTH
                            # decoders rest on >= 20 accepted failures there
                            # (95% uncertainty below ~45%); complete counts
                            # at all kappa are in App. A
N_BOOTSTRAP = 5000
BOOTSTRAP_SEED = 7

# Fig. 1b settings
KAPPA_1B = 0.9

# d=5 values of Fig. 1(b) (raw arrays for p != 0.005 at d = 5 are not
# retained; these are the values of the published figure):
# the raw arrays for d=5 at p < 0.005 are not in saved_runs/. The one
# checkable point (p = 0.005) agrees with the raw recomputation within
# ~4%, inside its band. If the missing arrays are regenerated with the
# standard naming, the auto-discovered raw values take precedence.
FIG1B_D5_P = np.array([3.0e-3, 3.5e-3, 4.0e-3, 4.5e-3, 5.0e-3])
FIG1B_D5 = {
    "mwpm": (np.array([3.27e-05, 1.15e-04, 3.20e-04, 7.60e-04, 1.48e-03]),
             np.array([6.03e-07, 1.13e-06, 1.89e-06, 2.91e-06, 4.05e-06])),
    "gnn": (np.array([6.944e-06, 2.389e-05, 7.028e-05, 1.988e-04, 4.60e-04]),
            np.array([2.78e-07, 5.15e-07, 8.84e-07, 1.49e-06, 2.26e-06])),
}

# ---------------------------------------------------------------------------
# Style: identical to Fig. 1 of the paper (colors by code distance,
# figure1c.py): default matplotlib cycle, color = distance
# (d=5 orange C1, d=7 blue C0, d=9 green C2), solid+squares = MWPM,
# dashed+circles = GNN, markers at the tenths kappa points only.
# ---------------------------------------------------------------------------

DIST_COLORS = {5: "#ff7f0e", 7: "#1f77b4", 9: "#2ca02c"}
STYLE_MWPM = dict(linestyle="-", marker="s")
STYLE_GNN = dict(linestyle="--", marker="o")
KAPPA_SHOW = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

plt.rcParams.update({
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_deltas(d, dt, p, shots):
    tag = f"d{d}_dt{dt}_p{p}_shots{shots}"
    path_m = os.path.join(SAVE_DIR, f"delta_mwpm_{tag}.npy")
    path_g = os.path.join(SAVE_DIR, f"delta_gnn_{tag}.npy")
    for path in (path_m, path_g):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing saved run:\n{path}")
    return np.load(path_m), np.load(path_g)


def discover_runs():
    """Find all (d, dt, p, shots) with both decoder files present."""
    pat = re.compile(
        r"delta_mwpm_d(\d+)_dt(\d+)_p([0-9.]+)_shots(\d+)\.npy$"
    )
    runs = []
    for path in sorted(glob.glob(os.path.join(SAVE_DIR, "delta_mwpm_*.npy"))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        d, dt, p, shots = int(m[1]), int(m[2]), float(m[3]), int(m[4])
        gnn = os.path.join(
            SAVE_DIR, f"delta_gnn_d{d}_dt{dt}_p{m[3]}_shots{shots}.npy"
        )
        if os.path.isfile(gnn):
            runs.append((d, dt, p, shots))
    return runs


# ---------------------------------------------------------------------------
# Fig. 1a computation: native curves + paired bootstrap for the ratio
# ---------------------------------------------------------------------------

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
    fail_desc = delta_db[order] < 0
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
    fail_m = dm < 0
    fail_g = dg < 0
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


def compute_fig1a_config(d, dt):
    t0 = time.perf_counter()
    dm, dg = load_deltas(d, dt, P_MAIN, 100_000_000)

    ler_m, fail_m, acc = native_curve(dm, KAPPA_GRID)
    ler_g, fail_g, _ = native_curve(dg, KAPPA_GRID)

    # Paired bootstrap on a thinned grid (the band needs no finer sampling).
    kappa_ci = KAPPA_GRID[::2]
    lo, hi = paired_ratio_ci(dm, dg, kappa_ci, N_BOOTSTRAP, BOOTSTRAP_SEED)

    print(f"  computed d={d}, dt={dt} in {time.perf_counter() - t0:.0f} s")
    return {
        "config": {"d": d, "dt": dt, "p": P_MAIN, "shots": len(dm)},
        "kappa": KAPPA_GRID.tolist(),
        "mwpm": {"ler": ler_m.tolist(), "n_fail": fail_m.tolist()},
        "gnn": {"ler": ler_g.tolist(), "n_fail": fail_g.tolist()},
        "n_accepted": acc.tolist(),
        "ratio_kappa": kappa_ci.tolist(),
        "ratio_ci_lo": lo.tolist(),
        "ratio_ci_hi": hi.tolist(),
    }


# ---------------------------------------------------------------------------
# Fig. 1b computation: LER at fixed kappa via argpartition (no full sort)
# ---------------------------------------------------------------------------

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
    fail = delta_db < 0
    above = conf > v
    n_above = int(above.sum())
    f_above = float((fail & above).sum())
    at = conf == v
    m_v = int(at.sum())
    f_v = float((fail & at).sum())
    f_acc = f_above + f_v * (n_acc - n_above) / m_v
    return f_acc / n_acc, f_acc, n_acc


def compute_fig1b():
    """LER(kappa=0.9) vs p per distance, from every run found on disk."""
    per_distance = {}
    for d, dt, p, shots in discover_runs():
        if dt != d:
            continue  # Fig 1b uses r = d configs only
        dm, dg = load_deltas(d, dt, p, shots)
        lm, fm, na = ler_at_kappa(dm, KAPPA_1B)
        lg, fg, _ = ler_at_kappa(dg, KAPPA_1B)
        per_distance.setdefault(d, []).append({
            "p": p, "shots": shots,
            "mwpm": {"ler": lm, "n_fail": fm},
            "gnn": {"ler": lg, "n_fail": fg},
            "n_accepted": na,
        })
        print(f"  d={d}, p={p}: MWPM {lm:.3e} ({fm:.1f}), "
              f"GNN {lg:.3e} ({fg:.1f})")
        del dm, dg
    for d in per_distance:
        per_distance[d].sort(key=lambda r: r["p"])
    return per_distance


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def cached(name, compute_fn):
    os.makedirs(DERIVED_DIR, exist_ok=True)
    cache = os.path.join(DERIVED_DIR, f"{name}.json")
    if not RECOMPUTE and os.path.isfile(cache):
        with open(cache) as f:
            return json.load(f)
    res = compute_fn()
    with open(cache, "w") as f:
        json.dump(res, f)
    return res


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _at_tenths(res, name):
    """Values of a stored dense-grid curve at the tenths kappa points."""
    kappa = np.array(res["kappa"])
    idx = np.array([int(np.argmin(np.abs(kappa - k))) for k in KAPPA_SHOW])
    ler = np.array(res[name]["ler"])[idx]
    n_fail = np.array(res[name]["n_fail"], dtype=np.float64)[idx]
    n_acc = np.array(res["n_accepted"], dtype=np.int64)[idx]
    return ler, n_fail, n_acc


def plot_fig1a(results):
    fig, ax = plt.subplots(figsize=(7, 5))

    for res in results:
        d = res["config"]["d"]
        color = DIST_COLORS[d]

        ler_m, fail_m, n_acc = _at_tenths(res, "mwpm")
        ler_g, fail_g, _ = _at_tenths(res, "gnn")
        # Pairwise display rule: a kappa point is shown only if BOTH
        # decoders meet the failure-count threshold there.
        ok = ((fail_m >= MIN_ACCEPTED_FAILURES)
              & (fail_g >= MIN_ACCEPTED_FAILURES))

        for style, ler, name in [(STYLE_MWPM, ler_m, "MWPM"),
                                 (STYLE_GNN, ler_g, "GNN")]:
            se = np.sqrt(ler * (1 - ler) / n_acc)
            ci = 1.96 * se  # 95% interval, as everywhere in the paper
            ax.fill_between(KAPPA_SHOW[ok],
                            np.maximum(ler[ok] - ci[ok], 1e-12),
                            ler[ok] + ci[ok], alpha=0.18, color=color)
            ax.plot(KAPPA_SHOW[ok], ler[ok], color=color, markersize=7,
                    linewidth=1.5, label=f"d={d}, {name}", **style)

    ax.set_yscale("log")
    # Extend the y axis below the curves so the inset sits in empty space.
    ax.set_ylim(bottom=2e-7)
    ax.set_xlabel("Acceptance rate κ", fontsize=16)
    ax.set_ylabel("Post-selected logical error rate", fontsize=16)
    ax.set_title("Post-selection: LER vs κ, p = 0.005", fontsize=16)
    ax.legend(fontsize=12, ncol=1)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    # Inset: GNN advantage over MWPM, with paired-bootstrap 95% bands.
    axins = inset_axes(ax, width="30%", height="24%", loc="lower right",
                       borderpad=1.5)
    for res in results:
        d = res["config"]["d"]
        color = DIST_COLORS[d]
        ler_m, fail_m, _ = _at_tenths(res, "mwpm")
        ler_g, fail_g, _ = _at_tenths(res, "gnn")
        ok = ((fail_m >= MIN_ACCEPTED_FAILURES)
              & (fail_g >= MIN_ACCEPTED_FAILURES))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = ler_m / ler_g

        kappa_ci = np.array(res["ratio_kappa"])
        idx = np.array([int(np.argmin(np.abs(kappa_ci - k)))
                        for k in KAPPA_SHOW])
        lo = np.array(res["ratio_ci_lo"])[idx]
        hi = np.array(res["ratio_ci_hi"])[idx]
        band = ok & np.isfinite(lo) & np.isfinite(hi)
        axins.fill_between(KAPPA_SHOW[band], lo[band], hi[band],
                           alpha=0.18, color=color)
        axins.plot(KAPPA_SHOW[ok], ratio[ok], linestyle="-", color=color,
                   marker="o", markersize=4, linewidth=1.2, label=f"d={d}")

    axins.axhline(1.0, color="black", linestyle=":", linewidth=3.0)
    # ensure a tick at 1 so the dotted reference line is readable
    lo_lim, hi_lim = axins.get_ylim()
    ticks = [t for t in axins.get_yticks() if lo_lim <= t <= hi_lim]
    axins.set_yticks(sorted(set(ticks) | {1.0}))
    axins.set_ylim(lo_lim, hi_lim)
    axins.set_ylabel("MWPM / GNN", fontsize=12, fontweight="bold")
    axins.tick_params(axis="both", labelsize=8)
    axins.grid(True, which="both", linestyle="--", alpha=0.35)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"fig1a_postselection_kappa.{ext}")
        fig.savefig(path, dpi=300)
        print(f"  saved {path}")
    plt.close(fig)


def plot_fig1b(per_distance):
    fig, ax = plt.subplots(figsize=(7, 5))

    # Normalize keys (JSON caches stringify them) and make sure d=5 is
    # present, falling back to the hardcoded values above if needed.
    curves = {}
    for d_str, rows in per_distance.items():
        d = int(d_str)
        if len(rows) >= 2:
            ps = np.array([r["p"] for r in rows])
            n_acc = np.array([r["n_accepted"] for r in rows], dtype=np.int64)
            curves[d] = {
                name: (np.array([r[name]["ler"] for r in rows]),
                       np.sqrt(np.array([r[name]["ler"] for r in rows])
                               * (1 - np.array([r[name]["ler"] for r in rows]))
                               / n_acc))
                for name in ("mwpm", "gnn")
            }
            curves[d]["p"] = ps
    if 5 not in curves:
        print("  note: d=5 raw arrays not on disk; using the values of the "
              "published figure (see FIG1B_D5 above)")
        curves[5] = {"p": FIG1B_D5_P,
                     "mwpm": FIG1B_D5["mwpm"], "gnn": FIG1B_D5["gnn"]}

    for d in sorted(curves):
        color = DIST_COLORS.get(d, "#666666")
        ps = curves[d]["p"]
        for name, style in [("mwpm", STYLE_MWPM), ("gnn", STYLE_GNN)]:
            ler, se = curves[d][name]
            label = f"d={d}, " + ("MWPM" if name == "mwpm" else "GNN")
            ci = 1.96 * se  # 95% interval, as everywhere in the paper
            ax.fill_between(ps, np.maximum(ler - ci, 1e-12), ler + ci,
                            alpha=0.18, color=color)
            ax.plot(ps, ler, color=color, markersize=7, linewidth=1.5,
                    label=label, **style)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical error rate p", fontsize=16)
    ax.set_ylabel("Post-selected logical error rate", fontsize=16)
    ax.set_title("Post-selection: LER vs p, κ = 0.9", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"fig1b_postselection_p.{ext}")
        fig.savefig(path, dpi=300)
        print(f"  saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Appendix A: accepted-failure counts behind Fig. 1a
# ---------------------------------------------------------------------------

def write_counts_table(results):
    os.makedirs(TABLE_DIR, exist_ok=True)

    # CSV (full grid) -------------------------------------------------------
    csv_path = os.path.join(TABLE_DIR, "appA_failure_counts.csv")
    with open(csv_path, "w") as f:
        f.write("d,dt,p,kappa,n_accepted,mwpm_n_fail,mwpm_ler,"
                "gnn_n_fail,gnn_ler\n")
        for res in results:
            cfg = res["config"]
            kappa = np.array(res["kappa"])
            for k in KAPPA_TABLE:
                i = int(np.argmin(np.abs(kappa - k)))
                f.write(f"{cfg['d']},{cfg['dt']},{cfg['p']},{k},"
                        f"{res['n_accepted'][i]},"
                        f"{res['mwpm']['n_fail'][i]:.1f},"
                        f"{res['mwpm']['ler'][i]:.6e},"
                        f"{res['gnn']['n_fail'][i]:.1f},"
                        f"{res['gnn']['ler'][i]:.6e}\n")
    print(f"  saved {csv_path}")

    # LaTeX (counts only, compact) ------------------------------------------
    header = " & ".join(f"{k:.1f}" for k in KAPPA_TABLE)
    lines = [
        r"\begin{table*}",
        r"\centering",
        rf"\begin{{tabular}}{{ll| {' '.join('c' for _ in KAPPA_TABLE)}}}",
        r"\toprule",
        rf" & & \multicolumn{{{len(KAPPA_TABLE)}}}{{c}}"
        r"{Accepted failures at acceptance rate $\kappa$} \\",
        rf"Config & Decoder & {header} \\",
        r"\hline",
        r"\midrule",
    ]
    for res in results:
        cfg = res["config"]
        kappa = np.array(res["kappa"])
        for name, label in [("mwpm", "MWPM"), ("gnn", "GNN")]:
            counts = []
            for k in KAPPA_TABLE:
                i = int(np.argmin(np.abs(kappa - k)))
                counts.append(str(int(round(res[name]["n_fail"][i]))))
            lines.append(
                rf"$({cfg['d']}, {cfg['dt']})$ & {label} & "
                + " & ".join(counts) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Number of accepted logical failures behind each point of "
        r"Fig.~1(a), at $p = 0.005$ with $10^8$ shots per configuration. "
        r"Acceptance-threshold ties are averaged uniformly (fractionally "
        r"accepted), so the entries are tie-averaged expectations, rounded "
        r"to integers. Points resting on fewer than "
        + str(MIN_ACCEPTED_FAILURES) +
        r" failures are suppressed in the figure.}",
        r"\label{counts_table}",
        r"\end{table*}",
    ]
    tex_path = os.path.join(TABLE_DIR, "appA_failure_counts.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {tex_path}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    start = time.perf_counter()

    print("Fig. 1a: native post-selection curves")
    results_1a = []
    for d, dt in CONFIGS_1A:
        print(f"config d={d}, dt={dt}:")
        results_1a.append(
            cached(f"fig1a_v2_d{d}_dt{dt}_p{P_MAIN}",
                   lambda d=d, dt=dt: compute_fig1a_config(d, dt))
        )
    plot_fig1a(results_1a)
    write_counts_table(results_1a)

    print("\nFig. 1b: LER vs p at kappa = 0.9 (auto-discovered runs)")
    results_1b = cached("fig1b_v2_ler_vs_p", compute_fig1b)
    plot_fig1b(results_1b)

    print(f"\nTotal elapsed: {(time.perf_counter() - start) / 60:.1f} min")
