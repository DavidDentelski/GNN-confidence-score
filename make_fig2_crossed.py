"""
Figure 2 + Table I: crossed post-selection analysis.

Separates hard-decoder quality from confidence-score quality by exchanging
the ranking scores between the two decoders. For each config (d, dt) at
p = 0.005, four exact post-selection curves are computed from the saved
shot-aligned signed-delta arrays:

    decoder MWPM, ranked by |g_MWPM|   (native)
    decoder MWPM, ranked by |g_GNN|    (crossed)
    decoder GNN,  ranked by |g_GNN|    (native)
    decoder GNN,  ranked by |g_MWPM|   (crossed)

plus the tie-aware AUC of each (score, decoder) pair:

    AUC = P(conf_success > conf_fail) + 0.5 P(tie),

a monotone-invariant measure of how well a score ranks that decoder's
failures. Everything is recomputed from saved_runs on first press and
cached as JSON in figures3/derived/ (delete the JSON or set RECOMPUTE=True
to force a re-run; a full recompute takes ~2 min per config).

Outputs:
    figures3/fig2a_crossed_d9.pdf/.png      main-text Fig. 2(a), (9, 9)
    figures3/fig2b_crossed_d5.pdf/.png      main-text Fig. 2(b), (5, 5)
    figures3/appB_crossed_d7_dt7.*          Appendix B, (7, 7)
    figures3/tables/table1_auc.tex          AUC table, all configs
"""

import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt

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
# Main-text Fig. 2: (a) = (9, 9) shows the native pairings winning, (b) =
# (5, 5) shows the reversal (gap saturation). Only (7, 7) goes to Appendix B.
MAIN_PANELS = {(9, 9): "fig2a_crossed_d9", (5, 5): "fig2b_crossed_d5"}

RECOMPUTE = False          # True: ignore cached JSON and recompute everything
KAPPA_GRID = np.linspace(0.2, 1.0, 161)
KAPPA_ANCHORS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
MIN_ACCEPTED_FAILURES = 20  # a kappa point is shown only if all curves in
                            # the panel rest on >= 20 accepted failures
                            # (95% uncertainty below ~45%), as in Fig. 1

# ---------------------------------------------------------------------------
# Style: color = hard decoder (blue = MWPM, red = GNN, used for the two
# decoders in every figure of the paper), linestyle = pairing (solid =
# decoder ranked by its own score, dashed = ranked by the other decoder's
# score), marker = hard decoder. Markers at the tenths kappa points only, as in
# Fig. 1.
# ---------------------------------------------------------------------------

COLOR_MWPM = "#1f77b4"   # blue
COLOR_GNN = "#e60000"    # red

KAPPA_SHOW = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

plt.rcParams.update({
    "pdf.fonttype": 42,       # embed editable TrueType text in the PDF
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Computation (from the saved shot-aligned arrays)
# ---------------------------------------------------------------------------

def load_deltas(d, dt):
    tag = f"d{d}_dt{dt}_p{P}_shots{SHOTS}"
    path_m = os.path.join(SAVE_DIR, f"delta_mwpm_{tag}.npy")
    path_g = os.path.join(SAVE_DIR, f"delta_gnn_{tag}.npy")
    for path in (path_m, path_g):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing saved run:\n{path}")
    dm = np.load(path_m)
    dg = np.load(path_g)
    if dm.shape != dg.shape:
        raise ValueError(f"Shape mismatch: {dm.shape} vs {dg.shape}")
    return dm, dg


def midranks_ascending(conf):
    """Tie-averaged 1-based ranks. Returns (ascending order, ranks in sorted order)."""
    order = np.argsort(conf, kind="stable")
    sc = conf[order]
    new = np.empty(len(sc), dtype=bool)
    new[0] = True
    np.not_equal(sc[1:], sc[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    counts = np.diff(np.append(starts, len(sc)))
    mid_of_group = starts + (counts + 1) / 2.0
    gid = np.cumsum(new) - 1
    ranks_sorted = mid_of_group[gid]
    return order, ranks_sorted


def auc_from_ranks(ranks_sorted, fail_sorted):
    """AUC = P(conf_success > conf_fail) + 0.5 P(tie)."""
    n = len(ranks_sorted)
    n_f = int(fail_sorted.sum())
    n_s = n - n_f
    r_fail = float(ranks_sorted[fail_sorted].sum())
    u_fail = r_fail - n_f * (n_f + 1) / 2.0
    return 1.0 - u_fail / (n_f * n_s)


def fractional_curve(conf_desc, fail_desc, kappas):
    """
    Exact post-selected LER at each kappa with FRACTIONAL TIE HANDLING:
    when the acceptance threshold falls inside a block of shots sharing
    one confidence value, the block's failures are accepted proportionally
    to the accepted fraction of the block. This is equivalent to averaging
    over random orderings of tied shots, and matters where tie blocks are
    macroscopic (the saturated maximal gap at small d). Failure counts are
    therefore returned as floats.
    """
    n = len(conf_desc)
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
            # Acceptance boundary coincides with a tie-block boundary.
            f_acc = float(cum[n_acc - 1])
        else:
            f_before = float(cum[s - 1]) if s > 0 else 0.0
            f_block = float(cum[e - 1]) - f_before
            f_acc = f_before + f_block * (n_acc - s) / (e - s)
        lers.append(f_acc / n_acc)
        fails.append(f_acc)
        accepted.append(n_acc)
    return lers, fails, accepted


def tie_block_ids(conf, order):
    """
    Per-shot tie-block index in DESCENDING confidence order (block 0 = the
    most confident block). Used for the tie-averaged paired bootstrap.
    Returns (block_of_shot [int32], block_start [int64], block_end [int64],
    conf sorted descending).
    """
    n = len(conf)
    conf_desc = conf[order][::-1].copy()
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(conf_desc[1:], conf_desc[:-1], out=new[1:])
    block_start = np.flatnonzero(new)
    block_end = np.append(block_start[1:], n)
    bid_desc = (np.cumsum(new) - 1).astype(np.int32)
    block_of_shot = np.empty(n, dtype=np.int32)
    block_of_shot[order[::-1]] = bid_desc
    return block_of_shot, block_start, block_end, conf_desc


def paired_crossed_cis(fail, conf, n, kappas, n_boot=3000, seed=11):
    """
    Paired 95% CIs for the crossed-vs-native LER ratio at fixed decoder,
    at the anchor kappas, via a multinomial bootstrap over joint per-shot
    categories. The tie-averaging rule is reproduced inside every
    replicate: shots are classified per score as strictly-above /
    inside / strictly-below the threshold tie block, and the accepted
    fraction of the threshold block is recomputed from each replicate's
    category counts. Intervals therefore refer to the tie-averaged
    estimand.
    """
    rng = np.random.default_rng(seed)
    blocks = {}
    for s_name, c in conf.items():
        order = np.argsort(c, kind="stable")
        blocks[s_name] = tie_block_ids(c, order)
        del order

    out = {}
    for k in kappas:
        n_acc = int(np.floor(k * n))
        zones = {}
        block_sizes = {}
        for s_name in conf:
            b_of_shot, b_start, b_end, _ = blocks[s_name]
            tb = int(np.searchsorted(b_start, n_acc, side="right")) - 1
            if n_acc >= b_end[tb]:
                tb += 1  # boundary coincides with block edge; block tb is fully out
            # zone: 0 = strictly above threshold block, 1 = inside, 2 = below
            zones[s_name] = np.clip(b_of_shot - tb, -1, 1).astype(np.int8) + 1
            s = int(b_start[tb]) if tb < len(b_start) else n
            e = int(b_end[tb]) if tb < len(b_start) else n
            block_sizes[s_name] = (s, e)

        # Joint categories: fail_m(2) x fail_g(2) x zone_M(3) x zone_G(3) = 36
        cat = (fail["mwpm"].astype(np.int8) * 18
               + fail["gnn"].astype(np.int8) * 9
               + zones["g_mwpm"] * 3
               + zones["g_gnn"])
        counts = np.bincount(cat, minlength=36).astype(np.float64)
        draws = rng.multinomial(n, counts / n, size=n_boot).astype(np.float64)

        idx = np.arange(36)
        fm = (idx // 18) == 1
        fg = ((idx // 9) % 2) == 1
        zm = (idx // 3) % 3
        zg = idx % 3

        def accepted_fails(dr, fail_mask, zone):
            """Tie-averaged accepted failures for (decoder, ranking score)."""
            above = fail_mask & (zone == 0)
            inblk = fail_mask & (zone == 1)
            n_above = dr[:, zone == 0].sum(axis=1)
            m_blk = dr[:, zone == 1].sum(axis=1)
            frac = np.clip((n_acc - n_above) / np.maximum(m_blk, 1), 0.0, 1.0)
            return dr[:, above].sum(axis=1) + frac * dr[:, inblk].sum(axis=1)

        res_k = {}
        for dec_name, f_mask in [("mwpm", fm), ("gnn", fg)]:
            native = "g_mwpm" if dec_name == "mwpm" else "g_gnn"
            z_nat = zm if native == "g_mwpm" else zg
            z_cro = zg if native == "g_mwpm" else zm
            f_nat = accepted_fails(draws, f_mask, z_nat)
            f_cro = accepted_fails(draws, f_mask, z_cro)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = f_cro / f_nat
            ratios = ratios[np.isfinite(ratios)]
            if len(ratios) < n_boot // 2:
                res_k[dec_name] = None
            else:
                res_k[dec_name] = {
                    "ratio_lo": float(np.percentile(ratios, 2.5)),
                    "ratio_hi": float(np.percentile(ratios, 97.5)),
                }
        out[f"{k:g}"] = res_k
        del cat
    return out


def compute_config(d, dt):
    """All crossed quantities for one config. ~3 min for 1e8 shots."""
    t0 = time.perf_counter()
    dm, dg = load_deltas(d, dt)
    n = len(dm)

    fail = {"mwpm": dm < 0, "gnn": dg < 0}
    conf = {"g_mwpm": np.abs(dm), "g_gnn": np.abs(dg)}
    del dm, dg

    out = {
        "config": {"d": d, "dt": dt, "p": P, "shots": n},
        "n_fail": {k: int(v.sum()) for k, v in fail.items()},
        "both_fail": int((fail["mwpm"] & fail["gnn"]).sum()),
        "kappa": KAPPA_GRID.tolist(),
        "curves": {},
        "auc": {},
    }

    for score_name, c in conf.items():
        order, ranks_sorted = midranks_ascending(c)
        conf_desc = c[order][::-1].copy()
        for dec_name, f in fail.items():
            f_sorted_asc = f[order]
            out["auc"][f"{score_name}->{dec_name}"] = auc_from_ranks(
                ranks_sorted, f_sorted_asc
            )
            lers, fails, accepted = fractional_curve(
                conf_desc, f_sorted_asc[::-1], KAPPA_GRID
            )
            out["curves"][f"decoder={dec_name},rank={score_name}"] = {
                "ler": lers,
                "n_fail_accepted": fails,
                "n_accepted": accepted,
            }
            del f_sorted_asc
        del order, ranks_sorted, conf_desc

    out["paired_ratio_ci"] = paired_crossed_cis(
        fail, conf, n, KAPPA_ANCHORS
    )

    print(f"  computed d={d}, dt={dt} in {time.perf_counter() - t0:.0f} s")
    return out


def get_config_results(d, dt):
    """Load from the JSON cache, or compute and cache."""
    os.makedirs(DERIVED_DIR, exist_ok=True)
    cache = os.path.join(DERIVED_DIR, f"crossed_v2_d{d}_dt{dt}_p{P}.json")
    if not RECOMPUTE and os.path.isfile(cache):
        with open(cache) as f:
            return json.load(f)
    res = compute_config(d, dt)
    with open(cache, "w") as f:
        json.dump(res, f)
    return res


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

CURVE_STYLE = {
    # (decoder, ranking score): color/linestyle/marker/label
    "decoder=mwpm,rank=g_mwpm": dict(
        color=COLOR_MWPM, linestyle="-", marker="s",
        label=r"MWPM, ranked by $g_{\rm MWPM}$"),
    "decoder=mwpm,rank=g_gnn": dict(
        color=COLOR_MWPM, linestyle="--", marker="s",
        label=r"MWPM, ranked by $g_{\rm GNN}$"),
    "decoder=gnn,rank=g_gnn": dict(
        color=COLOR_GNN, linestyle="-", marker="o",
        label=r"GNN, ranked by $g_{\rm GNN}$"),
    "decoder=gnn,rank=g_mwpm": dict(
        color=COLOR_GNN, linestyle="--", marker="o",
        label=r"GNN, ranked by $g_{\rm MWPM}$"),
}


def plot_crossed(res, save_stem):
    d, dt = res["config"]["d"], res["config"]["dt"]
    kappa = np.array(res["kappa"])
    idx = np.array([int(np.argmin(np.abs(kappa - k))) for k in KAPPA_SHOW])

    # Pairwise display rule across the panel: a kappa point is shown only
    # if ALL FOUR decoder-score combinations rest on enough failures there.
    ok = np.ones(len(KAPPA_SHOW), dtype=bool)
    for key in CURVE_STYLE:
        n_fail = np.array(res["curves"][key]["n_fail_accepted"],
                          dtype=np.float64)[idx]
        ok &= n_fail >= MIN_ACCEPTED_FAILURES

    fig, ax = plt.subplots(figsize=(7, 5))

    for key, style in CURVE_STYLE.items():
        c = res["curves"][key]
        ler = np.array(c["ler"], dtype=np.float64)[idx]
        n_acc = np.array(c["n_accepted"], dtype=np.int64)[idx]
        se = np.sqrt(ler * (1 - ler) / n_acc)
        ci = 1.96 * se  # 95% interval, as everywhere in the paper

        ax.fill_between(KAPPA_SHOW[ok],
                        np.maximum(ler[ok] - ci[ok], 1e-12),
                        ler[ok] + ci[ok], alpha=0.18, color=style["color"])
        ax.plot(KAPPA_SHOW[ok], ler[ok], color=style["color"],
                linestyle=style["linestyle"], marker=style["marker"],
                markersize=7, linewidth=1.5, label=style["label"])

    ax.set_yscale("log")
    ax.set_xlabel("Acceptance rate κ", fontsize=16)
    ax.set_ylabel("Post-selected logical error rate", fontsize=16)
    ax.set_title(f"Exchanged confidence scores: d = r = {d}, p = {P}",
                 fontsize=16)
    ax.legend(fontsize=12, ncol=1)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = f"{save_stem}.{ext}"
        fig.savefig(path, dpi=300)
        print(f"  saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Table I: AUC of each (ranking score, decoder) pair
# ---------------------------------------------------------------------------

def write_auc_table(all_results):
    os.makedirs(TABLE_DIR, exist_ok=True)
    cols = " ".join("c" for _ in CONFIGS)
    header = " & ".join(rf"$({d}, {dt})$" for d, dt in CONFIGS)

    def pair_rows(decoder, score_self, score_other):
        # Bold the better of the two scores per column, within each decoder.
        v_self = [all_results[(d, dt)]["auc"][f"{score_self}->{decoder}"]
                  for d, dt in CONFIGS]
        v_other = [all_results[(d, dt)]["auc"][f"{score_other}->{decoder}"]
                   for d, dt in CONFIGS]
        fmt = lambda v, w: rf"\textbf{{{v:.4f}}}" if v > w else f"{v:.4f}"
        row_self = " & ".join(fmt(a, b) for a, b in zip(v_self, v_other))
        row_other = " & ".join(fmt(b, a) for a, b in zip(v_self, v_other))
        return row_self, row_other

    mwpm_self, mwpm_other = pair_rows("mwpm", "g_mwpm", "g_gnn")
    gnn_self, gnn_other = pair_rows("gnn", "g_gnn", "g_mwpm")

    lines = [
        r"\begin{table*}",  # 10 columns: span both columns of the two-column layout
        r"\centering",
        rf"\begin{{tabular}}{{ll| {cols}}}",
        r"\toprule",
        rf"Decoder & Ranked by & {header} \\",
        r"\hline",
        r"\midrule",
        rf"MWPM & $g_{{\rm MWPM}}$ & {mwpm_self} \\",
        rf"MWPM & $g_{{\rm GNN}}$  & {mwpm_other} \\",
        r"\hline",
        rf"GNN  & $g_{{\rm GNN}}$  & {gnn_self} \\",
        rf"GNN  & $g_{{\rm MWPM}}$ & {gnn_other} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Tie-aware AUC of each confidence score for ranking each "
        r"decoder's logical failures at $p = 0.005$, across code distances $d$ "
        r"and stabilizer rounds $r$ ($10^8$ shots per configuration; ties "
        r"receive half credit). For each decoder, bold marks the "
        r"better-ranking score; paired DeLong standard errors on same-decoder "
        r"AUC differences are below $10^{-4}$, so all visible differences are "
        r"statistically resolved. For $d \geq 7$ each decoder is ranked best "
        r"by its own score; at $d = 5$ the GNN logit ranks both decoders' "
        r"failures best.}",
        r"\label{auc_table}",
        r"\end{table*}",
    ]
    path = os.path.join(TABLE_DIR, "table1_auc.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {path}")


def write_crossed_counts_table(all_results):
    """
    Accepted-failure counts behind the exchanged-ranking curves of Fig. 2
    and Appendix B (the native pairings are in the Fig. 1 counts table).
    """
    os.makedirs(TABLE_DIR, exist_ok=True)
    header = " & ".join(f"{k:.1f}" for k in KAPPA_SHOW)
    lines = [
        r"\begin{table*}",
        r"\centering",
        rf"\begin{{tabular}}{{ll| {' '.join('c' for _ in KAPPA_SHOW)}}}",
        r"\toprule",
        rf" & & \multicolumn{{{len(KAPPA_SHOW)}}}{{c}}"
        r"{Accepted failures at acceptance rate $\kappa$} \\",
        rf"Config & Curve & {header} \\",
        r"\hline",
        r"\midrule",
    ]
    curve_label = {
        "decoder=mwpm,rank=g_gnn": r"MWPM, ranked by $g_{\rm GNN}$",
        "decoder=gnn,rank=g_mwpm": r"GNN, ranked by $g_{\rm MWPM}$",
    }
    for d, dt in [(5, 5), (7, 7), (9, 9)]:
        res = all_results[(d, dt)]
        kappa = np.array(res["kappa"])
        idx = [int(np.argmin(np.abs(kappa - k))) for k in KAPPA_SHOW]
        for key, lab in curve_label.items():
            nf = np.array(res["curves"][key]["n_fail_accepted"])[idx]
            row = " & ".join(f"{v:.0f}" for v in nf)
            lines.append(rf"$({d}, {dt})$ & {lab} & {row} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Number of accepted logical failures behind the "
        r"exchanged-ranking curves of Fig.~2 and Appendix~B, at $p = 0.005$ "
        r"with $10^8$ shots per configuration. Acceptance-threshold ties are "
        r"averaged uniformly (fractionally accepted), so the entries are "
        r"tie-averaged expectations, rounded to integers. The native "
        r"pairings are tabulated in Table~\ref{counts_table}.}",
        r"\label{crossed_counts_table}",
        r"\end{table*}",
    ]
    path = os.path.join(TABLE_DIR, "appA_crossed_failure_counts.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    start = time.perf_counter()

    all_results = {}
    for d, dt in CONFIGS:
        print(f"config d={d}, dt={dt}:")
        all_results[(d, dt)] = get_config_results(d, dt)

    # Figures: (9, 9) and (5, 5) in the main text (Fig. 2a/2b); (7, 7) in
    # Appendix A. (All configs are still computed above — they enter Table I.)
    for d, dt in [(5, 5), (7, 7), (9, 9)]:
        res = all_results[(d, dt)]
        if (d, dt) in MAIN_PANELS:
            stem = os.path.join(FIG_DIR, MAIN_PANELS[(d, dt)])
        else:
            stem = os.path.join(FIG_DIR, f"appB_crossed_d{d}_dt{dt}")
        plot_crossed(res, stem)

    write_auc_table(all_results)
    write_crossed_counts_table(all_results)

    # Console summary: AUC and the kappa=0.9 crossed comparison.
    print("\nAUC summary (score -> decoder failures; ties get half credit):")
    for d, dt in CONFIGS:
        a = all_results[(d, dt)]["auc"]
        print(f"  (d={d}, r={dt}):  "
              f"gM->M={a['g_mwpm->mwpm']:.4f}  gG->M={a['g_gnn->mwpm']:.4f}  |  "
              f"gG->G={a['g_gnn->gnn']:.4f}  gM->G={a['g_mwpm->gnn']:.4f}")

    print("\nDecomposition at kappa = 0.9 (ratios MWPM/GNN, > 1 favors GNN):")
    for d, dt in CONFIGS:
        res = all_results[(d, dt)]
        kappa = np.array(res["kappa"])
        i09 = int(np.argmin(np.abs(kappa - 0.9)))
        i10 = int(np.argmin(np.abs(kappa - 1.0)))
        ler_m = np.array(res["curves"]["decoder=mwpm,rank=g_mwpm"]["ler"])
        ler_g = np.array(res["curves"]["decoder=gnn,rank=g_gnn"]["ler"])
        package = ler_m[i09] / ler_g[i09]
        hard = ler_m[i10] / ler_g[i10]
        selection = package / hard
        print(f"  (d={d}, r={dt}):  package {package:.2f} = "
              f"hard {hard:.2f} x selection {selection:.2f}")

    print("\nCrossed/native LER ratio at kappa = 0.9 (paired 95% CI, "
          "tie-averaged):")
    for d, dt in CONFIGS:
        pc = all_results[(d, dt)].get("paired_ratio_ci", {}).get("0.9", {})
        parts = []
        for dec in ("mwpm", "gnn"):
            r = pc.get(dec)
            if r:
                parts.append(f"{dec}: [{r['ratio_lo']:.3f}, {r['ratio_hi']:.3f}]")
        if parts:
            print(f"  (d={d}, r={dt}):  " + "  ".join(parts))

    print(f"\nTotal elapsed: {(time.perf_counter() - start) / 60:.1f} min")
