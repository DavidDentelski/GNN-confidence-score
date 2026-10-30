"""
Fig. 4 and Table II: exact-posterior benchmark at (d, r) = (3, 3), p = 0.005
(method: Appendix F).

For every syndrome the exact log-likelihood ratio of the full circuit-level
detector error model,
    Lambda(s) = log P(L = 0 | s) / P(L = 1 | s),
is computed with a Walsh-Hadamard transform over the 24 detectors + 1 observable,
    P_hat(k) = prod_j (1 - 2 p_j)^[k . v_j odd],   P = WHT(P_hat) / 2^25.
On the same sampled shots we compare, as estimates of Lambda (signed log-odds
for L = 0):
    MWPM_uncorr gap, MWPM_corr gap (correlated_gap.py), GNN logit (gnn_decoder.py),
and the posterior of the GNN's TRAINING distribution, Lambda_mix (even mixture of
non-empty syndromes generated at p = 1, 2, 3, 4, 5 x 10^-3; the network does not
receive p), which is what the binary-cross-entropy objective targets.

Reported on the non-trivial shots (the GNN does not see trivial syndromes):
  - logical error rate of the decision sign(lambda),
  - mean per-shot KL(P(.|s) || q(.|s)) to the exact posterior, q(L=0|s) =
    sigmoid(a lambda(s)), natively (a = 1) and with the best global a,
  - mean KL to the training-mixture posterior,
  - KL contributions by exact-confidence bin, and median |lambda| vs |Lambda|.

10^8 shots (seed 4242, drawn in sampler calls of 10,000 shots) are decoded in parallel
workers; per-shot results are written chunk by chunk to memory-mapped files in --work-dir
(local disk, ~3 GB), so an interrupted run resumes from the finished chunks when restarted
with the same arguments; the files are removed once the analysis is done.

Outputs (in --out): tables/exact_d3.tex/.csv (Table II), tables/exact_d3_bins.csv,
exact_d3.pdf/.png (Fig. 4), derived/exact_d3.json (cache; recomputed when missing, when
it holds a different number of shots, or with --recompute). With the cache present only
numpy, scipy and matplotlib are needed; decoding needs stim, PyMatching and torch.
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

import figure_style as fs
import runs
from make_postselection import FIG1_LW, FIG1_MS, FIG1_RC

D, R, P_EVAL, SHOTS, SEED = 3, 3, 0.005, 100_000_000, 4242
CHUNK = 10_000   # shots per sampler call (fixes the shot stream)
SCORED = ("mwpm", "mwpmcorr", "gnn")
P_TRAIN = (0.001, 0.002, 0.003, 0.004, 0.005)
LN2 = np.log(2)
DB = 10 / np.log(10)
BINS_DB = [0, 5, 10, 15, 20, 30, 40, 60, 1e9]
plt.rcParams.update({"pdf.fonttype": 42, "savefig.bbox": "tight"})


def circuit(p):
    import stim
    return stim.Circuit.generated("surface_code:rotated_memory_z", rounds=R, distance=D,
                                  after_clifford_depolarization=p, before_round_data_depolarization=p,
                                  before_measure_flip_probability=p, after_reset_flip_probability=p)


def exact_joint(p):
    """P(s, L) for all 2^(n_det + 1) (syndrome, observable) pairs; bit n_det = observable."""
    dem = circuit(p).detector_error_model(decompose_errors=False)
    nd = dem.num_detectors
    n = 1 << (nd + 1)
    k = np.arange(n, dtype=np.uint32)
    ph = np.ones(n)
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        v = 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                v ^= 1 << t.val
            elif t.is_logical_observable_id():
                v ^= 1 << nd
        ph[(np.bitwise_count(k & np.uint32(v)) & 1).astype(bool)] *= (1 - 2 * inst.args_copy()[0])
    del k
    h = 1
    while h < n:
        a = ph.reshape(-1, 2, h)
        x = a[:, 0, :].copy()
        a[:, 0, :] += a[:, 1, :]
        a[:, 1, :] = x - a[:, 1, :]
        h *= 2
    return ph / n, nd


def dem_size(p=P_EVAL):
    """Detectors and independent error mechanisms of the undecomposed detector error model."""
    dem = circuit(p).detector_error_model(decompose_errors=False)
    return {"detectors": dem.num_detectors,
            "error_mechanisms": sum(1 for inst in dem.flattened() if inst.type == "error")}


def kl_bits(lam_true, lam_q):
    ls = lambda z: -np.logaddexp(0, -z)  # log sigmoid
    p0 = np.exp(ls(lam_true))
    return (p0 * (ls(lam_true) - ls(lam_q)) + (1 - p0) * (ls(-lam_true) - ls(-lam_q))) / LN2


def work_paths(work_dir, shots, chunk):
    tag = f"d{D}_dt{R}_p{P_EVAL}_shots{shots}_chunk{chunk}"
    return {k: os.path.join(work_dir, f"exact_d3_{k}_{tag}.npy")
            for k in ("done", "nontrivial", "y", "syndrome", *SCORED)}


def open_work_files(paths, shots, chunk):
    """Create the per-shot files, or keep them (resume) if they exist with the right shapes."""
    spec = {"done": (np.uint8, shots // chunk), "nontrivial": (np.uint8, shots), "y": (np.uint8, shots),
            "syndrome": (np.int32, shots), **{n: (np.float64, shots) for n in SCORED}}
    ok = all(os.path.exists(p) for p in paths.values())
    if ok:
        for k, (dtype, n) in spec.items():
            m = np.load(paths[k], mmap_mode="r")
            ok &= m.dtype == dtype and m.shape == (n,)
            del m
    if not ok:
        for k, (dtype, n) in spec.items():
            m = np.lib.format.open_memmap(paths[k], mode="w+", dtype=dtype, shape=(n,))
            m[:] = 0
            m.flush()
            del m
    return int(np.load(paths["done"], mmap_mode="r").sum())


def worker(args):
    """Decode this worker's chunks of the shot stream (non-trivial shots only) into the per-shot files."""
    import torch
    from correlated_gap import CorrelatedGapDecoder
    from gnn_decoder import GNNDecoder
    from post_selection import MWPMSoftDecoder
    torch.set_num_threads(1)   # parallelism comes from the worker processes
    shots, chunk, wid, nw, paths = args
    c = circuit(P_EVAL)
    dem = c.detector_error_model(decompose_errors=True)
    nd = dem.num_detectors
    decoders = {"mwpm": MWPMSoftDecoder(dem), "mwpmcorr": CorrelatedGapDecoder(dem),
                "gnn": GNNDecoder(dem, D, R, runs.model_path(D, R), device="cpu")}
    out = {k: np.load(p, mmap_mode="r+") for k, p in paths.items()}
    todo = [ci for ci in range(wid, shots // chunk, nw) if not out["done"][ci]]
    done, t0 = 0, time.perf_counter()
    sampler = c.compile_detector_sampler(seed=SEED)
    for ci in range(todo[-1] + 1 if todo else 0):
        syn, obs = sampler.sample(chunk, separate_observables=True)   # every chunk is drawn: fixes the stream
        if ci % nw != wid or out["done"][ci]:
            continue
        sl = slice(ci * chunk, (ci + 1) * chunk)
        nontriv = syn.any(axis=1)
        syn = syn[nontriv].astype(np.uint8)
        out["nontrivial"][sl] = nontriv
        out["y"][sl] = obs[:, 0]
        out["syndrome"][sl][nontriv] = syn.astype(np.int64) @ (1 << np.arange(nd, dtype=np.int64))
        for n, dec in decoders.items():
            out[n][sl][nontriv] = dec.decode_with_gap(syn)[1]
        for k in ("nontrivial", "y", "syndrome", *SCORED):
            out[k].flush()
        out["done"][ci] = 1
        out["done"].flush()
        done += 1
        if wid == 0 and (done % 50 == 0 or done == len(todo)):
            el = time.perf_counter() - t0
            print(f"  worker 0: {done}/{len(todo)} chunks, ETA {el / done * (len(todo) - done) / 60:.0f} min", flush=True)
    return done


def compute(shots, chunk, workers, work_dir):
    t0 = time.perf_counter()
    os.makedirs(work_dir, exist_ok=True)
    paths = work_paths(work_dir, shots, chunk)
    finished = open_work_files(paths, shots, chunk)
    print(f"  {shots:,} shots, {workers} workers" +
          (f"; resuming, {finished} of {shots // chunk} chunks already done" if finished else ""), flush=True)
    if finished < shots // chunk:
        with mp.get_context("spawn").Pool(workers) as pool:
            pool.map(worker, [(shots, chunk, w, workers, paths) for w in range(workers)])
    if not np.load(paths["done"]).all():
        raise RuntimeError("not all chunks were decoded")
    print(f"  decoded in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    joint = {}
    for p in P_TRAIN:
        joint[p], nd = exact_joint(p)
        print(f"  exact joint at p = {p}: {time.perf_counter() - t0:.0f} s", flush=True)
    nontriv = np.load(paths["nontrivial"]).astype(bool)
    y = np.load(paths["y"])[nontriv].astype(bool)
    idx = np.load(paths["syndrome"])[nontriv].astype(np.int64)
    pj = joint[P_EVAL]
    lam = np.log(pj[idx]) - np.log(pj[idx + (1 << nd)])
    num = sum(joint[p][idx] / (1 - joint[p][0] - joint[p][1 << nd]) for p in P_TRAIN)
    den = sum(joint[p][idx + (1 << nd)] / (1 - joint[p][0] - joint[p][1 << nd]) for p in P_TRAIN)
    lam_mix = np.log(num) - np.log(den)
    del joint, num, den, idx
    if not (np.isfinite(lam).all() and np.isfinite(lam_mix).all()):
        raise RuntimeError("non-finite exact posterior on a sampled syndrome")
    scores = {"exact": lam, "mixture": lam_mix, **{n: np.load(paths[n])[nontriv] for n in SCORED}}
    out = {"shots": shots, "nontrivial": int(nontriv.sum()), "seed": SEED, "chunk": chunk, "scores": {}, "bins": []}
    del nontriv
    for name, s in scores.items():
        a = minimize_scalar(lambda a: kl_bits(lam, a * s).mean(), bounds=(0.3, 2.0), method="bounded").x
        k1, ka, km = kl_bits(lam, s), kl_bits(lam, a * s), kl_bits(lam_mix, s)
        se = lambda v: float(v.std() / np.sqrt(len(v)))
        out["scores"][name] = {"ler": float(np.mean((s < 0) != y)), "kl": float(k1.mean()), "kl_se": se(k1),
                               "best_a": float(a), "kl_rescaled": float(ka.mean()), "kl_rescaled_se": se(ka),
                               "kl_to_mixture": float(km.mean()), "kl_to_mixture_se": se(km)}
    absl = np.abs(lam) * DB
    for lo, hi in zip(BINS_DB[:-1], BINS_DB[1:]):
        m = (absl >= lo) & (absl < hi)
        row = {"lo": lo, "hi": None if hi > 1e8 else hi, "frac": float(m.mean()), "median_abs_lambda_db": {},
               "kl_contrib": {}}
        for name, s in scores.items():
            row["median_abs_lambda_db"][name] = float(np.median(np.abs(s[m]) * DB)) if m.any() else None
            row["kl_contrib"][name] = float(kl_bits(lam[m], s[m]).sum() / len(lam)) if m.any() else None
        out["bins"].append(row)
    for p in paths.values():
        os.remove(p)
    print(f"  done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return out


LABELS = {"exact": "exact posterior", "mixture": "training-mixture posterior",
          "mwpm": r"MWPM$_{\rm uncorr}$ gap", "mwpmcorr": r"MWPM$_{\rm corr}$ gap", "gnn": "GNN logit"}
TABLE_LABELS = {"exact": r"exact $\Lambda$", "mixture": r"$\Lambda_{\rm mix}$", "mwpm": r"MWPM$_{\rm uncorr}$",
                "mwpmcorr": r"MWPM$_{\rm corr}$", "gnn": "GNN"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(runs.BUILD_DIR, "exact_d3"))
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument("--chunk", type=int, default=CHUNK, help="shots per sampler call (fixes the shot stream)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--work-dir", default=os.path.join(os.path.expanduser("~"), ".cache", "gnn_confidence_runs"),
                    help="local disk for the per-shot files of a run in progress")
    a = ap.parse_args()
    if a.shots % a.chunk:
        raise SystemExit("--shots must be a multiple of --chunk")
    for sub in ("derived", "tables"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    cache = os.path.join(a.out, "derived", "exact_d3.json")
    res = json.load(open(cache)) if os.path.exists(cache) and not a.recompute else None
    if res is None or res["shots"] != a.shots:
        res = compute(a.shots, a.chunk, a.workers, a.work_dir)
        res["dem"] = dem_size()
        json.dump(res, open(cache, "w"), indent=1)
    if "dem" not in res:
        res["dem"] = dem_size()
        json.dump(res, open(cache, "w"), indent=1)

    order = ["exact", "mixture", "mwpm", "mwpmcorr", "gnn"]
    with open(os.path.join(a.out, "tables", "exact_d3.csv"), "w") as f:
        f.write("score,ler,kl_mbit,kl_se,best_a,kl_rescaled_mbit,kl_rescaled_se,kl_to_mixture_mbit\n")
        for n in order:
            s = res["scores"][n]
            f.write(f"{n},{s['ler']:.6e},{1e3 * s['kl']:.4f},{1e3 * s['kl_se']:.4f},{s['best_a']:.4f},"
                    f"{1e3 * s['kl_rescaled']:.4f},{1e3 * s['kl_rescaled_se']:.4f},{1e3 * s['kl_to_mixture']:.4f}\n")
    with open(os.path.join(a.out, "tables", "exact_d3_bins.csv"), "w") as f:
        f.write("abs_Lambda_lo_db,abs_Lambda_hi_db,fraction," + ",".join(f"median_abs_{n}_db" for n in order)
                + "," + ",".join(f"kl_contrib_{n}_mbit" for n in order) + "\n")
        for b in res["bins"]:
            f.write(f"{b['lo']},{b['hi']},{b['frac']:.5f}," + ",".join(
                "" if b["median_abs_lambda_db"][n] is None else f"{b['median_abs_lambda_db'][n]:.3f}" for n in order)
                + "," + ",".join("" if b["kl_contrib"][n] is None else f"{1e3 * b['kl_contrib'][n]:.4f}"
                                 for n in order) + "\n")
    lines = [r"\begin{table}", r"\centering", r"\begin{tabular}{l|c|cc|c}", r"\toprule",
             r"Score & LER & KL ($a = 1$) & KL (best $a$) & KL to $\Lambda_{\rm mix}$ \\", r"\hline", r"\midrule"]
    for n in order:
        s = res["scores"][n]
        lines.append(rf"{TABLE_LABELS[n]} & {s['ler']:.3e} & {1e3 * s['kl']:.2f}$\pm${1e3 * s['kl_se']:.2f} & "
                     rf"{1e3 * s['kl_rescaled']:.2f} ({s['best_a']:.3f}) & {1e3 * s['kl_to_mixture']:.2f} \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{Exact-posterior benchmark at $(d, r) = (3, 3)$, $p = 0.005$ ({res['nontrivial']:,} "
              r"non-trivial shots): logical error rate of the decision ${\rm sign}(\lambda)$ and mean per-shot "
              r"Kullback-Leibler divergence (mbit/shot) of $q(L|s) = \sigma(a\lambda(s))$ from the exact "
              r"posterior, natively and with the best global rescaling $a$ (in parentheses), and from the "
              r"posterior $\Lambda_{\rm mix}$ of the GNN's training distribution.}",
              r"\label{exact_d3_table}", r"\end{table}"]
    open(os.path.join(a.out, "tables", "exact_d3.tex"), "w").write("\n".join(lines) + "\n")

    # Fig. 4, drawn like Figs. 1-3 (PRA two-column width)
    centers = [(b["lo"] + (b["hi"] if b["hi"] is not None else 70)) / 2 for b in res["bins"]]
    styles = {"exact": dict(color="black", linestyle=":", marker="None"),
              "mixture": dict(color="#9467bd", linestyle="-.", marker="None"),
              **{n: fs.style(n) for n in ("mwpm", "mwpmcorr", "gnn")}}
    nan = lambda v, f=1.0: np.nan if v is None else f * v   # empty confidence bins
    with plt.rc_context(FIG1_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75), gridspec_kw=dict(wspace=0.28))
        for n in order:
            med = [nan(b["median_abs_lambda_db"][n]) for b in res["bins"]]
            axes[0].plot(centers, med, label=LABELS[n], markersize=FIG1_MS, linewidth=FIG1_LW, **styles[n])
            if n != "exact":
                axes[1].plot(centers, [nan(b["kl_contrib"][n], 1e3) for b in res["bins"]], label=LABELS[n],
                             markersize=FIG1_MS, linewidth=FIG1_LW, **styles[n])
        axes[0].set_ylabel(r"Median score $|\lambda|$ (dB)")
        axes[1].set_ylabel("KL contribution (mbit/shot)")
        axes[1].set_yscale("log")
        for ax, letter in zip(axes, "ab"):
            ax.set_xlabel(r"Exact confidence $|\Lambda|$ (dB)")
            ax.set_title(rf"$\mathbf{{({letter})}}$   $(d, r) = (3, 3)$, $p = {P_EVAL}$", loc="left")
            ax.grid(True, which="major", linestyle="-", alpha=0.18, linewidth=0.5)
            ax.tick_params(which="both", direction="in", top=True, right=True)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0),
                   handlelength=2.2, columnspacing=1.0)
        fig.subplots_adjust(top=0.8)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(a.out, f"exact_d3.{ext}"), dpi=300)
        plt.close(fig)
    for n in order:
        s = res["scores"][n]
        print(f"  {n:9s} LER {s['ler']:.4e} | KL {1e3 * s['kl']:7.2f} mbit (a=1) {1e3 * s['kl_rescaled']:7.2f} "
              f"(a={s['best_a']:.3f}) | KL to mixture {1e3 * s['kl_to_mixture']:7.2f}")


if __name__ == "__main__":
    main()
