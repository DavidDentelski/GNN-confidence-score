"""
Appendix B: two constructions of a correlation-aware complementary gap, compared on the
original shots (seed 42 + d + r, generate_shot_data.py) at p = 0.005.

  shared graph (correlated_gap.py, used in the paper): one first pass on the measured
      syndrome; both logical sectors are matched on the same reweighted graph,
      gap = w'(sector 1) - w'(sector 0)  (the construction of Yoked surface codes);
  per sector (as in pymatching-rs, decode_gap with correlations): the decision and its
      weight come from PyMatching's correlated decoder; the complementary sector is decoded
      separately, with its own first pass and reweighting, on the graph whose logical
      observable is a virtual detector forced to the other value; gap = |w_alt - w_hard|.

Both scores rank the same decisions (PyMatching's correlated prediction). Reported per
distance: consistency of the shared-graph gap with PyMatching's correlated decoder (decision
mismatches, all of them at exact ties; weight of the decided sector, with the patched
PyMatching build) and bit-identity with the saved delta_mwpmcorr arrays, tie-aware AUC of
each score, post-selected LER at several acceptance rates (with the GNN pair on the same
shots) and a chunk-bootstrap 95% interval for the ratio, and the fraction of shots on which
the complementary sector becomes lighter than the decided one after its own reweighting.

The default is all 10^8 shots of the saved (d, d) arrays. Per-shot results are written chunk
by chunk to memory-mapped files in --work-dir (local disk, ~1.7 GB per distance), so an
interrupted run resumes from the finished chunks when restarted with the same arguments; the
files are removed once the output is written. An existing output with the requested number
of shots is not recomputed (--force recomputes it).

Output: <out>/gap_constructions_d<d>.json (default build/gap_constructions/).
Usage (Linux, PyMatching 2.4.0):
    python compare_gap_constructions.py --distances 5 7               # 10^8 shots each
    python compare_gap_constructions.py --distances 5 7 --chunks 200  # first 2e6 shots
"""

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

import runs

P = runs.P_MAIN
SAVED_SHOTS = runs.SHOTS
CHUNK = 10_000   # sampler chunk of the shot stream (generate_shot_data.CHUNK)
KAPPAS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
BOOT_KAPPAS, N_BOOT, BOOT_SEED = (0.8, 0.9, 0.95), 200, 5
# per-chunk record: done flag, decision mismatches vs PyMatching (all, at nonzero gap),
# max |decided-sector weight - PyMatching weight| (NaN without the patched build),
# shots not bit-identical to the saved delta_mwpmcorr array (NaN if it does not exist),
# shots with the complementary sector lighter after its own reweighting, failures among them
STATS = ("done", "mismatch", "mismatch_nonzero_gap", "max_abs_weight_diff", "saved_not_bit_identical",
         "sign_inconsistent", "sign_inconsistent_failures")


def saved_path(name, d):
    return runs.array_path(name, d, d, P, SAVED_SHOTS)


def work_paths(work_dir, d, n_chunks):
    tag = runs.tag(d, d, P, n_chunks * CHUNK)
    return {k: os.path.join(work_dir, f"gapcons_{k}_{tag}.npy") for k in ("fail", "conf_shared", "raw_sector", "stats")}


def open_work_files(paths, n_chunks):
    """Create the per-shot files, or keep them (resume) if they exist with the right shapes."""
    n = n_chunks * CHUNK
    spec = {"fail": (np.uint8, (n,)), "conf_shared": (np.float64, (n,)), "raw_sector": (np.float64, (n,)),
            "stats": (np.float64, (n_chunks, len(STATS)))}
    ok = all(os.path.exists(p) for p in paths.values())
    if ok:
        for k, (dtype, shape) in spec.items():
            m = np.load(paths[k], mmap_mode="r")
            ok &= m.dtype == dtype and m.shape == shape
            del m
    if not ok:
        for k, (dtype, shape) in spec.items():
            m = np.lib.format.open_memmap(paths[k], mode="w+", dtype=dtype, shape=shape)
            m[:] = 0
            m.flush()
            del m
    return int(np.load(paths["stats"], mmap_mode="r")[:, 0].sum())


def worker(args):
    import pymatching
    import generate_shot_data as g
    from correlated_gap import CorrelatedGapDecoder
    from post_selection import MWPMSoftDecoder
    assert g.CHUNK == CHUNK
    d, n_chunks, wid, nw, paths = args
    circuit = g.build_circuit(d, d, P)
    dem = circuit.detector_error_model(decompose_errors=True)
    dec = CorrelatedGapDecoder(dem)
    per_sector = pymatching.Matching.from_detector_error_model(
        MWPMSoftDecoder._promote_observable(dem, 0), enable_correlations=True)
    nd = dem.num_detectors
    out = {k: np.load(p, mmap_mode="r+") for k, p in paths.items()}
    saved = np.load(saved_path("mwpmcorr", d), mmap_mode="r") if os.path.exists(saved_path("mwpmcorr", d)) else None
    todo = [ci for ci in range(wid, n_chunks, nw) if not out["stats"][ci, 0]]
    done, t0 = 0, time.perf_counter()
    sampler = circuit.compile_detector_sampler(seed=g.seed_for(d, d))
    for ci in range(todo[-1] + 1 if todo else 0):
        syn, obs = sampler.sample(g.CHUNK, separate_observables=True)   # every chunk is drawn: fixes the stream
        if ci % nw != wid or out["stats"][ci, 0]:
            continue
        sl = slice(ci * g.CHUNK, (ci + 1) * g.CHUNK)
        syn = np.ascontiguousarray(syn, dtype=np.uint8)
        pred, w_hard = dec.matching.decode_batch(syn, return_weights=True, enable_correlations=True)
        pred = pred[:, 0].astype(np.uint8)
        aug = np.zeros((len(syn), nd + 1), dtype=np.uint8)
        aug[:, :nd] = syn
        aug[:, nd] = 1 - pred
        _, w_alt = per_sector.decode_batch(aug, return_weights=True, enable_correlations=True)
        if dec.implementation == "cpp":   # sector weights are exposed by the patched PyMatching
            w = dec.promoted_corr._matching_graph.decode_both_sectors_shared_reweighting(
                dec.matching._matching_graph, syn, dec.boundary_to_virtual, dec.virtual)
            gap, w_pred = w[:, 1] - w[:, 0], np.where(pred == 1, w[:, 1], w[:, 0])
        else:
            gap, w_pred = dec.decode_with_gap(syn)[1], np.full(len(syn), np.nan)
        obs = obs[:, 0].astype(np.uint8)
        w_hard, w_alt = np.asarray(w_hard), np.asarray(w_alt)
        fail = pred != obs
        raw_sector = (w_alt - w_hard) * g.NATS_TO_DB
        mism = pred != (gap < 0)
        out["fail"][sl] = fail
        out["conf_shared"][sl] = np.abs(gap) * g.NATS_TO_DB
        out["raw_sector"][sl] = raw_sector
        for k in ("fail", "conf_shared", "raw_sector"):
            out[k].flush()
        out["stats"][ci] = (
            1, mism.sum(), (mism & (gap != 0)).sum(), np.max(np.abs(w_pred - w_hard)),
            np.nan if saved is None else np.sum(g.signed_delta(gap, obs[:, None]) != np.asarray(saved[sl])),
            (raw_sector < 0).sum(), (fail & (raw_sector < 0)).sum())
        out["stats"].flush()
        done += 1
        if wid == 0 and (done % 50 == 0 or done == len(todo)):
            el = time.perf_counter() - t0
            print(f"  d = {d}, worker 0: {done}/{len(todo)} chunks, ETA {el / done * (len(todo) - done) / 60:.0f} min",
                  flush=True)
    return done


def ler_at(conf, fail, kappa):
    """Post-selected LER at acceptance rate kappa, ties accepted fractionally (Appendix A)."""
    n = len(conf)
    n_acc = int(np.floor(kappa * n))
    v = np.partition(conf, n - n_acc)[n - n_acc]
    above, at = conf > v, conf == v
    f = float((fail & above).sum()) + float((fail & at).sum()) * (n_acc - int(above.sum())) / int(at.sum())
    return f / n_acc, f


def auc(conf, fail):
    """Tie-aware AUC (midranks): P(conf of a correct shot > conf of a failure), ties half."""
    order = np.argsort(conf, kind="stable")
    sc = conf[order]
    new = np.empty(len(sc), dtype=bool)
    new[0] = True
    np.not_equal(sc[1:], sc[:-1], out=new[1:])
    starts = np.flatnonzero(new)
    ranks = (starts + (np.diff(np.append(starts, len(sc))) + 1) / 2.0)[np.cumsum(new) - 1]
    f = fail[order]
    nf = int(f.sum())
    ns = len(f) - nf
    return 1.0 - (ranks[f].sum() - nf * (nf + 1) / 2.0) / (nf * ns)


def compare(d, n_chunks, workers, out_dir, work_dir, force):
    n = n_chunks * CHUNK
    path = os.path.join(out_dir, f"gap_constructions_d{d}.json")
    if not force and os.path.exists(path) and json.load(open(path)).get("shots") == n:
        print(f"d = {d}: {os.path.basename(path)} with {n:,} shots exists; not recomputed (--force)", flush=True)
        return
    os.makedirs(work_dir, exist_ok=True)
    paths = work_paths(work_dir, d, n_chunks)
    finished = open_work_files(paths, n_chunks)
    print(f"d = {d}: {n:,} shots, {workers} workers" +
          (f"; resuming, {finished} of {n_chunks} chunks already done" if finished else ""), flush=True)
    t0 = time.perf_counter()
    if finished < n_chunks:
        with mp.get_context("spawn").Pool(workers) as pool:
            pool.map(worker, [(d, n_chunks, w, workers, paths) for w in range(workers)])
    stats = np.load(paths["stats"])
    if not stats[:, 0].all():
        raise RuntimeError(f"d = {d}: {int((stats[:, 0] == 0).sum())} chunks not decoded")
    print(f"d = {d}: decoded in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    s = dict(zip(STATS, stats.T))

    checks = {"decision_mismatch_vs_pymatching": int(s["mismatch"].sum()),
              "decision_mismatch_at_nonzero_gap": int(s["mismatch_nonzero_gap"].sum())}
    if np.isfinite(s["max_abs_weight_diff"]).all():
        checks["max_abs_decided_sector_weight_minus_pymatching"] = float(s["max_abs_weight_diff"].max())
    if np.isfinite(s["saved_not_bit_identical"]).all():
        checks["saved_mwpmcorr_bit_identical"] = bool(s["saved_not_bit_identical"].sum() == 0)
    gnn = np.asarray(np.load(saved_path("gnn", d), mmap_mode="r")[:n]) if os.path.exists(saved_path("gnn", d)) else None
    fail = np.load(paths["fail"]).astype(bool)
    conf_shared = np.load(paths["conf_shared"])
    conf_sector = np.abs(np.load(paths["raw_sector"]))
    res = {"d": d, "p": P, "shots": n, "checks": checks, "hard_ler": float(fail.mean()),
           "auc": {"shared": auc(conf_shared, fail), "per_sector": auc(conf_sector, fail)}, "kappa": {}}
    for k in KAPPAS:
        ls, fs_ = ler_at(conf_shared, fail, k)
        lp, fp = ler_at(conf_sector, fail, k)
        res["kappa"][str(k)] = {"shared": [ls, fs_], "per_sector": [lp, fp],
                                "ratio_per_sector_over_shared": lp / ls if ls else None}
        if gnn is not None:   # the GNN pair on the same shots, for comparison
            res["kappa"][str(k)]["gnn"] = list(ler_at(np.abs(gnn), np.signbit(gnn), k))
    del gnn
    rng = np.random.default_rng(BOOT_SEED)
    idx = np.arange(n).reshape(n_chunks, CHUNK)
    boot = {k: [] for k in BOOT_KAPPAS}
    for _ in range(N_BOOT):
        ix = idx[rng.integers(0, n_chunks, n_chunks)].ravel()
        cs, cp, fx = conf_shared[ix], conf_sector[ix], fail[ix]
        for k in BOOT_KAPPAS:
            ls = ler_at(cs, fx, k)[0]
            boot[k].append(ler_at(cp, fx, k)[0] / ls if ls else np.nan)
    res["ratio_ci95"] = {str(k): [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
                         for k, v in boot.items()}
    n_inc = int(s["sign_inconsistent"].sum())
    res["per_sector_sign_inconsistent"] = {"fraction": n_inc / n,
                                           "failure_rate": float(s["sign_inconsistent_failures"].sum()) / n_inc
                                           if n_inc else None}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(path, "w"), indent=1)
    for p in paths.values():
        os.remove(p)
    print(f"  decisions: {checks}")
    k9 = res["kappa"]["0.9"]
    print(f"  L(0.9): shared {k9['shared'][0]:.2e}, per sector {k9['per_sector'][0]:.2e}"
          + (f", GNN {k9['gnn'][0]:.2e}" if "gnn" in k9 else ""))
    print(f"  AUC shared {res['auc']['shared']:.4f}, per sector {res['auc']['per_sector']:.4f}; "
          f"L(0.9) per sector / shared = {res['kappa']['0.9']['ratio_per_sector_over_shared']:.2f} "
          f"(95% {res['ratio_ci95']['0.9'][0]:.2f}-{res['ratio_ci95']['0.9'][1]:.2f}); "
          f"sign-inconsistent {100 * res['per_sector_sign_inconsistent']['fraction']:.2f}% of shots "
          f"(failure rate {100 * res['per_sector_sign_inconsistent']['failure_rate']:.1f}%) -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distances", type=int, nargs="+", default=[5, 7])
    ap.add_argument("--chunks", type=int, default=SAVED_SHOTS // CHUNK,
                    help="chunks of 10,000 shots (default 10,000 = all 1e8 saved shots; 200 = 2e6)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=os.path.join(runs.BUILD_DIR, "gap_constructions"))
    ap.add_argument("--work-dir", default=os.path.join(os.path.expanduser("~"), ".cache", "gnn_confidence_runs"),
                    help="local disk for the per-shot files of a run in progress")
    ap.add_argument("--force", action="store_true", help="recompute an existing output")
    a = ap.parse_args()
    for d in a.distances:
        compare(d, a.chunks, a.workers, a.out, a.work_dir, a.force)


if __name__ == "__main__":
    main()
