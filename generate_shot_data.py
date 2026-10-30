"""
Per-shot datasets of the paper: the signed confidence Delta of each decoder on
the SAME sampled shots.

    <data>/delta_{decoder}_d{d}_dt{r}_p{p}_shots{N}.npy   (float32, length N; <data> = runs.DATA_DIR)
    decoder in {mwpm, mwpmcorr, gnn}
        mwpm      uncorrelated MWPM, complementary gap   (post_selection.py)
        mwpmcorr  correlated MWPM, shared-graph gap       (correlated_gap.py)
        gnn       pretrained GNN, logit                   (gnn_decoder.py)
    Delta = gap * (+1 if the true logical label is 0, else -1) * 10 / ln(10)
          => Delta > 0: decoder correct; |Delta| = confidence in dB.
    <data>/run_d{d}_dt{r}_p{p}_shots{N}.json   provenance and checks.

Shot stream. Rotated surface code, Z memory, uniform circuit-level noise
(stim-generated circuit with all four noise channels equal to p). Shots are
drawn sequentially from ONE detector sampler seeded with 42 + d + r, in chunks
of 10,000, so all arrays with the same (d, r, p, N) refer to the same shots.
stim reproduces a seeded stream only for the same stim version on the same
kind of machine: the original arrays were produced on Linux (Google Colab,
stim 1.16.0) and are reproduced bit-exactly on Linux (e.g. WSL) with stim
1.16.0 and PyMatching 2.4.0, but NOT on Windows.

Existing arrays are never overwritten; they serve as references. If the mwpm
array exists it is recomputed and must agree bit-for-bit on EVERY shot, which
proves that newly written arrays are aligned with the existing ones. An
existing gnn array can be spot-checked on its first --verify-gnn-shots shots
(decisions must agree; values agree to floating-point rounding between the
original GPU inference and CPU inference).

Extending a run (--extend). The first M shots of an N-shot run are the M shots
of an M-shot run (same seed, same chunks), so an N-shot array is built from the
longest existing shorter array of the same (d, r, p) and decoder: its values are
copied and only shots M..N are decoded. The shorter arrays are kept. Before any
new shot is decoded, every worker recomputes mwpm on one of the last chunks of
the existing mwpm array and requires bit-for-bit agreement, which proves that
its sampler continues the original stream. --verify-gnn-shots then spot-checks
the GNN against the existing (shorter) gnn array.

Examples (Linux):
    python generate_shot_data.py --d 7 --p 0.005 --shots 100000000 \
        --decoders mwpm mwpmcorr --workers 14
    python generate_shot_data.py --d 7 --p 0.003 --shots 40000000 \
        --decoders mwpm mwpmcorr gnn --workers 8 --extend --verify-gnn-shots 10000
"""

import argparse
import datetime
import json
import multiprocessing as mp
import os
import platform
import re
import shutil
import sys
import time

import numpy as np
import pymatching
import stim

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

NATS_TO_DB = 10.0 / np.log(10.0)
CHUNK = 10_000          # shots per sampler call (fixes the shot stream; do not change)
GNN_CHUNK = 2_000       # default shots per GNN forward pass (memory only; --gnn-chunk)
DECODERS = ("mwpm", "mwpmcorr", "gnn")


def seed_for(d, r):
    return 42 + d + r


def build_circuit(d, r, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z", rounds=r, distance=d,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p)


def run_tag(d, r, p, shots):
    return f"d{d}_dt{r}_p{p}_shots{shots}"


def array_path(save_dir, name, d, r, p, shots):
    return os.path.join(save_dir, f"delta_{name}_{run_tag(d, r, p, shots)}.npy")


def longest_prefix(save_dir, name, d, r, p, shots):
    """(path, M) of the longest existing array with M < shots shots, or (None, 0)."""
    best = (None, 0)
    for f in os.listdir(save_dir):
        m = re.fullmatch(rf"delta_{name}_{re.escape(run_tag(d, r, p, ''))}(\d+)\.npy", f)
        if m and best[1] < int(m[1]) < shots and int(m[1]) % CHUNK == 0:
            best = (os.path.join(save_dir, f), int(m[1]))
    return best


def make_decoder(name, dem, d, r, device, gnn_chunk=GNN_CHUNK):
    if name == "mwpm":
        from post_selection import MWPMSoftDecoder
        return MWPMSoftDecoder(dem)
    if name == "mwpmcorr":
        from correlated_gap import CorrelatedGapDecoder
        return CorrelatedGapDecoder(dem)
    if name == "gnn":
        import torch
        torch.set_num_threads(1)  # parallelism comes from the worker processes
        from gnn_decoder import GNNDecoder
        import runs
        ckpt = runs.model_path(d, r)
        # shots per forward pass set the memory: peak ~0.9 GB per process at d = 5 with 2,000
        # (~2.3 GB with 10,000), ~1.0 GB at d = 7 with 500 (~2.4 GB with 2,000); logits agree
        # to float32 rounding (< 1e-5 nats).
        return GNNDecoder(dem, d, r, ckpt, device=device, chunk_size=gnn_chunk)
    raise ValueError(name)


def signed_delta(gaps, obs):
    return (gaps * (1 - 2 * obs[:, 0].astype(np.int8)) * NATS_TO_DB).astype(np.float32)


def _worker(job):
    d, r, p, shots, names, wid, nworkers, write_paths, ref_paths, device, start, anchor_path, gnn_chunk = job
    circuit = build_circuit(d, r, p)
    dem = circuit.detector_error_model(decompose_errors=True)
    decoders = {n: make_decoder(n, dem, d, r, device, gnn_chunk) for n in names}
    outs = {n: np.load(path, mmap_mode="r+") for n, path in write_paths.items()}
    refs = {n: np.load(path, mmap_mode="r") for n, path in ref_paths.items()}
    ref_stats = {n: {"shots": 0, "not_bit_identical": 0, "decision_mismatch": 0, "max_abs_diff_db": 0.0}
                 for n in refs}
    corr_stats = {"shots": 0, "decision_vs_pymatching_mismatch": 0, "mismatch_exact_ties": 0}
    # extension: this worker's chunk among the last nworkers chunks of the existing mwpm array
    anchor_stats = {"shots": 0, "not_bit_identical": 0}
    anchor_chunks = set()
    if anchor_path:
        anchor = np.load(anchor_path, mmap_mode="r")
        na = len(anchor) // CHUNK
        anchor_chunks = {ci for ci in range(max(0, na - nworkers), na) if ci % nworkers == wid}
        anchor_dec = decoders["mwpm"] if "mwpm" in decoders else make_decoder("mwpm", dem, d, r, device)

    sampler = circuit.compile_detector_sampler(seed=seed_for(d, r))
    n_chunks = shots // CHUNK
    first = min(start.values())  # chunks before this one are copied, not decoded
    mine = len([ci for ci in range(wid, n_chunks, nworkers) if ci >= first or ci in anchor_chunks])
    done, t0 = 0, time.perf_counter()
    for ci in range(n_chunks):
        syn, obs = sampler.sample(CHUNK, separate_observables=True)  # every worker replays the stream
        if ci % nworkers != wid or (ci < first and ci not in anchor_chunks):
            continue
        sl = slice(ci * CHUNK, (ci + 1) * CHUNK)
        if ci in anchor_chunks:
            delta = signed_delta(anchor_dec.decode_with_gap(syn)[1], obs)
            anchor_stats["shots"] += CHUNK
            anchor_stats["not_bit_identical"] += int(np.sum(delta != np.asarray(anchor[sl])))
            if anchor_stats["not_bit_identical"]:
                raise RuntimeError(
                    f"chunk {ci}: recomputed mwpm differs from {os.path.basename(anchor_path)} -- the shot "
                    f"stream is not continued (wrong platform, stim version or seed).")
        for n, dec in decoders.items():
            if ci < start[n]:
                continue
            pred, gaps = dec.decode_with_gap(syn)
            delta = signed_delta(gaps, obs)
            if n == "mwpmcorr":
                mism = pred != dec.reference_predictions(syn)
                corr_stats["shots"] += CHUNK
                corr_stats["decision_vs_pymatching_mismatch"] += int(mism.sum())
                corr_stats["mismatch_exact_ties"] += int((mism & (gaps == 0)).sum())
            if n in outs:
                outs[n][sl] = delta
            if n in refs:
                ref = np.asarray(refs[n][sl])
                st = ref_stats[n]
                st["shots"] += CHUNK
                st["not_bit_identical"] += int(np.sum(delta != ref))
                st["decision_mismatch"] += int(np.sum((delta < 0) != (ref < 0)))
                st["max_abs_diff_db"] = max(st["max_abs_diff_db"],
                                            float(np.max(np.abs(delta.astype(np.float64) - ref))))
                if n == "mwpm" and st["not_bit_identical"]:
                    raise RuntimeError(
                        f"chunk {ci}: recomputed mwpm differs from the saved array -- the shot "
                        f"stream is not reproduced (wrong platform, stim version or seed).")
        done += 1
        if wid == 0 and (done % 25 == 0 or done == mine):
            el = time.perf_counter() - t0
            print(f"  [{run_tag(d, r, p, shots)}] worker 0: {done}/{mine} chunks, "
                  f"{1e6 * el / (done * CHUNK):.0f} us/shot, ETA {el / done * (mine - done) / 60:.1f} min",
                  flush=True)
    for o in outs.values():
        o.flush()
    return ref_stats, corr_stats, anchor_stats


def verify_gnn(d, r, p, shots, n_check, ref_path, device, gnn_chunk=GNN_CHUNK):
    """Recompute the GNN on the first n_check shots of an existing gnn array."""
    circuit = build_circuit(d, r, p)
    dec = make_decoder("gnn", circuit.detector_error_model(decompose_errors=True), d, r, device, gnn_chunk)
    ref = np.load(ref_path, mmap_mode="r")
    sampler = circuit.compile_detector_sampler(seed=seed_for(d, r))
    st = {"shots": 0, "bit_identical": 0, "decision_mismatch": 0, "max_abs_diff_db": 0.0}
    for ci in range(-(-n_check // CHUNK)):
        syn, obs = sampler.sample(CHUNK, separate_observables=True)
        delta = signed_delta(dec.decode_with_gap(syn)[1], obs)
        r_ = np.asarray(ref[ci * CHUNK:(ci + 1) * CHUNK])
        st["shots"] += CHUNK
        st["bit_identical"] += int(np.sum(delta == r_))
        st["decision_mismatch"] += int(np.sum((delta < 0) != (r_ < 0)))
        st["max_abs_diff_db"] = max(st["max_abs_diff_db"], float(np.max(np.abs(delta.astype(np.float64) - r_))))
    return st


def versions(with_torch, gnn_chunk=GNN_CHUNK):
    v = {"python": platform.python_version(), "platform": platform.platform(),
         "stim": stim.__version__, "pymatching": pymatching.__version__, "numpy": np.__version__}
    if with_torch:
        import torch, torch_geometric
        v["torch"] = torch.__version__
        v["torch_geometric"] = torch_geometric.__version__
        v["gnn_chunk_size"] = gnn_chunk
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--r", type=int, default=None, help="rounds (default: d)")
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--shots", type=int, required=True)
    ap.add_argument("--decoders", nargs="+", choices=DECODERS, required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--save-dir", default=None, help="default: runs.DATA_DIR (./data or $GNN_CONFIDENCE_DATA)")
    ap.add_argument("--work-dir", default=os.path.join(os.path.expanduser("~"), ".cache", "gnn_confidence_runs"),
                    help="local disk for partial outputs (moved to --save-dir when complete)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--verify-gnn-shots", type=int, default=0)
    ap.add_argument("--verify-existing", action="store_true",
                    help="re-check existing arrays even when nothing new is written")
    ap.add_argument("--gnn-chunk", type=int, default=GNN_CHUNK,
                    help="shots per GNN forward pass (memory only; use 500 at d = 7 for ~1 GB per worker)")
    ap.add_argument("--extend", action="store_true",
                    help="start new arrays from the longest existing shorter arrays of the same (d, r, p) "
                         "and decode only the remaining shots")
    a = ap.parse_args()
    if a.save_dir is None:
        import runs
        a.save_dir = runs.DATA_DIR
    d, r, p, shots = a.d, (a.r or a.d), a.p, a.shots
    if shots % CHUNK:
        sys.exit(f"--shots must be a multiple of {CHUNK}")
    if platform.system() != "Linux":
        print("WARNING: seeded stim streams are not reproduced across platforms; the original "
              "datasets were generated on Linux.", flush=True)
    tag = run_tag(d, r, p, shots)
    os.makedirs(a.save_dir, exist_ok=True)

    write, refs, gnn_check = {}, {}, None
    for n in a.decoders:
        path = array_path(a.save_dir, n, d, r, p, shots)
        if not os.path.exists(path):
            write[n] = path
        elif n == "mwpm":
            refs[n] = path
        elif n == "gnn":
            if a.verify_gnn_shots > 0:
                gnn_check = path
            else:
                print(f"{os.path.basename(path)} exists; not recomputed", flush=True)
        else:
            print(f"{os.path.basename(path)} exists; not overwritten", flush=True)
    mwpm_path = array_path(a.save_dir, "mwpm", d, r, p, shots)
    if not write and not a.verify_existing:
        print(f"[{tag}] all requested arrays exist; nothing to do (use --verify-existing to re-check)", flush=True)
        return
    if write and "mwpm" not in write and os.path.exists(mwpm_path):
        refs["mwpm"] = mwpm_path  # always prove alignment of new arrays against existing ones

    prefix, anchor = {}, None  # prefix[n] = (path, M): shots 0..M of n are copied from path
    if a.extend:
        for n in write:
            path, m = longest_prefix(a.save_dir, n, d, r, p, shots)
            if path:
                prefix[n] = (path, m)
                print(f"[{tag}] {n}: shots 0..{m:,} copied from {os.path.basename(path)}", flush=True)
            else:
                print(f"[{tag}] {n}: no shorter array; all shots decoded", flush=True)
        if prefix and "mwpm" not in refs:
            anchor = longest_prefix(a.save_dir, "mwpm", d, r, p, shots)[0]
            if not anchor:
                sys.exit("--extend needs an existing (shorter) mwpm array to prove that the stream is continued")
        if "gnn" in prefix and a.verify_gnn_shots > 0:
            gnn_check = prefix["gnn"][0]

    meta_path = os.path.join(a.save_dir, f"run_{tag}.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    meta.setdefault("config", {"task": "surface_code:rotated_memory_z", "distance": d, "rounds": r,
                               "p": p, "shots": shots, "seed": seed_for(d, r), "chunk_size": CHUNK,
                               "noise": "uniform circuit-level: after_clifford_depolarization = "
                                        "before_round_data_depolarization = before_measure_flip_probability"
                                        " = after_reset_flip_probability = p"})
    meta.setdefault("arrays", {})
    now = datetime.datetime.now().isoformat(timespec="seconds")

    if gnn_check:
        t0 = time.perf_counter()
        st = verify_gnn(d, r, p, shots, a.verify_gnn_shots, gnn_check, a.device, a.gnn_chunk)
        print(f"[{tag}] gnn check on {st['shots']:,} shots: decision mismatches {st['decision_mismatch']}, "
              f"bit-identical {st['bit_identical']:,}, max |diff| {st['max_abs_diff_db']:.2e} dB "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
        meta["arrays"].setdefault("gnn", {})["check_" + now] = {**st, "versions": versions(True, a.gnn_chunk)}
        if st["decision_mismatch"]:
            json.dump(meta, open(meta_path, "w"), indent=1)
            sys.exit("GNN check failed")

    names = sorted(set(write) | set(refs), key=DECODERS.index)
    if names:
        os.makedirs(a.work_dir, exist_ok=True)
        partial = {n: os.path.join(a.work_dir, f"partial_{n}_{tag}.npy") for n in write}
        for n, path in partial.items():
            m = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(shots,))
            m[:] = np.nan
            if n in prefix:
                src = np.load(prefix[n][0], mmap_mode="r")
                for i in range(0, len(src), 1000 * CHUNK):
                    j = min(i + 1000 * CHUNK, len(src))
                    m[i:j] = src[i:j]
                del src
            m.flush()
            del m
        start = {n: prefix[n][1] // CHUNK if n in prefix else 0 for n in names}
        print(f"[{tag}] computing {names} with {a.workers} workers; writing {sorted(write)}; "
              f"verifying {sorted(refs)}" + (f"; first decoded shot {min(start.values()) * CHUNK:,}"
                                             if prefix else ""), flush=True)
        t0 = time.perf_counter()
        jobs = [(d, r, p, shots, names, w, a.workers, partial, refs, a.device, start, anchor, a.gnn_chunk)
                for w in range(a.workers)]
        try:
            with mp.get_context("spawn").Pool(a.workers) as pool:
                results = pool.map(_worker, jobs)
        except Exception:
            for path in partial.values():
                if os.path.exists(path):
                    os.remove(path)
            raise
        minutes = (time.perf_counter() - t0) / 60

        ref_tot = {}
        for rs, _, _ in results:
            for n, st in rs.items():
                tot = ref_tot.setdefault(n, {"shots": 0, "not_bit_identical": 0, "decision_mismatch": 0,
                                             "max_abs_diff_db": 0.0})
                for k in ("shots", "not_bit_identical", "decision_mismatch"):
                    tot[k] += st[k]
                tot["max_abs_diff_db"] = max(tot["max_abs_diff_db"], st["max_abs_diff_db"])
        corr_tot = {k: sum(res[1][k] for res in results) for k in results[0][1]}
        for n, st in ref_tot.items():
            print(f"[{tag}] {n} reference: {st['shots']:,} shots, {st['not_bit_identical']} not bit-identical",
                  flush=True)
            meta["arrays"].setdefault(n, {})["check_" + now] = st
        if anchor:
            anchor_tot = {k: sum(res[2][k] for res in results) for k in results[0][2]}
            anchor_tot["array"] = os.path.basename(anchor)
            print(f"[{tag}] stream continuation: mwpm recomputed on {anchor_tot['shots']:,} shots at the end of "
                  f"{anchor_tot['array']}, {anchor_tot['not_bit_identical']} not bit-identical", flush=True)

        for n, path in partial.items():
            arr = np.load(path, mmap_mode="r")
            if np.isnan(arr).any():
                sys.exit(f"{n}: unwritten shots in {path}")
            del arr
            shutil.move(path, write[n])
            entry = {"created": now, "minutes": round(minutes, 1), "workers": a.workers,
                     "versions": versions(n == "gnn", a.gnn_chunk), "error_rate": None}
            arr = np.load(write[n], mmap_mode="r")
            entry["error_rate"] = float(np.mean(np.signbit(arr)))
            if n in prefix:
                src, m = prefix[n]
                src_meta = os.path.join(a.save_dir, f"run_{run_tag(d, r, p, m)}.json")
                entry["decoded_shots"] = [m, shots]  # minutes, workers, versions and checks refer to these
                entry["extended_from"] = {
                    "array": os.path.basename(src), "shots": m,
                    "record": (json.load(open(src_meta))["arrays"].get(n) if os.path.exists(src_meta) else None)}
                if anchor:
                    entry["stream_continuation_check"] = anchor_tot
            if n == "mwpmcorr":
                entry["consistency_with_pymatching_correlated"] = corr_tot
                fast = hasattr(pymatching.Matching()._matching_graph, "decode_both_sectors_shared_reweighting")
                entry["implementation"] = ("cpp" if fast and os.environ.get("CORRELATED_GAP_IMPL", "auto").lower()
                                           != "python" else "python")
            meta["arrays"][n] = {**meta["arrays"].get(n, {}), **entry}
            print(f"[{tag}] wrote {os.path.basename(write[n])}: error rate {entry['error_rate']:.4e}", flush=True)
        if "mwpmcorr" in write:
            print(f"[{tag}] mwpmcorr decisions vs PyMatching correlated: "
                  f"{corr_tot['decision_vs_pymatching_mismatch']} mismatches "
                  f"({corr_tot['mismatch_exact_ties']} exact ties)", flush=True)
        print(f"[{tag}] done in {minutes:.1f} min", flush=True)

    json.dump(meta, open(meta_path, "w"), indent=1)


if __name__ == "__main__":
    main()
