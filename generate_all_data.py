"""
Generate the 20 datasets of the paper (runs.py) with generate_shot_data.py.

Every run samples 10^8 shots (seed 42 + d + r, sampler chunks of 10,000; Linux,
stim 1.16.0) and writes the signed confidence of MWPM_uncorr, MWPM_corr and the
GNN on the same shots to runs.DATA_DIR. Existing arrays are never overwritten:
an existing mwpm array is recomputed and must agree bit for bit (this proves that
new arrays are aligned with it), and an existing GNN array is spot-checked on its
first --verify-gnn-shots shots.

Cost (16-thread CPU, patched PyMatching, see pymatching_patch/): MWPM_corr takes
~15 min (d = 5) to ~1.7 h (d = 9) per run; the GNN on CPU dominates, ~1 ms per
shot and worker at d = 5 and 4-5x that at d = 7, i.e. days for all runs. The GNN
needs the checkpoints of fetch_gnn_models.py and ~1 GB of memory per worker.

    python generate_all_data.py                               # everything
    python generate_all_data.py --only d5_dt5_p0.005          # one run (prefix of runs.tag)
    python generate_all_data.py --decoders mwpm mwpmcorr      # matching decoders only
    python generate_all_data.py --shots 100000 --save-dir /tmp/test   # first 10^5 shots of every run
"""

import argparse
import os
import subprocess
import sys

import runs

HERE = os.path.dirname(os.path.abspath(__file__))
GNN_CHUNK = {5: 2000, 7: 500, 9: 500}   # shots per GNN forward pass: ~1 GB per worker (memory only)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None, help="run tags (or prefixes), e.g. d7_dt7_p0.003")
    ap.add_argument("--shots", type=int, default=runs.SHOTS)
    ap.add_argument("--decoders", nargs="+", choices=runs.DECODERS, default=list(runs.DECODERS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--gnn-workers", type=int, default=None,
                    help="workers when the GNN is decoded (default: --workers; ~1 GB of memory each)")
    ap.add_argument("--verify-gnn-shots", type=int, default=10_000)
    ap.add_argument("--save-dir", default=runs.DATA_DIR)
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = ap.parse_args()
    for d, r, p, _ in runs.runs():
        tag = runs.tag(d, r, p, a.shots)
        if a.only and not any(tag.startswith(o) for o in a.only):
            continue
        workers = a.gnn_workers if ("gnn" in a.decoders and a.gnn_workers) else a.workers
        cmd = [sys.executable, "-W", "ignore", os.path.join(HERE, "generate_shot_data.py"),
               "--d", str(d), "--r", str(r), "--p", str(p), "--shots", str(a.shots),
               "--decoders", *a.decoders, "--workers", str(workers), "--save-dir", a.save_dir,
               "--gnn-chunk", str(GNN_CHUNK[d]), "--verify-gnn-shots", str(a.verify_gnn_shots)]
        print(f"== {tag}: " + " ".join(cmd[2:]), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, check=True, cwd=HERE)


if __name__ == "__main__":
    main()
