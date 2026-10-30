"""
The datasets of the paper and where they live.

Every analysis uses the same 20 runs: rotated surface code, Z memory, uniform
circuit-level noise, 10^8 shots each, with the signed confidence of three
decoders on the SAME shots (generate_shot_data.py):

    CONFIGS at P_MAIN = 0.005          (d, r) = (5, 5), (5, 7), (5, 9), (5, 11),
                                                (7, 7), (7, 9), (7, 11), (9, 9)
    P-scan, d = r = 5, 7, 9            p = 0.003, 0.0035, 0.004, 0.0045 (and 0.005)

Per-shot arrays (float32, dB; Delta > 0 iff the decoder is correct):

    <DATA_DIR>/delta_{decoder}_d{d}_dt{r}_p{p}_shots{N}.npy,  decoder in DECODERS
    <DATA_DIR>/run_d{d}_dt{r}_p{p}_shots{N}.json               provenance and checks

DATA_DIR is ./data, or the folder named by the environment variable
GNN_CONFIDENCE_DATA. The arrays are published on Zenodo
(https://doi.org/10.5281/zenodo.23817170); without
them every figure and table is rebuilt from the cached derived data in build/.

MODEL_DIR (./gnn_models, or GNN_CONFIDENCE_MODELS) holds the pretrained GNN
checkpoints d{d}_d_t_{r}.pt of Lange et al. (fetch_gnn_models.py); they are
needed only to generate data, not to rebuild the figures.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("GNN_CONFIDENCE_DATA", os.path.join(HERE, "data"))
MODEL_DIR = os.environ.get("GNN_CONFIDENCE_MODELS", os.path.join(HERE, "gnn_models"))
BUILD_DIR = os.path.join(HERE, "build")

DECODERS = ("mwpm", "mwpmcorr", "gnn")   # MWPM_uncorr, MWPM_corr, GNN
P_MAIN = 0.005
SHOTS = 100_000_000
CONFIGS = [(5, 5), (5, 7), (5, 9), (5, 11), (7, 7), (7, 9), (7, 11), (9, 9)]
PSCAN_DISTANCES = (5, 7, 9)
PSCAN_P = (0.003, 0.0035, 0.004, 0.0045, 0.005)


def runs():
    """All 20 runs as (d, r, p, shots)."""
    out = [(d, r, P_MAIN, SHOTS) for d, r in CONFIGS]
    out += [(d, d, p, SHOTS) for d in PSCAN_DISTANCES for p in PSCAN_P if p != P_MAIN]
    return sorted(out)


def pscan_runs(d):
    """The p-scan of distance d (r = d), ascending in p, as (d, r, p, shots)."""
    return [(d, d, p, SHOTS) for p in PSCAN_P]


def tag(d, r, p=P_MAIN, shots=SHOTS):
    return f"d{d}_dt{r}_p{p}_shots{shots}"


def array_path(decoder, d, r, p=P_MAIN, shots=SHOTS):
    return os.path.join(DATA_DIR, f"delta_{decoder}_{tag(d, r, p, shots)}.npy")


def model_path(d, r):
    return os.path.join(MODEL_DIR, f"d{d}_d_t_{r}.pt")


def have_arrays(d, r, p=P_MAIN, shots=SHOTS, decoders=DECODERS):
    return all(os.path.isfile(array_path(n, d, r, p, shots)) for n in decoders)


def require_arrays(d, r, p=P_MAIN, shots=SHOTS, decoders=DECODERS):
    """Raise a clear error when a result is neither cached nor computable from the per-shot arrays."""
    missing = [array_path(n, d, r, p, shots) for n in decoders if not os.path.isfile(array_path(n, d, r, p, shots))]
    if missing:
        raise FileNotFoundError(f"no cached result and {len(missing)} per-shot array(s) missing, e.g. {missing[0]}; "
                                "download the data (README.md) or set GNN_CONFIDENCE_DATA")
