"""
Validation of the correlated complementary gap (correlated_gap.py).

1. Consistency with PyMatching's correlated decoder (d = 3 and 5): the sign of
   the shared-graph gap must reproduce PyMatching's correlated decision on
   every shot except exact ties, and min(w'0, w'1) must equal PyMatching's
   correlated second-pass weight.
2. Distance to the exact posterior (d = r = 3): the exact log-likelihood ratio
   Lambda(s) = log P(L=0|s)/P(L=1|s) is computed for every syndrome from the
   full (undecomposed) detector error model with a Walsh-Hadamard transform
   over the 24 detectors + 1 observable,
       P_hat(k) = prod_j (1 - 2 p_j)^[k . v_j odd],   P = WHT(P_hat) / 2^25.
   For each gap g, q(L=0|s) = sigmoid(a g) and the mean per-shot
   KL(P(.|s) || q(.|s)) is reported for a = 1 and the best global a.

Runs on any platform (fresh shots; no stream reproduction needed).
"""

import argparse
import time

import numpy as np
import stim
from scipy.optimize import minimize_scalar

from correlated_gap import CorrelatedGapDecoder
from post_selection import MWPMSoftDecoder

P = 0.005
LN2 = np.log(2)
DB = 10 / np.log(10)


def circuit(d, r=None, p=P):
    return stim.Circuit.generated("surface_code:rotated_memory_z", rounds=r or d, distance=d,
                                  after_clifford_depolarization=p, before_round_data_depolarization=p,
                                  before_measure_flip_probability=p, after_reset_flip_probability=p)


def consistency(d, shots, seed=11):
    c = circuit(d)
    dec = CorrelatedGapDecoder(c.detector_error_model(decompose_errors=True))
    syn, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    t0 = time.perf_counter()
    pred, gap = dec.decode_with_gap(syn)
    dt = time.perf_counter() - t0
    ref_pred, ref_w = dec.matching.decode_batch(syn, return_weights=True, enable_correlations=True)
    ref_pred = ref_pred[:, 0].astype(np.uint8)
    mism = pred != ref_pred
    # weight check on a subsample
    aug = np.zeros(dec.num_detectors + 1, dtype=np.uint8)
    wmin = np.array([min(dec._sector_weights(s, aug)) for s in syn[:2000]])
    y = obs[:, 0].astype(np.uint8)
    print(f"d=r={d}: {shots:,} shots ({1e6 * dt / shots:.0f} us/shot)")
    print(f"  decision mismatches vs PyMatching correlated: {int(mism.sum())} "
          f"(exact ties among them: {int((mism & (gap == 0)).sum())})")
    print(f"  max |min(w'0, w'1) - PyMatching correlated weight| (2,000 shots): "
          f"{np.max(np.abs(wmin - ref_w[:2000])):.2e}")
    print(f"  LER: shared-graph gap {np.mean(pred != y):.4e}, PyMatching correlated {np.mean(ref_pred != y):.4e}")


def exact_joint(mechs, nbits):
    n = 1 << nbits
    k = np.arange(n, dtype=np.uint32)
    ph = np.ones(n)
    for v, p in mechs:
        ph[(np.bitwise_count(k & np.uint32(v)) & 1).astype(bool)] *= (1 - 2 * p)
    del k
    h = 1
    while h < n:  # in-place fast Walsh-Hadamard transform
        a = ph.reshape(-1, 2, h)
        x = a[:, 0, :].copy()
        a[:, 0, :] += a[:, 1, :]
        a[:, 1, :] = x - a[:, 1, :]
        h *= 2
    return ph / n


def kl_bits(lam_true, lam_q):
    ls = lambda z: -np.logaddexp(0, -z)  # log sigmoid
    p0 = np.exp(ls(lam_true))
    return (p0 * (ls(lam_true) - ls(lam_q)) + (1 - p0) * (ls(-lam_true) - ls(-lam_q))) / LN2


def exact_posterior_d3(shots, seed=4242):
    c = circuit(3)
    full = c.detector_error_model(decompose_errors=False)
    nd = full.num_detectors
    mechs = []
    for inst in full.flattened():
        if inst.type == "error":
            v = 0
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    v ^= 1 << t.val
                elif t.is_logical_observable_id():
                    v ^= 1 << nd
            mechs.append((v, inst.args_copy()[0]))
    pj = exact_joint(mechs, nd + 1)
    syn, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    idx = syn.astype(np.int64) @ (1 << np.arange(nd, dtype=np.int64))
    lam = np.log(pj[idx]) - np.log(pj[idx + (1 << nd)])
    dem = c.detector_error_model(decompose_errors=True)
    _, gap_u = MWPMSoftDecoder(dem).decode_with_gap(syn)
    _, gap_c = CorrelatedGapDecoder(dem).decode_with_gap(syn)
    y = obs[:, 0]
    print(f"\nd=r=3 exact posterior ({len(mechs)} mechanisms, {nd} detectors), {shots:,} shots; "
          f"exact MAP LER {np.mean((lam < 0) != y):.4e}")
    print("  mean per-shot KL to the exact posterior [mbit/shot]:")
    for name, g in (("uncorrelated gap", gap_u), ("correlated gap", gap_c)):
        a = minimize_scalar(lambda a: kl_bits(lam, a * g).mean(), bounds=(0.3, 2), method="bounded").x
        k1, ka = kl_bits(lam, g), kl_bits(lam, a * g)
        se = lambda v: v.std() / np.sqrt(len(v))
        print(f"    {name:17s} LER {np.mean((g < 0) != y):.4e} | a=1: {1e3 * k1.mean():6.2f} +- {1e3 * se(k1):.2f}"
              f" | best a={a:.3f}: {1e3 * ka.mean():6.2f} +- {1e3 * se(ka):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--consistency-shots", type=int, default=20_000)
    ap.add_argument("--exact-shots", type=int, default=400_000)
    a = ap.parse_args()
    for d in (3, 5):
        consistency(d, a.consistency_shots)
    exact_posterior_d3(a.exact_shots)
