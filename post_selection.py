"""
MWPM decoder with complementary-gap soft output for post-selection.

Implements the boundary-side observable promotion:
the logical observable in the DEM is replaced by a virtual detector.
Decoding the promoted DEM twice — once with the virtual
detector forced to 0, once to 1 — gives the matching weight for each
logical class; their signed difference is the complementary gap.

"""

import pymatching
import stim
import numpy as np


class MWPMSoftDecoder:
    """MWPM decoder with complementary-gap soft output."""

    def __init__(self, dem: stim.DetectorErrorModel, obs_idx: int = 0):
        self._n_det_orig = dem.num_detectors
        self._obs_idx = obs_idx
        self._dem_promoted = self._promote_observable(dem, obs_idx)
        self._matching = pymatching.Matching.from_detector_error_model(
            self._dem_promoted
        )

    @staticmethod
    def _promote_observable(
            dem: stim.DetectorErrorModel, obs_idx: int
    ) -> stim.DetectorErrorModel:
        """
        Return a new DEM in which logical observable `obs_idx` is replaced by
        a virtual detector at index `dem.num_detectors`.

        The promoted DEM has:
          - one extra detector (the virtual one)
          - zero logical observables (the promoted one is gone)
        """
        # The new detector takes the next available index.
        virtual_det_id = dem.num_detectors

        new_dem = stim.DetectorErrorModel()
        for inst in dem.flattened():
            if inst.type == "error":
                # An error has probability args_copy()[0] and a list of targets.
                # Targets may be detector ids, observable ids, or '^' separators
                p = inst.args_copy()[0]
                new_targets = []
                for t in inst.targets_copy():
                    if t.is_logical_observable_id() and t.val == obs_idx:
                        new_targets.append(
                            stim.target_relative_detector_id(virtual_det_id)
                        )
                    else:
                        # Everything else passes through unchanged.
                        new_targets.append(t)
                new_dem.append("error", [p], new_targets)

            elif inst.type == "detector":
                new_dem.append(inst)

        return new_dem

    def decode_with_gap(self, syndromes):
        """
        Decode a batch of syndromes and return predictions plus complementary gaps.

        Parameters
        ----------
        syndromes : array of shape (shots, n_det_orig)
        Does NOT include the virtual detector

        Returns
        -------
        predictions : (shots,), values in {0, 1}
            The MWPM prediction for the logical observable.
        gaps : (shots,)
            gap = w_1 - w_0, where w_b is the minimum matching weight when
            the virtual detector is forced to b.
              gap > 0  <=>  prediction = 0
              gap < 0  <=>  prediction = 1
            |gap| is the confidence in nats (PyMatching weights are
            log((1-p)/p), so this is a log-likelihood ratio).
        """
        # Ensure dtype/shape are what PyMatching expects
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        if syndromes.ndim != 2 or syndromes.shape[1] != self._n_det_orig:
            raise ValueError(
                f"Expected syndromes of shape (shots, {self._n_det_orig}); "
                f"got {syndromes.shape}."
            )
        shots = syndromes.shape[0]

        # The promoted matching graph has one extra detector (the virtual one).
        # We build two augmented syndrome arrays: aug_0 has the virtual bit = 0,
        # aug_1 has it = 1. The remaining columns are the same in both.
        zeros_col = np.zeros((shots, 1), dtype=np.uint8)
        ones_col = np.ones((shots, 1), dtype=np.uint8)
        aug_0 = np.concatenate([syndromes, zeros_col], axis=1)
        aug_1 = np.concatenate([syndromes, ones_col], axis=1)

        _, w_0 = self._matching.decode_batch(aug_0, return_weights=True)
        _, w_1 = self._matching.decode_batch(aug_1, return_weights=True)

        # Signed complementary gap; prediction reads off the sign.
        # If w_1 > w_0, the matching will choose w_0, i.e. no logical error
        # Else w_1 < w_0 (gaps < 0), so the matching will choose w_1 - logical error
        gaps = w_1 - w_0
        predictions = (gaps < 0).astype(np.uint8)
        return predictions, gaps

    def decode(self, syndromes):
        """
        Predictions only. Matches the signature of your existing Decoder classes
        so this can be dropped into run_experiment unmodified.
        """
        predictions, _ = self.decode_with_gap(syndromes)
        return predictions.astype(np.bool_)

class PostSelector:
    """
    Post-selection analysis: rank shots by confidence, compute LER at any
    acceptance fraction.

    Inputs (all 1-D arrays of length `shots`):
      predictions  : decoder's predicted logical class for each shot
      gaps         : signed complementary gap for each shot (in nats)
      true_labels  : ground-truth logical class (from stim's sampler)

    The shots are sorted internally by descending |gap| so that index 0 is the
    most-confident shot and index n-1 is the least-confident. The top-kappa
    fraction is then the most-confident subset to keep under post-selection.
    """

    def __init__(self, predictions, gaps, true_labels):
        preds = np.asarray(predictions, dtype=np.uint8).ravel()
        gaps = np.asarray(gaps, dtype=np.float64).ravel()
        labels = np.asarray(true_labels, dtype=np.uint8).ravel()
        if not (len(preds) == len(gaps) == len(labels)):
            raise ValueError(
                "predictions, gaps, true_labels must have the same length."
            )

        # Sort by |gap| descending — most confident shot first.
        # argsort gives ascending order; we negate |gap| to flip it.
        # 'stable' so ties keep their original order (cosmetic, not essential).
        order = np.argsort(-np.abs(gaps), kind="stable")
        self.predictions = preds[order]
        self.gaps = gaps[order]
        self.true_labels = labels[order]

        # An "error" is a shot where the decoder's prediction is wrong.
        # This is what we count in the LER. Stored as 0/1 so we can cumsum it.
        self.errors = (self.predictions != self.true_labels).astype(np.uint8)
        self.n_shots = len(self.predictions)

        # Cumulative error count: _cumerr[i] = number of errors among the
        # top-(i+1) most-confident shots. Precomputed once so any kappa lookup
        # is O(1) rather than O(n_shots).
        self._cumerr = np.cumsum(self.errors)

    def ler_at_acceptance(self, kappa):
        """LER if we keep the top-kappa most-confident fraction of shots."""
        if not 0 < kappa <= 1:
            raise ValueError(f"kappa must be in (0, 1]; got {kappa}")
        # Number of shots kept; at least 1 so we don't divide by zero.
        n_kept = max(1, int(round(kappa * self.n_shots)))
        # Total errors among those shots, divided by how many we kept.
        return float(self._cumerr[n_kept - 1] / n_kept)

    def acceptance_curve(self, kappas=None):
        """
        Vectorised version of ler_at_acceptance: returns (kappas, lers).

        Default kappas: 100 fractions from 0.01 to 1.0.
        At kappa = 1.0 the returned LER equals the standard (no post-selection) LER.
        """
        if kappas is None:
            kappas = np.linspace(0.01, 1.0, 100)
        kappas = np.asarray(kappas, dtype=np.float64)
        n_keep = np.clip(
            np.round(kappas * self.n_shots).astype(int),
            1, self.n_shots,
        )
        # Cumulative errors at each n_keep, divided by n_keep.
        lers = self._cumerr[n_keep - 1] / n_keep
        return kappas, lers

if __name__ == "__main__":
    d, rounds, p, shots = 5, 5, 5e-3, 100_000

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds, distance=d,
        after_clifford_depolarization=p, before_round_data_depolarization=p,
        before_measure_flip_probability=p, after_reset_flip_probability=p,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    sampler = circuit.compile_detector_sampler(seed=42)
    syndromes, observables = sampler.sample(shots, separate_observables=True)
    true_labels = observables[:, 0].astype(np.uint8)

    # Plain MWPM, the reference.
    plain = pymatching.Matching.from_detector_error_model(dem)
    plain_preds = plain.decode_batch(syndromes)[:, 0].astype(np.uint8)
    plain_ler = float(np.mean(plain_preds != true_labels))

    # Soft decoder.
    soft = MWPMSoftDecoder(dem)
    soft_preds, gaps = soft.decode_with_gap(syndromes)
    soft_ler = float(np.mean(soft_preds != true_labels))

    print(f"d={d}, p={p}, {shots} shots")
    print(f"  plain LER: {plain_ler:.5f}")
    print(f"  soft  LER: {soft_ler:.5f}")
    print(f"  disagreements: {int(np.sum(plain_preds != soft_preds))} / {shots}")
    print(f"  gap stats: mean(|gap|)={np.mean(np.abs(gaps)):.2f} nats, "
          f"min={gaps.min():.2f}, max={gaps.max():.2f}")

    # --- Post-selection ---
    ps = PostSelector(soft_preds, gaps, true_labels)

    # Sanity check 1: kappa=1 reproduces the plain (no post-selection) LER.
    selected_ler_at_1 = ps.ler_at_acceptance(1.0)
    delta = abs(selected_ler_at_1 - soft_ler)
    print(f"\n  kappa=1.0 LER: {selected_ler_at_1:.5f}  "
          f"(matches soft LER, |delta|={delta:.2e})")
    assert delta < 1e-12, "kappa=1 should reproduce the plain LER"

    # Sanity check 2: LER should drop monotonically as kappa decreases.
    print("\n  Acceptance curve:")
    print(f"  {'kappa':>6}  {'shots kept':>10}  {'LER':>10}  {'suppression':>12}")
    for k in [1.00, 0.99, 0.95, 0.90, 0.75, 0.50, 0.25, 0.10]:
        ler = ps.ler_at_acceptance(k)
        n_kept = max(1, int(round(k * shots)))
        suppression = soft_ler / ler if ler > 0 else float("inf")
        print(f"  {k:>6.2f}  {n_kept:>10,}  {ler:>10.5f}  {suppression:>12.1f}x")