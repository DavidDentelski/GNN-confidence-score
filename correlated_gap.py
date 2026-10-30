"""
Correlation-aware complementary gap for MWPM ("MWPM_corr").

PyMatching (>= 2.3) decodes with correlations in two passes
(sparse_blossom/driver/mwpm_decoding.cc):
  1. an ordinary matching on the graph loaded with enable_correlations=True
     gives the first-pass edge set E;
  2. for every e in E and every correlated partner a of e (another component
     of the same decomposed DEM error), the weight of a is lowered to the
     implied weight  log((1 - q) / q),  q = min(0.5, P(e and a) / P(e)),
     if that is smaller (user_graph.cc: populate_implied_edge_weights;
     flooder/graph.h: apply_reweights);
  3. a second matching on the reweighted graph gives the decision.

The complementary gap must compare the two logical sectors on ONE graph.
Here both sectors are matched on the same shot-dependent reweighted graph
(the logical observable promoted to a virtual detector, as in
post_selection.MWPMSoftDecoder, forced to 0 and to 1):

    gap_corr = w'(sector 1) - w'(sector 0)        [nats]
    gap_corr > 0  <=>  prediction 0.

By construction min(w'0, w'1) is PyMatching's correlated second-pass weight
and sign(gap_corr) its correlated decision (apart from exact ties); see
validate_correlated_gap.py. Forcing each sector with its own first pass
instead (two different reweighted graphs) gives an inconsistent gap and a
worse decision rule.

Two implementations give identical results:
  * "python" (reference): works with stock PyMatching 2.4.0; applies the
    reweighting by editing edge weights from Python, which makes PyMatching
    rebuild its internal graph for every shot (~1 ms/shot at d = 5-9);
  * "cpp" (fast, ~10x): needs a PyMatching 2.4.0 build with the small patch
    in pymatching_patch/ (MatchingGraph.decode_both_sectors_shared_reweighting),
    which performs the same first pass / reweight / two forced matchings /
    undo inside PyMatching's C++.
The fast path is used automatically when available; set the environment
variable CORRELATED_GAP_IMPL=python (or =cpp) to force one.

Requires PyMatching 2.4.0 (the version used for the paper).
"""

import os
from collections import defaultdict

import numpy as np
import pymatching
import stim

from post_selection import MWPMSoftDecoder

BOUNDARY = -1


def _xor(a, b):
    return a * (1 - b) + b * (1 - a)


def _decomposed_errors(dem):
    """Yield (p, [(detectors, obs_mask), ...]) per DEM error, split at '^' as PyMatching does."""
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        if p == 0:
            continue
        components = [[]]
        for t in inst.targets_copy():
            if t.is_separator():
                components.append([])
            else:
                components[-1].append(t)
        out = []
        for comp in components:
            dets = tuple(sorted(t.val for t in comp if t.is_relative_detector_id()))
            obs = 0
            for t in comp:
                if t.is_logical_observable_id():
                    obs ^= 1 << t.val
            out.append((dets, obs))
        if len(out) == 1 and not out[0][0]:
            continue  # undetectable error, ignored by PyMatching
        yield p, out


def _edge_key(dets):
    return (dets[0], BOUNDARY) if len(dets) == 1 else (dets[0], dets[1])


def implied_weight_rules(dem):
    """PyMatching's correlation rules in Python.

    Returns ({edge: [(partner_edge, implied_weight), ...]}, {edge: {obs masks}}),
    edges keyed (u, v) with u < v, or (u, BOUNDARY).
    """
    joint = defaultdict(dict)
    edge_obs = defaultdict(set)
    for p, comps in _decomposed_errors(dem):
        keys = [_edge_key(d) for d, _ in comps]
        for (_, obs), k in zip(comps, keys):
            edge_obs[k].add(obs)
        if len(keys) > 1:
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    a, b = keys[i], keys[j]
                    joint[a][b] = _xor(joint[a].get(b, 0.0), p)  # sequential, as in C++
                    joint[b][a] = _xor(joint[b].get(a, 0.0), p)  # (aliases when a == b)
        for k in keys:
            joint[k][k] = _xor(joint[k].get(k, 0.0), p)
    rules = {}
    for e, row in joint.items():
        marginal = row[e]
        if marginal == 0:
            continue
        lst = []
        for a, pj in row.items():
            if a != e:
                q = min(0.5, pj / marginal)
                lst.append((a, float(np.log((1 - q) / q))))
        if lst:
            rules[e] = lst
    return rules, edge_obs


class CorrelatedGapDecoder:
    """MWPM with PyMatching's two-pass correlations and a shared-graph complementary gap."""

    def __init__(self, dem: stim.DetectorErrorModel):
        self.num_detectors = dem.num_detectors
        self.virtual = dem.num_detectors
        self.matching = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
        promoted_dem = MWPMSoftDecoder._promote_observable(dem, 0)
        self.promoted = pymatching.Matching.from_detector_error_model(promoted_dem, enable_correlations=False)

        rules, edge_obs = implied_weight_rules(dem)
        to_promoted = {}
        for k, masks in edge_obs.items():
            if len(masks) != 1:
                raise ValueError(f"edge {k} carries inconsistent observable effects {masks}")
            if next(iter(masks)) & 1:
                if k[1] != BOUNDARY:
                    raise ValueError(f"bulk edge {k} flips the observable; promotion needs boundary edges")
                to_promoted[k] = (k[0], self.virtual)
            else:
                to_promoted[k] = k
        self.rules = {e: [(to_promoted[a], w) for a, w in lst] for e, lst in rules.items()}

        self.base = {}
        for u, v, data in self.promoted.edges():
            k = (u, BOUNDARY) if v is None else (min(u, v), max(u, v))
            self.base[k] = (data["weight"], data["error_probability"])
        missing = {a for lst in self.rules.values() for a, _ in lst if a not in self.base}
        if missing:
            raise ValueError(f"{len(missing)} partner edges missing from the promoted graph")

        impl = os.environ.get("CORRELATED_GAP_IMPL", "auto").lower()
        fast = hasattr(self.matching._matching_graph, "decode_both_sectors_shared_reweighting")
        if impl == "cpp" and not fast:
            raise RuntimeError("CORRELATED_GAP_IMPL=cpp needs the patched PyMatching build")
        self.implementation = "cpp" if (fast and impl != "python") else "python"
        if self.implementation == "cpp":
            self.promoted_corr = pymatching.Matching.from_detector_error_model(
                promoted_dem, enable_correlations=True)
            self.boundary_to_virtual = np.zeros(self.num_detectors, dtype=np.uint8)
            for k, masks in edge_obs.items():
                if k[1] == BOUNDARY and next(iter(masks)) & 1:
                    self.boundary_to_virtual[k[0]] = 1

    def _set_weight(self, k, w):
        _, pe = self.base[k]
        if k[1] == BOUNDARY:
            self.promoted.add_boundary_edge(k[0], fault_ids=set(), weight=w, error_probability=pe,
                                            merge_strategy="replace")
        else:
            self.promoted.add_edge(k[0], k[1], fault_ids=set(), weight=w, error_probability=pe,
                                   merge_strategy="replace")

    def _sector_weights(self, syndrome, aug):
        edges = self.matching.decode_to_edges_array(syndrome)  # first pass (no correlations)
        mods = {}
        for u, v in edges:
            u, v = int(u), int(v)
            k = (u, BOUNDARY) if v < 0 else ((v, BOUNDARY) if u < 0 else (min(u, v), max(u, v)))
            for a, w in self.rules.get(k, ()):
                if w < mods.get(a, self.base[a][0]):
                    mods[a] = w
        for a, w in mods.items():
            self._set_weight(a, w)
        aug[:self.num_detectors] = syndrome
        aug[self.num_detectors] = 0
        _, w0 = self.promoted.decode(aug, return_weight=True)
        aug[self.num_detectors] = 1
        _, w1 = self.promoted.decode(aug, return_weight=True)
        for a in mods:
            self._set_weight(a, self.base[a][0])
        return w0, w1

    def decode_with_gap(self, syndromes):
        """Return (predictions uint8, gaps float64 [nats]); gap = w'1 - w'0."""
        if self.implementation == "cpp":
            w = self.promoted_corr._matching_graph.decode_both_sectors_shared_reweighting(
                self.matching._matching_graph, np.ascontiguousarray(syndromes, dtype=np.uint8),
                self.boundary_to_virtual, self.virtual)
            gaps = w[:, 1] - w[:, 0]
            return (gaps < 0).astype(np.uint8), gaps
        return self.decode_with_gap_python(syndromes)

    def decode_with_gap_python(self, syndromes):
        """Reference implementation (stock PyMatching): same result as decode_with_gap."""
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        gaps = np.empty(syndromes.shape[0])
        aug = np.zeros(self.num_detectors + 1, dtype=np.uint8)
        for i, s in enumerate(syndromes):
            w0, w1 = self._sector_weights(s, aug)
            gaps[i] = w1 - w0
        return (gaps < 0).astype(np.uint8), gaps

    def reference_predictions(self, syndromes):
        """PyMatching's own correlated decisions (for consistency checks)."""
        return self.matching.decode_batch(np.asarray(syndromes, dtype=np.uint8),
                                          enable_correlations=True)[:, 0].astype(np.uint8)
