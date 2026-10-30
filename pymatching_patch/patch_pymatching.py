"""Copy PyMatching 2.4.0 source and add MatchingGraph.decode_both_sectors_shared_reweighting.
Usage: python patch_pymatching.py <src_dir> <dst_dir>"""
import shutil, sys

src, dst = sys.argv[1], sys.argv[2]
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree(src, dst)
path = f"{dst}/src/pymatching/sparse_blossom/driver/user_graph.pybind.cc"
code = open(path).read()

NEW = r'''    g.def(
        "decode_both_sectors_shared_reweighting",
        [](pm::UserGraph &self,
           pm::UserGraph &first_pass_graph,
           const py::array_t<uint8_t> &shots,
           const py::array_t<uint8_t> &boundary_to_virtual,
           uint64_t virtual_node) {
            // Correlation-aware complementary gap on ONE shared reweighted graph.
            //   self:             observable-promoted graph (logical observable -> detector
            //                     `virtual_node`), loaded with enable_correlations=True.
            //   first_pass_graph: the original graph, loaded with enable_correlations=True.
            // Per shot: (1) ordinary first pass on first_pass_graph (the same first pass as
            // PyMatching's correlated decoder); (2) boundary edges that flip the observable are
            // mapped to edges to the virtual node; (3) self is reweighted with those edges
            // (implied weights), decoded with the virtual detector = 0 and = 1, and the
            // reweighting is undone. Returns an (n, 2) array of rescaled weights (w0, w1).
            if (shots.ndim() != 2)
                throw std::invalid_argument("`shots` must be a 2D uint8 array (not bit-packed)");
            size_t n = shots.shape(0), nd = shots.shape(1);
            if (nd != (size_t)boundary_to_virtual.size() || virtual_node < nd)
                throw std::invalid_argument("inconsistent detector counts / virtual node");
            auto &mwpm_first = first_pass_graph.get_mwpm_with_search_graph();
            auto &mwpm = self.get_mwpm_with_search_graph();
            auto s = shots.unchecked<2>();
            auto b2v = boundary_to_virtual.unchecked<1>();
            py::array_t<double> out({(py::ssize_t)n, (py::ssize_t)2});
            auto o = out.mutable_unchecked<2>();
            std::vector<uint64_t> dets;
            std::vector<int64_t> edges;
            std::vector<uint8_t> obs_buf(std::max<size_t>(self.get_num_observables(), 1), 0);
            const double norm = mwpm.flooder.graph.normalising_constant;
            for (size_t i = 0; i < n; i++) {
                dets.clear();
                for (size_t j = 0; j < nd; j++)
                    if (s(i, j))
                        dets.push_back(j);
                edges.clear();
                pm::decode_detection_events_to_edges(mwpm_first, dets, edges);
                for (size_t k = 0; k < edges.size() / 2; k++) {
                    if (edges[2 * k + 1] == -1 && b2v(edges[2 * k]))
                        edges[2 * k + 1] = (int64_t)virtual_node;
                }
                mwpm.flooder.graph.reweight_for_edges(edges);
                mwpm.search_flooder.graph.reweight_for_edges(edges);
                for (int sector = 0; sector < 2; sector++) {
                    if (sector == 1)
                        dets.push_back(virtual_node);
                    pm::total_weight_int weight = 0;
                    pm::decode_detection_events(mwpm, dets, obs_buf.data(), weight, false);
                    o(i, sector) = (double)weight / norm;
                }
                mwpm.flooder.graph.undo_reweights();
                mwpm.search_flooder.graph.undo_reweights();
            }
            return out;
        },
        "first_pass_graph"_a,
        "shots"_a,
        "boundary_to_virtual"_a,
        "virtual_node"_a);
'''
anchor = '    g.def(\n        "decode_to_matched_detection_events_array",'
assert code.count(anchor) == 1, "anchor not found"
code = code.replace(anchor, NEW + anchor, 1)
if "#include <algorithm>" not in code:
    code = code.replace('#include "pybind11/pybind11.h"', '#include <algorithm>\n#include "pybind11/pybind11.h"', 1)
open(path, "w").write(code)
print("patched", path)
