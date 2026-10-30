# Optional PyMatching patch (fast correlated gap)

`correlated_gap.py` computes the correlation-aware complementary gap on one
shared reweighted graph. Its reference implementation works with stock
PyMatching 2.4.0 but edits edge weights from Python, which makes PyMatching
rebuild its internal graph for every shot (~1 ms/shot at d = 5-9).

`patch_pymatching.py` adds one method to PyMatching 2.4.0,
`MatchingGraph.decode_both_sectors_shared_reweighting`, which performs the same
steps inside PyMatching's C++ using only existing internals:
first pass (`decode_detection_events_to_edges`) -> `reweight_for_edges` ->
two forced matchings (`decode_detection_events`) -> `undo_reweights`.
Results are bit-identical to the Python reference (verified on fresh shots at
d = 3-9 and on the saved arrays), and 14-18x faster. `correlated_gap.py` uses it
automatically when present (force with `CORRELATED_GAP_IMPL=python|cpp`).

## Build (Linux)

Requirements: a C++17 compiler, CMake, a Python with development headers
(e.g. a `uv`-managed standalone Python), and internet access (the PyMatching
build fetches stim and googletest at pinned commits).

```bash
pip download --no-deps --no-binary :all: pymatching==2.4.0   # or get the sdist from PyPI
tar xzf pymatching-2.4.0.tar.gz
python patch_pymatching.py pymatching-2.4.0 pymatching-2.4.0-sharedgap
pip install ./pymatching-2.4.0-sharedgap
python -c "import pymatching; print(hasattr(pymatching.Matching()._matching_graph, 'decode_both_sectors_shared_reweighting'))"
```

Environment used for the paper: Python 3.14.9, stim 1.16.0, numpy 2.5.4,
torch 2.14.1 (CPU), torch_geometric 2.8.1, PyMatching 2.4.0 (+ this patch).
