# Soft information from neural-network logits and correlation-aware matching gaps

Code and derived data for the paper *"Soft information from neural-network
logits and correlation-aware matching gaps"* (D. Dentelski).

The paper compares three confidence scores on the same sampled syndromes of
rotated-surface-code memory experiments under uniform circuit-level noise
(10<sup>8</sup> shots per configuration): the logit of a pretrained graph
neural-network (GNN) decoder [Lange et al., Phys. Rev. Research **7**, 023181
(2025)], and the complementary gaps of uncorrelated and correlation-aware
minimum-weight perfect matching (MWPM). It evaluates them for post-selection,
under exchange between the decoders, against the exact posterior at
(d, r) = (3, 3), and as calibrated estimates of the logical failure probability.

## Reproducing the paper

Every figure, every table and every number quoted in the text is rebuilt by

```
pip install -r requirements.txt
python make_all.py
```

which takes a few minutes and needs neither the raw data nor a GPU: all steps
start from the cached derived data in `build/**/derived/` (JSON). The results
are written under the paper's numbering:

| Paper | Output | Scripts |
|---|---|---|
| Fig. 1 | `figures/fig1_postselection.pdf` | `make_postselection.py` |
| Fig. 2 | `figures/fig2_exchange.pdf` | `make_exchange.py`, `make_combined_figures.py` |
| Fig. 3 | `figures/fig3_distribution.pdf` | `make_distribution.py` |
| Fig. 4 | `figures/fig4_exact_d3.pdf` | `make_exact_d3.py` |
| Fig. 5 | `figures/fig5_calibration.pdf` | `make_calibration.py`, `make_combined_figures.py` |
| Fig. 6 | `figures/fig6_calibration_d5_d7.pdf` | `make_calibration.py`, `make_combined_figures.py` |
| Fig. 7 | `figures/fig7_score_pairing.pdf` | `make_score_pairing.py` |
| Fig. 8 | `figures/fig8_local_slopes.pdf` | `make_calibration_extras.py` |
| Fig. 9 | `figures/fig9_alpha_vs_p.pdf` | `make_calibration.py`, `make_combined_figures.py` |
| Table I | `tables/table1_auc.tex` | `make_exchange.py`, `make_robustness.py`, `make_tables.py` |
| Table II | `tables/table2_exact_d3.tex` | `make_exact_d3.py` |
| Table III | `tables/table3_failure_counts.tex` | `make_postselection.py` |
| Table IV | `tables/table4_exchanged_counts.tex` | `make_exchange.py` |
| Table V | `tables/table5_calibration.tex` | `make_calibration.py`, `make_calibration_extras.py`, `make_tables.py` |
| Table VI | `tables/table6_mechanism.tex` | `make_robustness.py` |
| Table VII | `tables/table7_trivial_syndromes.tex` | `make_robustness.py` |
| Table VIII | `tables/table8_fit_window.tex` | `make_calibration.py`, `make_calibration_extras.py`, `make_tables.py` |
| Numbers in the text | `numbers/text_numbers.md` | `make_text_numbers.py`, `compare_gap_constructions.py` (App. B) |

The tables are the `tabular` environments of the paper. Each step logs to
`build/logs/`, and `python make_all.py --steps ...` reruns selected steps.

## Three levels of reproduction

1. **From the derived data** (above): binned counts, curves and fit inputs
   cached in `build/**/derived/`.
2. **From the per-shot data.** The signed confidences of the three decoders on
   all shots of the 20 runs (24 GB) are available on Zenodo,
   [doi:10.5281/zenodo.23817170](https://doi.org/10.5281/zenodo.23817170).
   Place them in `data/` (or point the environment
   variable `GNN_CONFIDENCE_DATA` to them), delete `build/` and run
   `python make_all.py`: every cached result is recomputed from the arrays
   (~2 h on 16 threads). Two steps decode fresh shots instead and also need
   `requirements-data.txt`: the comparison of correlated-gap constructions of
   Appendix B (`compare_gap_constructions.py`, ~1.5 h) and the exact-posterior
   benchmark at d = 3 (`make_exact_d3.py`, ~1 h).
3. **From scratch.** `generate_all_data.py` regenerates the 20 runs (Linux only,
   `requirements-data.txt`, GNN checkpoints from `python fetch_gnn_models.py`;
   optionally the PyMatching patch in `pymatching_patch/` for a 14-18x faster
   correlated gap). The GNN runs on CPU and dominates the cost (days in total).

## Data

`runs.py` lists the 20 runs: (d, r) = (5, 5), (5, 7), (5, 9), (5, 11), (7, 7),
(7, 9), (7, 11), (9, 9) at p = 5 x 10<sup>-3</sup>, and d = r = 5, 7, 9 at
p = 3, 3.5, 4, 4.5 x 10<sup>-3</sup>, each with 10<sup>8</sup> shots. Per run:

- `delta_{decoder}_d{d}_dt{r}_p{p}_shots{N}.npy` for decoder `mwpm`
  (MWPM<sub>uncorr</sub>), `mwpmcorr` (MWPM<sub>corr</sub>) and `gnn`: the signed
  confidence Delta in dB of every shot (float32), positive where the decoder is
  correct, |Delta| = confidence;
- `run_d{d}_dt{r}_p{p}_shots{N}.json`: provenance and consistency checks.

The shots of a run are drawn from one stim detector sampler seeded with
42 + d + r in chunks of 10,000 shots, so every array of a run refers to the
same shots, and a seeded stream is reproduced exactly on Linux with stim 1.16.0
(not on other platforms). Some GNN arrays come from the original GPU inference;
recomputing them on CPU with `gnn_decoder.py` gave identical decisions and logits
within 1.4 x 10<sup>-3</sup> dB on the spot-checked shots (recorded in the run
JSONs). The matching arrays are reproduced bit for bit. A wrongly decided exact tie (gap 0, decision L = 0) is stored as -0.0 and
counted as a failure (`np.signbit`).

## Layout

```
runs.py                      the 20 runs, data and model paths
make_all.py                  builds everything (steps below), collects figures/, tables/, numbers/
make_postselection.py        Fig. 1, Table III
make_exchange.py             exchanged scores: Fig. 2 data, Tables I and IV
make_distribution.py         Fig. 3
make_exact_d3.py             exact posterior at d = 3: Fig. 4, Table II
make_calibration.py          calibration fits: Figs. 5, 6, 9 data, Tables V and VIII
make_calibration_extras.py   Fig. 8, two-parameter fits, goodness of fit per window
make_score_pairing.py        Fig. 7
make_robustness.py           Tables VI and VII, paired AUC differences
make_combined_figures.py     Figs. 2, 5, 6, 9 (both MWPM variants)
make_tables.py               Tables I, V, VIII
make_text_numbers.py         numbers quoted in the text
compare_gap_constructions.py Appendix B: shared-graph vs per-sector correlated gap
figure_style.py              colours and markers shared by all figures
generate_shot_data.py        per-shot data of one run (three decoders, same shots)
generate_all_data.py         all 20 runs
post_selection.py            MWPM complementary gap (uncorrelated)
correlated_gap.py            correlation-aware complementary gap on a shared reweighted graph
validate_correlated_gap.py   checks of correlated_gap.py against PyMatching's correlated decoder
gnn_decoder.py               the pretrained GNN of Lange et al. (per-shot logits)
fetch_gnn_models.py          downloads the GNN checkpoints
pymatching_patch/            optional C++ implementation of the correlated gap
build/                       cached derived data (tracked); logs and intermediate outputs (not tracked)
figures/  tables/  numbers/  the paper's figures, tables and text numbers
```

## Environment

Python 3.14.9 with the versions pinned in `requirements.txt` (rebuilding) and
`requirements-data.txt` (data generation; Linux, used in WSL2).

## License

MIT (`LICENSE`). The GNN architecture in `gnn_decoder.py` is copied from
Lange et al. (MIT) and the optional patch modifies PyMatching (Apache 2.0); see
`THIRD_PARTY_NOTICES.md`. The code of the version submitted in August 2026 is
in the git history (tag `2026-08-resubmission`).
