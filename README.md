# Neural network decoder confidence as a learned proxy for the logical gap

Analysis code and derived data for the paper *"Neural network decoder confidence
as a learned proxy for the logical gap"* (D. Dentelski).

The study compares, on the same 10⁸ sampled syndromes per configuration of a
rotated-surface-code memory experiment, the confidence score of a pretrained
graph-neural-network decoder (its pre-sigmoid logit) with the complementary gap
of a minimum-weight-perfect-matching (MWPM) decoder — as rankings for
post-selection, under exchange between the decoders, and as calibrated
estimates of the logical failure probability.

## Reproducing the figures and tables

Every figure and table of the paper regenerates from the derived data included
in `figures3/derived/` — no raw data or GPU required:

```
python make_fig1.py                  # Fig. 1a/1b + failure-counts tables (App. A)
python make_fig2_crossed.py          # Fig. 2a/2b, App. B crossed panel, Tables (AUC, crossed counts)
python make_fig3.py                  # Fig. 3a/3b + App. B distribution panel
python make_fig4.py                  # Fig. 4a/4b, App. B calibration figs + alpha table, App. D window table + alpha-vs-p
python make_appendixC.py             # App. C score-pairing histogram
python make_appendixD_robustness.py  # App. D robustness table + App. C mechanism table
```

Outputs are written to `figures3/` (PDF + PNG) and `figures3/tables/` (LaTeX).
Each script also prints the numerical values quoted in the paper (paired
confidence intervals, decomposition factors, exceedance statistics, fit
diagnostics).

Requirements: Python ≥ 3.9 with `numpy`, `scipy`, `matplotlib`.

## Recomputing from the per-shot data

The primary data are two shot-aligned arrays per configuration —
`delta_mwpm_*.npy` and `delta_gnn_*.npy`, the signed confidences (dB) of each
decoder on the same syndromes, negative on logical failures (~9.6 GB in
total). They are not included in this repository; they are available upon
reasonable request, and an archived copy will accompany the published version
of the paper.

With the arrays placed in `./saved_runs/`, deleting `figures3/derived/` (or
setting `RECOMPUTE = True` in a script) recomputes every analysis from the
per-shot level: exact tie-averaged post-selection, midrank AUCs, paired
bootstrap and DeLong intervals, and binomial maximum-likelihood calibration
fits.

The decoders themselves are not part of this repository: MWPM decoding uses
[PyMatching](https://github.com/oscarhiggott/PyMatching), and the GNN decoders
are the pretrained networks released with M. Lange *et al.*, *Data-driven
decoding of quantum error correcting codes using graph neural networks*,
Phys. Rev. Research **7**, 023181 (2025), available at
[github.com/LangeMoritz/GNN_decoder](https://github.com/LangeMoritz/GNN_decoder).

## Layout

```
make_fig*.py, make_appendix*.py   analysis / figure scripts (self-contained)
figures3/                         figures as in the paper (PDF + PNG)
figures3/tables/                  LaTeX tables as in the paper
figures3/derived/                 cached derived data (JSON) behind every figure and table
```
