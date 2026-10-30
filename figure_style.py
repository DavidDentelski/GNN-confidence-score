"""
Shared decoder colours, markers, line styles and labels for the figures of the revision.

Colour identifies the decoder-confidence pair, the same in every figure:

    mwpm      MWPM_uncorr + g_uncorr   gray    dotted, triangles
    mwpmcorr  MWPM_corr   + g_corr     blue    solid,  squares
    gnn       GNN         + g_GNN      red     dashed, circles

Line style and marker identify the pair as well, so identity never rests on colour alone
(colour-vision deficiency, grayscale print). The three colours keep their pairwise
separation under simulated protan/deutan/tritan vision and are visible on a white
background; the gray is deliberately unsaturated (the uncorrelated decoder is the
ablation baseline). Figures in which line style encodes something else (e.g. the
ranking score in Fig. 2) take colour and marker only.
"""

COLOR = {"mwpm": "#8a8985", "mwpmcorr": "#2a78d6", "gnn": "#e34948"}
MARKER = {"mwpm": "^", "mwpmcorr": "s", "gnn": "o"}
LINESTYLE = {"mwpm": ":", "mwpmcorr": "-", "gnn": "--"}
NAME = {"mwpm": r"MWPM$_{\rm uncorr}$", "mwpmcorr": r"MWPM$_{\rm corr}$", "gnn": "GNN"}
SCORE = {"mwpm": r"g_{\rm uncorr}", "mwpmcorr": r"g_{\rm corr}", "gnn": r"g_{\rm GNN}"}


def style(name):
    """matplotlib kwargs (colour, line style, marker) of decoder `name`."""
    return dict(color=COLOR[name], linestyle=LINESTYLE[name], marker=MARKER[name])


def pair_label(name):
    """Legend label of the native pair, e.g. 'MWPM$_{\\rm corr}$ + $g_{\\rm corr}$'."""
    return f"{NAME[name]} + ${SCORE[name]}$"
