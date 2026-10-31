"""
Build every figure, table and quoted number of the paper.

    python make_all.py                           # everything
    python make_all.py --steps calibration tables collect

Each step runs its script in its own process (log: build/logs/<step>.log); a
failing step is reported and the build continues. Intermediate results are
cached as JSON in build/**/derived/ (committed to the repository), so without
the per-shot arrays every step is rebuilt from the cache within minutes; with
the arrays (runs.DATA_DIR) a missing cache is recomputed from them. Deleting
build/ and running with the arrays recomputes everything (~2 h on 16 threads,
plus ~1.5 h for gap_constructions and ~1 h for exact_d3, which decode shots).

The collect step copies the paper's figures and tables, under the paper's
numbering, to
    figures/fig1_postselection.pdf ... figures/fig9_alpha_vs_p.pdf
    tables/table1_auc.tex ... tables/table8_fit_window.tex   (the tabular environments)
    numbers/text_numbers.md                                  (numbers quoted in the text)
"""

import argparse
import datetime
import os
import shutil
import subprocess
import sys
import time

import runs

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = runs.BUILD_DIR
VARIANTS = ("mwpmcorr", "mwpm")   # MWPM_corr (main text), MWPM_uncorr
STEPS = [  # (step, script, variants: None = run once)
    ("postselection", "make_postselection.py", None),           # Fig. 1, Table III
    ("exchange", "make_exchange.py", VARIANTS),                 # Fig. 2 data, Tables I and IV data
    ("distribution", "make_distribution.py", ("mwpmcorr",)),    # Fig. 3
    ("calibration", "make_calibration.py", VARIANTS),           # Figs. 5, 6, 9 data, Tables V and VIII data
    ("calibration_extras", "make_calibration_extras.py", VARIANTS),   # Fig. 8, Table V (2-par.), Table VIII
    ("pairing", "make_score_pairing.py", ("mwpmcorr",)),        # Fig. 7
    ("robustness", "make_robustness.py", VARIANTS),             # Tables VI and VII, Table I bold rule
    ("combined", "make_combined_figures.py", None),             # Figs. 2, 5, 6, 9
    ("exact_d3", "make_exact_d3.py", None),                     # Fig. 4, Table II
    ("high_confidence", "make_high_confidence_failures.py", None),   # App. D: the GNN's deep-cut failures
    ("tables", "make_tables.py", None),                         # Tables I, V, VIII
    ("text", "make_text_numbers.py", None),                     # numbers quoted in the text
    ("gap_constructions", "compare_gap_constructions.py", None),   # Appendix B
]
STEP_NAMES = [s for s, _, _ in STEPS] + ["collect"]
ARGS = {"postselection": ["--out", os.path.join(BUILD, "fig1")],
        "combined": ["--build", BUILD, "--out", os.path.join(BUILD, "combined")],
        "exact_d3": ["--out", os.path.join(BUILD, "exact_d3")],
        "high_confidence": ["--out", os.path.join(BUILD, "high_confidence")],
        "tables": ["--build", BUILD],
        "text": ["--out", os.path.join(BUILD, "text")],
        "gap_constructions": ["--out", os.path.join(BUILD, "gap_constructions")]}

FIGURES = {  # paper figure -> build product
    "fig1_postselection.pdf": "fig1/fig1_postselection.pdf",
    "fig2_exchange.pdf": "combined/fig2_crossed.pdf",
    "fig3_distribution.pdf": "mwpmcorr/fig3_distribution.pdf",
    "fig4_exact_d3.pdf": "exact_d3/exact_d3.pdf",
    "fig5_calibration.pdf": "combined/fig4_calibration.pdf",
    "fig6_calibration_d5_d7.pdf": "combined/appC_calibration_d5_d7.pdf",
    "fig7_score_pairing.pdf": "mwpmcorr/appC_score_pairing.pdf",
    "fig8_local_slopes.pdf": "mwpmcorr/appD_local_slopes.pdf",
    "fig9_alpha_vs_p.pdf": "combined/appD_alpha_vs_p_combined.pdf",
}
TABLES = {  # paper table -> build product(s); Table IV joins the MWPM_corr and MWPM_uncorr blocks
    "table1_auc.tex": ["tables/paper/table1_auc.tex"],
    "table2_exact_d3.tex": ["exact_d3/tables/exact_d3.tex"],
    "table3_failure_counts.tex": ["fig1/tables/fig1a_counts.tex"],
    "table4_exchanged_counts.tex": ["mwpmcorr/tables/appA_crossed_failure_counts.tex",
                                    "mwpm/tables/appA_crossed_failure_counts.tex"],
    "table5_calibration.tex": ["tables/paper/appB_alpha.tex"],
    "table6_mechanism.tex": ["mwpmcorr/tables/appC_mechanism_full.tex"],
    "table7_trivial_syndromes.tex": ["mwpmcorr/tables/appD_trivial_robustness.tex"],
    "table8_fit_window.tex": ["tables/paper/appD_window.tex"],
}


def run(name, script, args, status):
    os.makedirs(os.path.join(BUILD, "logs"), exist_ok=True)
    log = os.path.join(BUILD, "logs", f"{name}.log")
    t0 = time.perf_counter()
    with open(log, "w") as f:
        rc = subprocess.run([sys.executable, "-W", "ignore", os.path.join(HERE, script), *args],
                            cwd=HERE, stdout=f, stderr=subprocess.STDOUT,
                            env={**os.environ, "MPLBACKEND": "Agg"}).returncode
    minutes = (time.perf_counter() - t0) / 60
    status.append((name, "ok" if rc == 0 else f"FAILED (exit {rc}, see build/logs/{name}.log)", minutes))
    print(f"[{datetime.datetime.now():%H:%M}] {name}: {'ok' if rc == 0 else 'FAILED'} ({minutes:.1f} min)",
          flush=True)


def tabular(path):
    """The tabular environment of a generated LaTeX table, as a list of lines."""
    lines = open(path).read().split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(r"\begin{tabular}"))
    end = next(i for i, line in enumerate(lines) if line.startswith(r"\end{tabular}"))
    return lines[start:end + 1]


def collect():
    """Copy the paper's figures and tables (paper numbering) to figures/, tables/ and numbers/."""
    for sub in ("figures", "tables", "numbers"):
        os.makedirs(os.path.join(HERE, sub), exist_ok=True)
    for dst, src in FIGURES.items():
        shutil.copyfile(os.path.join(BUILD, src), os.path.join(HERE, "figures", dst))
    for dst, srcs in TABLES.items():
        lines = tabular(os.path.join(BUILD, srcs[0]))
        for src in srcs[1:]:   # append the body rows of the other block after an \hline
            extra = tabular(os.path.join(BUILD, src))
            body = extra[extra.index(r"\midrule") + 1:extra.index(r"\bottomrule")]
            end = lines.index(r"\bottomrule")
            lines = lines[:end] + [r"\hline"] + body + lines[end:]
        open(os.path.join(HERE, "tables", dst), "w").write("\n".join(lines) + "\n")
    shutil.copyfile(os.path.join(BUILD, "text", "text_numbers.md"), os.path.join(HERE, "numbers", "text_numbers.md"))
    print("collected -> figures/, tables/, numbers/", flush=True)


def data_inventory():
    lines = ["| run | " + " | ".join(runs.DECODERS) + " |", "|---" * (len(runs.DECODERS) + 1) + "|"]
    for d, r, p, s in runs.runs():
        lines.append(f"| {runs.tag(d, r, p, s)} | " + " | ".join(
            "yes" if os.path.isfile(runs.array_path(n, d, r, p, s)) else "--" for n in runs.DECODERS) + " |")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=STEP_NAMES, default=STEP_NAMES)
    a = ap.parse_args()
    os.makedirs(BUILD, exist_ok=True)
    status = []
    for step, script, variants in STEPS:
        if step not in a.steps:
            continue
        if variants is None:
            run(step, script, ARGS.get(step, []), status)
        else:
            for v in variants:
                run(f"{step}_{v}", script, ["--mwpm", v, "--out", os.path.join(BUILD, v)], status)
    failed = [n for n, s, _ in status if s != "ok"]
    if "collect" in a.steps:
        if failed:
            print(f"collect skipped: {', '.join(failed)} failed", flush=True)
        else:
            collect()
    lines = [f"# Build status ({datetime.datetime.now():%Y-%m-%d %H:%M})", "", "| step | status | minutes |",
             "|---|---|---|"] + [f"| {n} | {s} | {m:.1f} |" for n, s, m in status]
    lines += ["", f"## Per-shot arrays in {runs.DATA_DIR}", ""] + data_inventory()
    os.makedirs(os.path.join(BUILD, "logs"), exist_ok=True)
    open(os.path.join(BUILD, "logs", "status.md"), "w").write("\n".join(lines) + "\n")
    print("done" + (f"; FAILED: {', '.join(failed)}" if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
