# Native nonlinear interactions versus programmable emulation in hybrid oscillator–qubit simulation

This repository contains the manuscript of the same title, the code and numerical records behind its numbers, tables and figures, and a Lean 4 formalization of several of its finite-dimensional statements.

## Contents

| Path | Content |
|---|---|
| `paper/` | Manuscript source `main.tex`, bibliography `references.bib`, figures and the compiled `main.pdf` |
| `paper/scripts/`, `paper/data/` | Scripts that scan and check the double sine-Gordon circuits, derive further numbers and draw the figures, and the records they write |
| `pra/` | Simulation pipeline for the sine-Gordon, Duffing and double sine-Gordon comparisons, with its protocols, run manifests, records and results (see `pra/README.md`) |
| `sine_gordon/`, `duffing/` | Model operators, gates, compilers and metrics used by `pra/` and `paper/scripts/` |
| `cvdv/` | Lean 4 formalization of the statements listed in Appendix L (see `cvdv/README.md`) |

## Environment

The records were produced with Python 3.13.0, NumPy 2.3.4, SciPy 1.16.3 and Matplotlib 3.10.7, as declared in `.python-version` and `requirements.txt`. One way to set up this environment is

```sh
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the Python commands below from the repository root with this interpreter.

## Manuscript

The manuscript uses REVTeX 4.2 and compiles with Tectonic,

```sh
cd paper
tectonic main.tex
```

or with `latexmk -pdf main.tex` in a TeX Live installation. Fig. 1 is a TikZ drawing whose source is `paper/figures/fig_overview.tex`.

## Figures and numbers

`python paper/scripts/make_figures.py` redraws Figs. 2–9 from the records in `pra/results/` and `paper/data/` in a few seconds, without running simulations.

The scripts below write the records in `paper/data/`. The docstring of each script gives its commands. Times are approximate and refer to one workstation.

| Script | Output in `paper/data/` | Used for | Time |
|---|---|---|---|
| `duffing_amplitude_scan.py` | `duffing_amplitude_scan.json` | Fig. 3 | seconds |
| `kerr_apriori_bound.py` | `kerr_apriori_bound.json` | a priori Kerr bound, Sec. III B | seconds |
| `derived_numbers.py` | `derived_numbers.json` | numbers in Secs. III A and III B and Appendix F, computed from `pra/results/` | seconds |
| `class_diagnostics.py` | `class_diagnostics.json` | Table IV | 10 s |
| `splitting_baselines.py` | `splitting_baselines.json` | Fig. 6(b) | 2 min |
| `splitting_sweep.py` | `splitting_sweep.json` | Fig. 6(c) | 2 min |
| `dsg_step_scan.py`, `dsg_step_scan_lambda.py` | `dsg_step_scan*.json`, `dsg_step_scan_parts*/` | Table III, Figs. 5 and 6 | 40 min for the three values of λ₂, run as parallel processes |
| `dsg_chain.py` | `dsg_chain_N3_cut128_160.json`, `dsg_chain_parts_*/` | three-site rows of Table III | several hours, about 1 h for each three-site hybrid circuit |
| `dsg_boundary_check.py` | `dsg_boundary_check_lambda0.6*.json` | λ₂ = 0.6 entry of Table III | 5 min |
| `dsg_protocol_checks.py` | `dsg_common_times.json`, `dsg_common_times_parts/`, `dsg_gauss_factor_check.json` | Appendices J and K | up to 4 min per circuit |
| `dsg_thresholds.py` | `dsg_thresholds.json` | Table VII, Fig. 6(c) | under 1 s |
| `dsg_curvature.py` | `dsg_curvature.json` | Sec. III C | 1 min |
| `dsg_fock_tail.py` | `dsg_fock_tail.json` | Sec. III C, Appendix G | 1 min |
| `trig_counts.py` | primitive counts inside the scan records | Tables III and VII, Appendix J | seconds |

The scans cache reference trajectories in the directories named by the environment variables `DSG_SCAN_CACHE` and `DSG_CHAIN_CACHE`, which default to `/tmp/dsg_scan_cache` and `/tmp/dsg_chain_cache`. The three-site cache takes about 9 GB, and `dsg_fock_tail.py` reads both caches. `python paper/scripts/trig_counts.py verify` checks the rewritten trigonometric gate sequences against the exact gates and the stored counts in a few seconds.

## Simulation pipeline

`pra/results/` holds the records of the sine-Gordon, Duffing and archived double sine-Gordon runs, and `pra/README.md` describes the pipeline, its protocols and how to rerun a batch. `python -m pra.plot_results` rewrites `pra/results/analysis_summary.json`, which the scripts in `paper/scripts/` read, and `python -m unittest pra.test_e0` runs the small operator checks of the pipeline. The pipeline writes the git commit and diff of the checkout into its records, so run it from a git repository.

## Lean formalization

With elan installed, build the formalization from `cvdv/`,

```sh
cd cvdv
lake exe cache get
lake build
```

where `lake exe cache get` downloads prebuilt Mathlib files. `pra/records/lean-build.log` and `pra/records/lean-axioms.log` record a full build and an axiom audit.

## Data size

`pra/results/` takes 1.2 GB, mostly state trajectories in `.npz` files. The rest of the repository takes a few megabytes.
