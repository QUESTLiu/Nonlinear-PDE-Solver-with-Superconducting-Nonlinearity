# Native nonlinear interactions versus programmable emulation in hybrid oscillator–qubit simulation

This repository contains the manuscript, the code and numerical records behind its numbers, tables and figures, and a Lean 4 formalization of several of its statements. Code and data are grouped by benchmark, and each `<topic>_code/` folder writes to the matching `<topic>_data/` folder.

| Folder | Content |
|---|---|
| `paper/` | Manuscript source, bibliography, figures, compiled PDF, and the version with changes marked in blue |
| `sine_gordon_code/`, `sine_gordon_data/` | Sine-Gordon lattice, Sec. III A and Appendix F. The library in `sine_gordon_code/` also serves the double sine-Gordon lattice |
| `duffing_code/`, `duffing_data/` | Duffing oscillator, Sec. III B and Appendix H |
| `double_sine_gordon_code/`, `double_sine_gordon_data/` | Double sine-Gordon lattice, Sec. III C and Appendices I, J and K |
| `common_code/`, `common_data/` | Pipeline infrastructure shared by the benchmarks, the figure script, and records that span several benchmarks |
| `lean_proof/` | Lean 4 formalization of the statements in Appendix L |
| `legacy/` | Earlier code by other contributors that the manuscript does not use |

## Environment

The records were produced with Python 3.13.0, NumPy 2.3.4, SciPy 1.16.3 and Matplotlib 3.10.7 (`requirements.txt`). Run all Python commands from the repository root, scripts as `python <folder>/<script>.py` and pipeline modules as `python -m <folder>.<module>`. The manuscript compiles with `tectonic main.tex` in `paper/`, and the Lean project builds with `lake exe cache get` and `lake build` in `lean_proof/`.

## Where each result comes from

The pipeline drivers run the batches listed in the manifests of each data folder, for example `python -m common_code.run_batch duffing_data/manifests/duffing_padding_manifest.json --batch B3-D`, which prints the commands and runs them with `--execute`. The pipeline records the git commit and a hash of any uncommitted changes in its outputs, so run it from a git checkout with at least one commit. The archived records name the commit and interpreter of the development repository.

### Sine-Gordon lattice

| Code | Data | Manuscript |
|---|---|---|
| `sine_gordon_code/sglib.py`, `common.py`, `native_correction.py` | none (operators, gates and diagnostics) | used by all lattice results |
| `sine_gordon_code/run_classical.py` (experiment E3) | `sine_gordon_data/formal/E3/`, `sine_gordon_data/stage1/E3/` | Fig. 2, mean-field window in Sec. III A |
| `sine_gordon_code/run_polydeg.py` | `sine_gordon_data/formal/polynomial/` | Fig. 7, Appendix F |

### Duffing oscillator

| Code | Data | Manuscript |
|---|---|---|
| `duffing_code/duffing_padded_hybrid_validation.py`, `native_comparison.py` | none (model, compiler, padded validation, Kerr replacement) | used by all Duffing results |
| `common_code/run_pilot.py`, subcommands `e1` and `e1-formal` (experiment E1) | `duffing_data/stage1/E1/`, `duffing_data/formal/E1/`, `E1_retry/` | Fig. 3 |
| `common_code/run_pilot.py`, subcommand `e2` (development runs of E2) | `duffing_data/stage1/E2/` | inputs to the formal E2 runs |
| `duffing_code/run_duffing_formal.py` (experiment E2), `verify_e2_action.py` | `duffing_data/formal/E2/` | Figs. 4 and 8 |
| `duffing_code/run_duffing_refinement.py` | `duffing_data/duffing_refinement/` | Figs. 4 and 8, Table II, Appendix H |
| `duffing_code/run_duffing_padding.py` | `duffing_data/duffing_padding/` | Table II |
| `duffing_code/duffing_amplitude_scan.py` | `duffing_data/duffing_amplitude_scan.json` | Fig. 3, numbers in Sec. III B |
| `duffing_code/kerr_apriori_bound.py` | `duffing_data/kerr_apriori_bound.json` | a priori Kerr bound in Sec. III B |

### Double sine-Gordon lattice

The pipeline modules named `*_sg_*` and the data folders `sg*` belong to this lattice at λ₂ = 0.3 (experiment E4).

| Code | Data | Manuscript |
|---|---|---|
| `common_code/run_pilot.py`, subcommands `e4-reference`, `e4-refinement`, `e4-reference128` and `e4-compiled` | `double_sine_gordon_data/stage1/E4_reference/`, `stage1_continuation/`, `sg128/reference/`, `sg128/compiled*/` | Fig. 5, Table VI |
| `double_sine_gordon_code/run_sg_padding160.py` | `double_sine_gordon_data/sg160/` | Table VI, solver margins of the step scans |
| `double_sine_gordon_code/run_sg_formal.py` | `double_sine_gordon_data/sg_formal/` | Fig. 9 |
| `double_sine_gordon_code/run_sg_refinement.py`, `run_snapd.py`, `run_snapd_end_to_end.py` | `double_sine_gordon_data/sg_refinement/`, `sg_size/`, `sg_continuation/` | Table V, Appendix I |
| `double_sine_gordon_code/run_sg_independent.py` | `double_sine_gordon_data/sg_independent/` | second-preparation check of the r = 10 and 20 circuits |
| `double_sine_gordon_code/run_sg_convention.py`, `check_sg_split.py`, `check_sg_reference_grid.py`, `probe_sg_stationary.py`, `run_scaling.py` | `double_sine_gordon_data/sg128/`, `stage1/B0/`, `common_data/records/` | convention, splitting and reference-grid checks |
| `double_sine_gordon_code/dsg_step_scan.py`, `dsg_step_scan_lambda.py` | `double_sine_gordon_data/dsg_step_scan*.json`, `dsg_step_scan_parts*/` | Table III, Figs. 5 and 6 |
| `double_sine_gordon_code/dsg_chain.py` | `double_sine_gordon_data/dsg_chain_*` | three-site rows of Table III, propagator check in Appendix G |
| `double_sine_gordon_code/dsg_boundary_check.py` | `double_sine_gordon_data/dsg_boundary_check_*.json` | λ₂ = 0.6 entries of Tables III and VII |
| `double_sine_gordon_code/dsg_protocol_checks.py` | `double_sine_gordon_data/dsg_common_times*`, `dsg_gauss_factor_check.json` | Appendices J and K |
| `double_sine_gordon_code/dsg_thresholds.py` | `double_sine_gordon_data/dsg_thresholds.json` | Table VII, Fig. 6(c) |
| `double_sine_gordon_code/splitting_baselines.py`, `splitting_sweep.py` | `double_sine_gordon_data/splitting_*.json` | Fig. 6(b) and 6(c) |
| `double_sine_gordon_code/dsg_curvature.py`, `dsg_fock_tail.py` | `double_sine_gordon_data/dsg_curvature.json`, `dsg_fock_tail.json` | Sec. III C, Appendix G |
| `double_sine_gordon_code/trig_counts.py` | primitive counts inside the scan records | Tables III and VII, Appendix J |

The double sine-Gordon scans cache reference trajectories in the directories named by `DSG_SCAN_CACHE` and `DSG_CHAIN_CACHE` (default `/tmp/dsg_scan_cache` and `/tmp/dsg_chain_cache`, about 9 GB for three sites). Each script's docstring gives its commands and cost.

### Shared

| Code | Data | Manuscript |
|---|---|---|
| `common_code/run_pilot.py`, `metrics.py`, `records.py`, `run_batch.py` | `common_data/manifests/`, `common_data/formal/`, `common_data/stage1/` (batch execution records) | drivers and metrics for E1 to E4 |
| `common_code/run_e0.py`, `test_e0.py` | `common_data/records/e0.json`, `e0.log` | small operator checks |
| `common_code/plot_results.py` | `common_data/analysis_summary.json` | Tables II and V, Fig. 9 |
| `common_code/derived_numbers.py` | `common_data/derived_numbers.json` | numbers in Secs. III A and III B and Appendix F |
| `common_code/class_diagnostics.py` | `common_data/class_diagnostics.json` | Table IV |
| `common_code/make_figures.py` | `paper/figures/*.pdf` | Figs. 2–9 |
| `lean_proof/` | `common_data/records/lean-build.log`, `lean-axioms.log` | Appendix L |

`python common_code/make_figures.py` redraws Figs. 2–9 from the data folders in a few seconds. Fig. 1 is drawn with TikZ, and `tectonic fig_overview.tex` in `paper/figures/` rebuilds it. `paper/mark_changes.py` produces `paper/main_marked.tex` from `paper/main.tex` and the earlier draft.
