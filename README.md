# Native interactions and control costs in oscillator–qubit quantum simulation

This repository contains the code and numerical records behind the manuscript's numbers, tables and figures, and Lean 4 experiments on selected mathematical statements. Code and data are grouped by benchmark, and each `<topic>_code/` folder writes to the matching `<topic>_data/` folder.

| Folder | Content |
|---|---|
| `sine_gordon_code/`, `sine_gordon_data/` | Sine-Gordon lattice, Sec. III A. The library in `sine_gordon_code/` also serves the double sine-Gordon lattice |
| `duffing_code/`, `duffing_data/` | Duffing oscillator, Sec. III B and Appendix E |
| `double_sine_gordon_code/`, `double_sine_gordon_data/` | Double sine-Gordon lattice, Sec. III C and Appendices F, G and H |
| `common_code/`, `common_data/` | Pipeline infrastructure shared by the benchmarks, the figure script, and records that span several benchmarks |
| `lean_proof/` | Optional Lean 4 formalization experiments |
| `legacy/` | Earlier implementations, including the Kerr-lattice calculation used in Appendix I |

## Environment

The records were produced with Python 3.13.0, NumPy 2.3.4, SciPy 1.16.3 and Matplotlib 3.10.7 (`requirements.txt`). Run all Python commands from the repository root, scripts as `python <folder>/<script>.py` and pipeline modules as `python -m <folder>.<module>`. The optional Lean project builds with `lake exe cache get` and `lake build` in `lean_proof/`.

## Gaussian circuit evidence

The Gaussian route evaluates the circuit that its primitive count describes. Each complete readout has two boundary half intervals and full interior Gaussian intervals. Commuting position phases merge by adding their quadratic generators. The finite shear matrices do not in general satisfy the canonical Gaussian group law.

`common_data/manifests/gaussian_revision_candidates.json` fixes the sources and candidate dependencies for the resource tables, tighter thresholds, second preparation, common readout times, and duration curves. Records in `common_data/records/gaussian_revision/` compare the merged circuit with the archived repeated-half-step sequence at both cutoffs. They contain measured state separations, observable changes, new padding differences, boundary populations, and pass/fail intervals. Reference and solver margins are empirical numerical estimates.

The original scan JSON files describe the repeated-half-step sequence. `double_sine_gordon_code/gaussian_evidence.py` joins those records to the merged-circuit evidence for figure generation and selection. Its scalar Gaussian error fields are upper bounds, accompanied by lower bounds and an evidence label. It rejects missing evidence and unresolved selection decisions. The hybrid and kinetic–potential records keep their original meaning. Fresh Gaussian scans and merge outputs carry a `_merged` filename suffix, and merge commands exclude the unmerged Gaussian rows. The published figures read the archived sources plus the comparison evidence rather than these fresh scan outputs.

Run the small independent checks and regenerate the affected summaries from the repository root:

```sh
python -m unittest common_code.test_e0
python double_sine_gordon_code/dsg_thresholds.py
python double_sine_gordon_code/dsg_control_tradeoffs.py
python common_code/make_figures.py
```

The last three commands read stored data and perform no propagation or fitting. To replay the Gaussian comparison, use the bounded manifests `gaussian_revision_two_site.json`, `gaussian_revision_chain.json`, and `gaussian_revision_common.json` under `common_data/manifests/`. Copy a manifest, set `output_root` to a new run directory, keep `execution_record` as a filename inside that directory, and append `--output-dir <new-run-directory>` to the comparison command using the same directory. The job log, execution receipt, and candidate records then all go into the new location. Each replay needs its own run directory. The runner refuses to overwrite an existing execution receipt or job log. The archived runs use one BLAS/OpenMP thread and a 4 GiB process cap. The `references` and `scan` commands of the original simulation drivers can rebuild target trajectories and have substantially larger costs.

## Where each result comes from

The pipeline drivers run the batches listed in the manifests of each data folder, for example `python -m common_code.run_batch duffing_data/manifests/duffing_padding_manifest.json --batch B3-D`, which prints the commands and runs them with `--execute`. The output records include the interpreter and library versions.

### Sine-Gordon lattice

| Code | Data | Manuscript |
|---|---|---|
| `sine_gordon_code/sglib.py`, `common.py`, `native_correction.py` | none (operators, gates and diagnostics) | used by all lattice results |
| `sine_gordon_code/run_classical.py` (experiment E3) | `sine_gordon_data/formal/E3/`, `sine_gordon_data/stage1/E3/` | Fig. 2, mean-field window in Sec. III A |
| `sine_gordon_code/run_polydeg.py` | `sine_gordon_data/formal/polynomial/` | Auxiliary cosine-polynomial comparison |

### Duffing oscillator

| Code | Data | Manuscript |
|---|---|---|
| `duffing_code/duffing_padded_hybrid_validation.py`, `native_comparison.py` | none (model, compiler, padded validation, Kerr replacement) | used by all Duffing results |
| `common_code/run_pilot.py`, subcommands `e1` and `e1-formal` (experiment E1) | `duffing_data/stage1/E1/`, `duffing_data/formal/E1/`, `E1_retry/` | Fig. 3 |
| `common_code/run_pilot.py`, subcommand `e2` (development runs of E2) | `duffing_data/stage1/E2/` | inputs to the formal E2 runs |
| `duffing_code/run_duffing_formal.py` (experiment E2), `verify_e2_action.py` | `duffing_data/formal/E2/` | Figs. 4 and 8 |
| `duffing_code/run_duffing_refinement.py` | `duffing_data/duffing_refinement/` | Figs. 4 and 8, Table II, Appendix E |
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
| `double_sine_gordon_code/run_sg_refinement.py`, `run_snapd.py`, `run_snapd_end_to_end.py` | `double_sine_gordon_data/sg_refinement/`, `sg_size/`, `sg_continuation/` | Table V, Appendix F |
| `double_sine_gordon_code/run_sg_independent.py` | `double_sine_gordon_data/sg_independent/` | second-preparation check of the r = 10 and 20 circuits |
| `double_sine_gordon_code/run_sg_convention.py`, `check_sg_split.py`, `check_sg_reference_grid.py`, `probe_sg_stationary.py`, `run_scaling.py` | `double_sine_gordon_data/sg128/`, `stage1/B0/`, `common_data/records/` | convention, splitting and reference-grid checks |
| `double_sine_gordon_code/dsg_step_scan.py`, `dsg_step_scan_lambda.py` | `double_sine_gordon_data/dsg_step_scan*.json`, `dsg_step_scan_parts*/` | Table III, Figs. 5 and 6 |
| `double_sine_gordon_code/dsg_chain.py` | `double_sine_gordon_data/dsg_chain_*` | three-site rows of Table III, propagator check in Appendix D |
| `double_sine_gordon_code/dsg_boundary_check.py` | `double_sine_gordon_data/dsg_boundary_check_*.json` | λ₂ = 0.6 entries of Tables III and VII |
| `double_sine_gordon_code/dsg_protocol_checks.py` | `double_sine_gordon_data/dsg_common_times*`, `dsg_gauss_factor_check.json` | Appendices G and H |
| `double_sine_gordon_code/dsg_thresholds.py` | `double_sine_gordon_data/dsg_thresholds.json` | Table VII, Fig. 6(d) |
| `double_sine_gordon_code/dsg_control_tradeoffs.py` | `double_sine_gordon_data/dsg_control_tradeoffs.json` | Fig. 7, circuit selection with gate durations and native switching overhead in Sec. IV B |
| `double_sine_gordon_code/splitting_baselines.py`, `splitting_sweep.py` | `double_sine_gordon_data/splitting_*.json` | Fig. 6(b) and 6(c) |
| `double_sine_gordon_code/dsg_curvature.py`, `dsg_fock_tail.py` | `double_sine_gordon_data/dsg_curvature.json`, `dsg_fock_tail.json` | Sec. III C, Appendix D |
| `double_sine_gordon_code/trig_counts.py` | primitive counts inside the scan records | Tables III and VII, Appendix G |
| `double_sine_gordon_code/joint_phase.py`, `double_sine_gordon_code/check_joint_phase.py` | `double_sine_gordon_data/joint_phase_check.json` | Joint Fourier phase compilation in Appendix G |
| `double_sine_gordon_code/dsg_grounding_check.py` | `double_sine_gordon_data/dsg_grounding_check.json` | Grounded-target comparison in Sec. IV C and Appendix G |

The double sine-Gordon scans cache reference trajectories in the directories named by `DSG_SCAN_CACHE` and `DSG_CHAIN_CACHE` (default `/tmp/dsg_scan_cache` and `/tmp/dsg_chain_cache`, about 9 GB for three sites). Each script's docstring gives its commands and cost.

The joint phase comparison uses two sites at λ₂ = 0.3, outer-step counts 6–8, Fourier degrees 1–10, and fixed or alternating completion signs. It counts recovered controlled-displacement and qubit-rotation sequences after combining adjacent qubit rotations. The record contains 120 compiler configurations and ten full circuit evaluations at cutoffs 128/160. All eight programmable candidates below 209 primitives fail the original acceptance criteria. The two representative accurate circuits use 338 hybrid and 810 programmable primitives and pass those criteria. These results concern the implemented Fourier truncation and spectral completion. They do not establish optimality over quantum signal processing constructions.

To repeat this bounded comparison without replacing its evidence, choose a new output path:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python double_sine_gordon_code/check_joint_phase.py --output /tmp/joint_phase_replay.json
```

The runner refuses an existing output file and enforces a 600-second wall limit. The recorded run took 170 seconds in the environment above. It checks coefficient evaluation, pulse recovery, whole-circuit rotation merging, the matrix-free native propagator, and the quadratic-grounding shear identity before evaluating trajectories. A timed-out run is marked incomplete. Source filenames and the environment are stored with the results.

The grounded-target check adds the same onsite quadratic term to the native and target Hamiltonians at `g_sh = 0.1` and λ₂ = 0.3. It scans 30 hybrid and Gaussian-block circuits with `r = 4,…,9`, the original block-order choices, and cutoffs 128/160. The preparation, basis frequency, horizon, and acceptance thresholds match the primary comparison. A separate DOP853 solution supplies the grounded target's empirical solver margin. The lowest accepted counts in this scan remain 64 and 209, with state errors 0.0270 and 0.0294. The record includes every candidate and its numerical margins.

To repeat the grounding check, choose a new output path. The runner refuses existing output files and has a 1200-second wall limit:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python double_sine_gordon_code/dsg_grounding_check.py --output /tmp/dsg_grounding_replay.json
```

The duration analysis in Fig. 7 is post-processing of 34 evaluated two-site candidates in `dsg_step_scan.json` at λ₂ = 0.3, preparation (1, 0), and s = 1. It selects circuits using the stated numerical acceptance criteria and an additional amplitude-distance allowance, then assigns gate-type durations and an overhead for each native interval. `dsg_control_tradeoffs.json` records the candidates, the selections at allowances 0, 0.005, 0.010, and 0.020, and the duration boundaries. The inputs are the stored ideal-circuit results.

### Analytic correction in a Kerr lattice

Appendix I uses `legacy/nlse_fock_cutoff_hybrid.py` to compare a native Kerr lattice with a target containing the local interaction W N(N−1)(N−2)/6. `common_code/check_number_diagonal.py` reruns one parameter point, checks the analytic SNAP phases independently using integer occupations, and halves the time step at the largest tested cutoff.

Run from the repository root in the environment above:

```bash
python common_code/check_number_diagonal.py
```

The calculation uses three sites with open boundaries, ω = 0, J = 0.5, U = 0.8, W = 0.3, T = 4, local cutoffs 6, 8, 10, and 12, and reference cutoff 16. The initial coherent wavepacket has amplitude 1.25, width 0.9, and phase gradient 0.6. The symmetric split uses 160 steps and saves 81 equally spaced times. The step refinement uses cutoff 12 and 320 steps on the same observation grid. The runner has a 120-second wall-time limit and monitors the legacy subprocess against a 1 GiB RSS limit.

Outputs are grouped in `common_data/number_diagonal_check/`:

| File | Content |
|---|---|
| `summary.csv`, `trace_cutoff_*_W_0.3.npz` | Parameters and maximum infidelities (CSV), with the corresponding fidelity time series (NPZ) against the cutoff-16 and same-cutoff targets |
| `reference_W_0.3.npz` | Observation times and final reference state |
| `refinement_cutoff12_steps320.npz` | Fidelity to the same-cutoff target after step refinement |
| `validation.json` | Parameters, metric definitions, commands, interpreter and library versions, resource measurements, phase checks, and refinement results |
| `artifact_audit.json` | Agreement between time series and summary values, plus output file sizes |
| `legacy_run.log` | Output of the underlying propagation command |
| `plots/` | Diagnostic plots generated from the cutoff comparison |

The reported error is the maximum infidelity, 1 − |overlap|², over the 81 stored times. At cutoff 12 the native and corrected errors against the cutoff-16 reference are 0.2029 and 2.99 × 10⁻⁷. The corrected error against the same-cutoff target is 6.85 × 10⁻⁹ and falls to 4.27 × 10⁻¹⁰ at 320 steps. These records support the finite Kerr-lattice correction example. The phase identity is analytic, and the numerical comparison measures cutoff differences and splitting error.

### Shared

| Code | Data | Manuscript |
|---|---|---|
| `common_code/run_pilot.py`, `metrics.py`, `records.py`, `run_batch.py` | `common_data/manifests/`, `common_data/formal/`, `common_data/stage1/` (batch execution records) | drivers and metrics for E1 to E4 |
| `common_code/run_e0.py`, `test_e0.py` | `common_data/records/e0.json`, `e0.log` | small operator checks |
| `common_code/plot_results.py` | `common_data/analysis_summary.json` | Tables II and V, Fig. 9 |
| `common_code/derived_numbers.py` | `common_data/derived_numbers.json` | numbers in Secs. III A and III B and the auxiliary polynomial comparison |
| `common_code/class_diagnostics.py` | `common_data/class_diagnostics.json` | Table IV |
| `common_code/make_figures.py` | `paper/figures/*.pdf` | Figs. 2–6 and 8–9, plus the auxiliary polynomial plot |
| `legacy/nlse_fock_cutoff_hybrid.py`, `common_code/check_number_diagonal.py` | `common_data/number_diagonal_check/` | Appendix I, analytic number-diagonal correction and the cutoff check at W = 0.3 |
| `lean_proof/` | `common_data/records/lean-build.log`, `lean-axioms.log` | Formalization experiments |

`python common_code/make_figures.py` redraws the benchmark figures from the data folders in a few seconds. `python double_sine_gordon_code/dsg_control_tradeoffs.py` reads the stored two-site step scan and regenerates the conditional duration comparison in Fig. 7, including its JSON record.

Figures 2–6 and 8–9 are drawn from the archived benchmark data. Their plotting script controls panel layout, labels, and visual encodings. The auxiliary polynomial plot shows the full stored infidelity range. Figure 6 separates splitting-error ratios from compiled primitive ratios, and Figure 9 separates the two outer-step counts. Redrawing these figures reads the stored results and performs no propagation or optimization.
