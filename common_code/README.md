# Reproducing the finite-model comparisons

This pipeline produced the records in the `*_data/` folders that the manuscript uses for its sine-Gordon, Duffing and double sine-Gordon comparisons. Its primary outputs are finite-family numerical comparisons, with separate estimates for reference cutoff, numerical solver and circuit padding. A successful optimizer termination or small training loss does not establish physical accuracy. In the records of experiment E4, which treat the double sine-Gordon lattice at λ2 = 0.3, the routes H_trig, H_trig_ret, H_SNAP and G_trig are the hybrid, retuned hybrid, SNAP+D hybrid and kinetic–potential circuits of the manuscript. In these records the label SG and the folders named `sg*` refer to this double sine-Gordon instance. Experiment E3 provides the sine-Gordon calibration of Fig. 2 and the polynomial diagnostics of Fig. 8, E1 the native Duffing grid of Fig. 3, and E2 with its refinements the Duffing compiler of Figs. 4 and 9 and Table II.

## Environment

The recorded interpreter is `/Users/zhen002/miniconda3/envs/cvdv/bin/python`, version 3.13.0, with NumPy 2.3.4, SciPy 1.16.3 and Matplotlib 3.10.7. `../requirements.txt` describes this environment. Each batch sets BLAS and OpenMP thread counts to one and executes scientific jobs serially.

Run commands from the repository root. The small independent checks are:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m common_code.run_e0
```

`run_e0` rewrites `../common_data/records/e0.json` and `e0.log`. It and the batch runner record the git commit and a SHA-256 hash of the uncommitted diff, so they must run inside a git checkout with at least one commit. `python -m unittest common_code.test_e0` runs the same checks without writing a record.

Rewrite `../common_data/analysis_summary.json`, which `make_figures.py` and `derived_numbers.py` read, from the existing records without running simulations. The same command draws diagnostic figures into `common_data/diagnostic_figures/`.

```sh
python -m common_code.plot_results
```

## Numerical definitions

`metrics.py` owns phase-insensitive distance, the native-reference residual integral, the three native-error verdicts and exact-reference action sums. `../duffing_code/duffing_padded_hybrid_validation.py` owns the actual finite displacement derivative and the common-phase compiler objective. `../duffing_code/native_comparison.py` constructs the Kerr replacement and measures tails at its actual locations. `../sine_gordon_code/native_correction.py` defines the spectral and physical-sandwich operators, single-ancilla gates, reduced metrics and finite first-moment residuals.

The SG spectral model uses the squares and trigonometric functions of finite quadratures. The physical-sandwich comparison uses projected infinite-oscillator matrix elements. Raising a validation cutoff keeps trained gate parameters fixed. The two conventions and the two frequencies `omega_b` and `Omega_time` are recorded independently.

## Records and bounded execution

| Record | Content |
|---|---|
| `../common_data/records/e0.json` and `e0.log` | Independent small-operator and public-driver checks |
| `../double_sine_gordon_data/stage1/B0/` | B0 cost measurement |
| `../common_data/stage1/` | Development E1, E2 and E3 data, and the initial SG reference |
| `../double_sine_gordon_data/stage1_continuation/` | SG cutoff 64 refinement and its failed continuation condition |
| `../double_sine_gordon_data/sg128/` | 96/128 reference, compiled pilot and convention check |
| `../double_sine_gordon_data/sg_formal/comparison/` | Fixed formal SG restart and circuit comparison |
| `../double_sine_gordon_data/sg160/` | Same-parameter 128/160 refinement of three existing trigonometric circuits |
| `../double_sine_gordon_data/sg_refinement/paired/` | Eighteen paired-frame starts and six full-circuit SNAP comparisons |
| `../double_sine_gordon_data/sg_size/expanded/` | Three larger-space, 12-layer SNAP starts and their selected circuit |
| `../double_sine_gordon_data/sg_continuation/fit/` | One bounded iteration continuation of the larger selected fit |
| `../double_sine_gordon_data/sg_independent/check/` | Fixed second-preparation check of the three representative trigonometric circuits |
| `../duffing_data/duffing_refinement/comparison/` | Iteration-only and broader-training arms, with 12 new coherent inputs |
| `../duffing_data/duffing_padding/comparison/` | Same selected Duffing pair at 96/112, 112/128 and 128/160 |
| `../duffing_data/formal/E1_retry/` | The complete native Duffing grid after the exact-self-distance repair |
| `../duffing_data/formal/E2/` | All Duffing restart records, training-only selection and ten-input evaluations |
| `../sine_gordon_data/formal/E3/` and `../sine_gordon_data/formal/polynomial/` | Classical calibration, fixed-variance controls and polynomial diagnostics |
| `../common_data/manifests/formal_protocol.json` | Fixed comparison protocol, provenance of pilot reuse and implementation errata |
| `../common_data/analysis_summary.json` | Unrounded values used by the plotting script |

The initial failed E1 and formal E2 execution records remain alongside the corrected runs. E1 failed because complex self-overlap roundoff gave a nonzero distance for an identical-state control. E2 stopped after a completed optimizer because its summary constructor repeated the seed field. Recovery reused the completed checkpoint and counted its original cost.

`run_batch.py` prints a manifest's commands unless `--execute` is present. It monitors wall time, resident memory and output size, and refuses to overwrite an existing execution receipt or job log. To reproduce a complete batch, copy its manifest, set `output_root` to a new run directory, and keep `execution_record` as a filename inside that directory. Redirect the output arguments in the job commands to the same new directory, keeping the numerical protocol unchanged. Manifest commands name the interpreter of the archived runs, so the copy must name the local interpreter.

```sh
python -m common_code.run_batch common_data/manifests/formal_manifest.json --batch B3-D
```

The original Duffing protocol chooses one restart per `(r,L)` using training loss before constructing the eight independent evaluation inputs. The two original real coherent inputs are development data. All selected pairs, failed accuracy criteria, optimizer limits and empirical margins contribute to the reported comparison. An iteration limit is not evidence of convergence to a local optimum. Subsequent refinements treat all original inputs as development data and specify fresh evaluation inputs before training.

The SG initialization comparison changes only the initial parameterization. Its six selected circuits still fail the state criterion. The larger `(N,d,columns,L)=(96,48,32,12)` fit and its continuation pass both observable criteria but fail the state criterion. The continuation lowers training loss while increasing full-circuit state error. These results describe the tested search and do not prove an ansatz capacity limit.

The second SG preparation, with requested site-0 field 0.75 and momentum 0.25, was specified in `../double_sine_gordon_data/manifests/sg_refinement_protocol.json` before the new training. After development selection, the same three representative trigonometric circuits pass all criteria at cutoffs 128/160. No SNAP circuit enters that check because none meets the development selection condition. The reference state-cutoff comparison uses the 21 outer-step times. Observable cutoff comparisons and independent solver comparisons additionally cover the 257-point diagnostic grid. A text-only scope correction, its original metadata copies and the unchanged numerical records are documented in `../double_sine_gordon_data/sg_independent/check/grid_scope_erratum.json`.

Each refinement batch directory includes a source snapshot, and `../common_data/records/refinement-decisions-2026-09-24.md` records the adjudication of the external compiler diagnosis. The protocols and manifests state dimensions, initialization, selection, output scope and resource caps. `../double_sine_gordon_code/check_sg_reference_grid.py` contracts archived original-preparation states on the common circuit times without rerunning trajectories. Its record shows that those observable cutoff differences do not exceed the archived diagnostic-grid estimates.

The Duffing iteration-only arm improves training objective but fails the full input sets. The broader arm uses eight initial phases, 12 layers and three paired-frame starts. Its selected pair of circuits, the hybrid circuit H with native Kerr waits and the circuit S_K with every wait replaced by a SNAP gate, passes the primary 1e-3 criterion on all ten development and twelve independent coherent inputs. Same-parameter padding checks through 128/160 preserve that verdict. The original failure records and all unsuccessful starts remain part of the comparison. The development exploration record is `../common_data/records/refinement-exploration-2026-09-24.md`.

## Formal proofs

The theorem map of the optional Lean project is in `../lean_proof/README.md`. Build and axiom-audit logs are in `../common_data/records/lean-build.log` and `../common_data/records/lean-axioms.log`. The numerical checks do not require Lean. The composition lemmas are conditional results for normed groups and rings. The native-reference integral and the classical-transfer estimate are proved in the manuscript.

Physical pulse synthesis, calibrated noise, continuum convergence and experimental readout are separate from these ideal finite-model results. Resource expressions keep pulse durations, native time conversion and measurement costs explicit.

The SG resource aggregation also cancels two inverse qubit rotations at each internal outer-step boundary, since the intervening native and Gaussian blocks act only on the system. Raw per-block schedules remain available. The maximum ancilla population refers to the logical per-step rotation frame. It is not a physical transient-excitation measurement for the merged pulse sequence. The manuscript recounts these circuits with the qubit basis change applied once per circuit (`../double_sine_gordon_code/trig_counts.py`), which gives lower counts for the trigonometric circuits than those archived here.
