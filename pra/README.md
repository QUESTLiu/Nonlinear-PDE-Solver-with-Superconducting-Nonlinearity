# Reproducing the finite-model comparisons

This pipeline produced the records in `results/` that the manuscript in `../paper/` uses for its sine-Gordon, Duffing and double sine-Gordon comparisons. Its primary outputs are finite-family numerical comparisons, with separate estimates for reference cutoff, numerical solver and circuit padding. A successful optimizer termination or small training loss does not establish physical accuracy. In the records of experiment E4, which treat the double sine-Gordon lattice at λ2 = 0.3, the routes H_trig, H_trig_ret, H_SNAP and G_trig are the hybrid, retuned hybrid, SNAP+D hybrid and kinetic–potential circuits of the manuscript.

## Environment

The recorded interpreter is `/Users/zhen002/miniconda3/envs/cvdv/bin/python`, version 3.13.0, with NumPy 2.3.4, SciPy 1.16.3 and Matplotlib 3.10.7. `../.python-version` and `../requirements.txt` describe this environment. Each batch sets BLAS and OpenMP thread counts to one and executes scientific jobs serially.

Run commands from the repository root. The small independent checks are:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pra.run_e0
```

`run_e0` and the batch runner write the git commit and diff of the checkout into their records, so they must run inside a git repository. `python -m unittest pra.test_e0` runs the same checks without writing a record.

Rewrite `results/analysis_summary.json`, which `../paper/scripts/` reads, from the existing records without running simulations. The same command draws diagnostic figures into `pra/figures/`.

```sh
python -m pra.plot_results
```

## Numerical definitions

`metrics.py` owns phase-insensitive distance, the native-reference residual integral, the three native-error verdicts and exact-reference action sums. `duffing/duffing_padded_hybrid_validation.py` owns the actual finite displacement derivative and the common-phase compiler objective. `duffing/native_comparison.py` constructs the Kerr replacement and measures tails at its actual locations. `sine_gordon/native_correction.py` defines the spectral and physical-sandwich operators, single-ancilla gates, reduced metrics and finite first-moment residuals.

The SG spectral model uses the squares and trigonometric functions of finite quadratures. The physical-sandwich comparison uses projected infinite-oscillator matrix elements. Raising a validation cutoff keeps trained gate parameters fixed. The two conventions and the two frequencies `omega_b` and `Omega_time` are recorded independently.

## Records and bounded execution

| Record | Content |
|---|---|
| `records/e0.json` and `e0.log` | Independent small-operator and public-driver checks |
| `results/stage1/B0/` | B0 cost measurement |
| `results/stage1/` | Development E1, E2 and E3 data, and the initial SG reference |
| `results/stage1_continuation/` | SG cutoff 64 refinement and its failed continuation condition |
| `results/sg128/` | Authorized 96/128 reference, compiled pilot and convention check |
| `results/sg_formal/comparison/` | Fixed formal SG restart and circuit comparison |
| `results/sg160/` | Same-parameter 128/160 refinement of three existing trigonometric circuits |
| `results/sg_refinement/paired/` | Eighteen paired-frame starts and six full-circuit SNAP comparisons |
| `results/sg_size/expanded/` | Three larger-space, 12-layer SNAP starts and their selected circuit |
| `results/sg_continuation/fit/` | One bounded iteration continuation of the larger selected fit |
| `results/sg_independent/check/` | Fixed second-preparation check of the three representative trigonometric circuits |
| `results/duffing_refinement/comparison/` | Iteration-only and broader-training arms, with 12 new coherent inputs |
| `results/duffing_padding/comparison/` | Same selected Duffing pair at 96/112, 112/128 and 128/160 |
| `results/formal/E1_retry/` | The complete native Duffing grid after the exact-self-distance repair |
| `results/formal/E2/` | All Duffing restart records, training-only selection and ten-input evaluations |
| `results/formal/E3/` and `results/formal/polynomial/` | Classical calibration, fixed-variance controls and polynomial diagnostics |
| `formal_protocol.json` | Fixed comparison protocol, provenance of pilot reuse and implementation errata |
| `results/analysis_summary.json` | Unrounded values used by the plotting script |

The initial failed E1 and formal E2 execution records remain alongside the corrected runs. E1 failed because complex self-overlap roundoff gave a nonzero distance for an identical-state control. E2 stopped after a completed optimizer because its summary constructor repeated the seed field. Recovery reused the completed checkpoint and counted its original cost.

`run_batch.py` prints a manifest's commands unless `--execute` is present. It monitors wall time, resident memory and output size. An existing execution record cannot be overwritten. To reproduce a complete batch, copy its manifest with a new output directory and execution-record name, keeping the protocol unchanged. Manifest commands name the interpreter of the archived runs, so the copy must name the local interpreter. Resource caps do not grant permission to expand a workload.

```sh
python -m pra.run_batch pra/formal_manifest.json --batch B3-D
```

The original Duffing protocol chooses one restart per `(r,L)` using training loss before constructing the eight independent evaluation inputs. The two original real coherent inputs are development data. All selected pairs, failed accuracy criteria, optimizer limits and empirical margins contribute to the reported comparison. An iteration limit is not evidence of convergence to a local optimum. Subsequent refinements treat all original inputs as development data and specify fresh evaluation inputs before training.

The SG initialization comparison changes only the initial parameterization. Its six selected circuits still fail the state criterion. The larger `(N,d,columns,L)=(96,48,32,12)` fit and its continuation pass both observable criteria but fail the state criterion. The continuation lowers training loss while increasing full-circuit state error. These results describe the tested search and do not prove an ansatz capacity limit.

The second SG preparation, with requested site-0 field 0.75 and momentum 0.25, was specified in `sg_refinement_protocol.json` before the new training. After development selection, the same three representative trigonometric circuits pass all criteria at cutoffs 128/160. No SNAP circuit enters that check because none meets the development selection condition. The reference state-cutoff comparison uses the 21 outer-step times. Observable cutoff comparisons and independent solver comparisons additionally cover the 257-point diagnostic grid. A text-only scope correction, its original metadata copies and the unchanged numerical records are documented in `results/sg_independent/check/grid_scope_erratum.json`.

Each refinement batch directory includes a source snapshot. Its protocol and manifest state dimensions, initialization, selection, output scope and resource caps. `records/refinement-decisions-2026-09-24.md` records the adjudication of the external compiler diagnosis. `check_sg_reference_grid.py` contracts archived original-preparation states on the common circuit times without rerunning trajectories. Its record shows that those observable cutoff differences do not exceed the archived diagnostic-grid estimates.

The Duffing iteration-only arm improves training objective but fails the full input sets. The broader arm uses eight initial phases, 12 layers and three paired-frame starts. Its selected H/S_K pair passes the primary 1e-3 criterion on all ten development and twelve independent coherent inputs. Same-parameter padding checks through 128/160 preserve that verdict. The original failure records and all unsuccessful starts remain part of the comparison. The complete exploration record is `records/refinement-exploration-2026-09-24.md`.

## Formal proofs

The theorem map is in `../cvdv/README.md`. The full Lean build and the axiom audit are in `records/lean-build.log` and `records/lean-axioms.log`. The composition lemmas are conditional finite-space results. The native-reference integral and the classical-transfer estimate are proved in the manuscript.

Physical pulse synthesis, calibrated noise, continuum convergence and experimental readout are separate from these ideal finite-model results. Resource expressions keep pulse durations, native time conversion and measurement costs explicit.

The SG resource aggregation also cancels two inverse qubit rotations at each internal outer-step boundary, since the intervening native and Gaussian blocks act only on the system. Raw per-block schedules remain available. The maximum ancilla population refers to the logical per-step rotation frame. It is not a physical transient-excitation measurement for the merged pulse sequence. The manuscript recounts these circuits with the qubit basis change applied once per circuit (`../paper/scripts/trig_counts.py`), which gives lower counts for the trigonometric circuits than those archived here.
