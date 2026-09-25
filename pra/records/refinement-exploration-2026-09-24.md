# Compiler exploration record, 2026-09-24

This record was assembled after the runs. The linked protocols, initial parameter arrays, source snapshots and raw outputs preserve what each experiment tested. Numerical values below are finite-input observations with empirical reference, solver and padding estimates.

In SG, H_trig is the principal native-assisted construction. H_SNAP is the residual-compilation baseline under the same native interaction, and G_trig is the fully programmable baseline. An unsuccessful H_SNAP search is a completed baseline comparison with a negative outcome.

All numerical work used `/Users/zhen002/miniconda3/envs/cvdv/bin/python`, Python 3.13.0, NumPy 2.3.4 and SciPy 1.16.3. Scientific batches used one BLAS thread and a 4 GiB memory cap. Each batch has its own wall-time and output-size caps. Execution receipts report actual costs separately from those limits.

## Earlier experiments and recoveries

The [revision record](../../manuscript/pra_revision_notes.md) gives the complete implementation and theorem map. [Stage 1](../results/stage1/) and [its continuation](../results/stage1_continuation/) include the B0 measurement, initial pilots and unsuccessful low-cutoff SG continuation checks. The [formal protocol](../formal_protocol.json) records the subsequent fixed comparisons and checkpoint reuse.

The first formal E1 run stopped because self-overlap roundoff produced a nonzero identical-state distance. Its [original output](../results/formal/E1/) and the [corrected run](../results/formal/E1_retry/) both remain. The formal E2 wrapper stopped after an optimizer completed because the summary repeated a seed field. The completed parameters were reused, their original cost was counted, and both execution receipts remain in [the formal output directory](../results/formal/).

The original SG compiled validation was interrupted while repeated CSR actions dominated runtime. Its optimizer checkpoint and two completed N96 circuits remain in [the original directory](../results/sg128/compiled/). The [tensor continuation](../results/sg128/compiled_tensor/) uses the same finite action, checked against a dense oracle and those two completed circuits. The [differential record](sg_tensor_differential.json) preserves the comparison. Iteration and CPU counters absent from the interrupted checkpoint are reported as unavailable.

The [original Duffing study](../results/formal/E2/summary.json) has 18 optimizer runs and six selected pairs. The full ten-input family fails despite small errors on the two real development inputs. The [original SG comparison](../results/sg_formal/comparison/summary.json) includes every selected SNAP baseline and all 72 trigonometric configurations. Its G_trig r20 circuits meet the state and observable tolerances but fail the N96 boundary requirement. Three fixed representative circuits pass after [128/160 refinement](../results/sg160/summary.json).

## Diagnosis before further optimization

The [consultation brief](cc-refinement-brief-2026-09-24.md) separates the observed failures from candidate explanations. The [returned opinion](cc-refinement-opinion-2026-09-24.md) recommends examining Duffing training coverage and SG initialization. The [decision record](refinement-decisions-2026-09-24.md) corrects stronger causal inferences that the data do not establish. In particular, failed nonconvex searches do not prove a capacity limit, and a loose leakage upper bound does not establish the actual source of error.

The [SG parameterization probe](sg_parity_stationary_probe.json) evaluates objective and gradient without optimizing or evolving a trajectory. At N48, a zero-displacement diagonal fit and its paired-frame representation differ in unitary norm by 8.86e-15. With regularization set to zero for this geometry check, their gradient norms are 4.52e-16 and 4.81e-3. This supports an initialization intervention. It does not predict the final full-circuit accuracy. Production fits continue to use the original regularizer.

## SG baseline refinements

| Stage | Deliberate change | Observation | Optimizer / total time |
|---|---|---|---:|
| [Paired initialization](../results/sg_refinement/paired/summary.json) | Opposite displacement pairs and a diagonal first SNAP, with the original N48, d24, 16 training columns, layer/step grid and solver limits | All six selected circuits fail state accuracy. The smallest total amplitude is 0.07968 against the 0.03162 threshold. Five fail C12 and two fail S2 | 94.24 / 292.17 s |
| [Larger fit](../results/sg_size/expanded/summary.json) | N96, d48, 32 training columns, L12, r10, three starts, at most 2400 iterations | The training-selected circuit has total amplitude 0.05699. C12 and S2 pass. All three starts reach the iteration limit | 247.30 / 277.64 s |
| [One continuation](../results/sg_continuation/fit/summary.json) | Resume the minimum-loss larger fit, restarting SLSQP with a 900-second cap and at most 12000 further iterations | Solver success after 5063 additional iterations. Objective decreases from 7.7865e-4 to 6.4987e-4, but total amplitude increases to 0.05941. C12 and S2 still pass | 204.04 / 237.07 s |

The corresponding [initialization](../sg_refinement_protocol.json), [size](../sg_size_protocol.json) and [continuation](../sg_continuation_protocol.json) protocols precede their runs. Every start is saved. Selection uses training objective before full-circuit evaluation. The continuation starts from the exact recorded parameter vector and identifies it by SHA256. The larger trial changes several size variables together and is a feasibility check, not an isolated causal ablation.

The N48/N96 training-MSE difference is at most 1.45e-7 across the paired starts. The larger fit and its continuation also have consistent training MSE at N96/N128. These are checks of the specified training diagnostic. They do not establish full-trajectory convergence or useful accuracy outside the training columns.

The completed study contains fourteen selected H_SNAP configurations across the original and additional stages. None passes the state criterion on the original physical input. The larger fits' observable success and the continuation's worsening state error are both reported. No additional size ladder follows this study.

## Fixed SG circuits on a second preparation

The second coherent preparation was specified in the initialization protocol before the new training. It has requested site-0 means `(phi,pi)=(0.75,0.25)`, with zero means on site 1 and the same covariance, Hamiltonian and horizon. The fixed H_trig r10/order2 and G_trig r20/order1/order2 circuits are evaluated after development selection. No H_SNAP circuit enters this check because none meets the stated development condition.

All three pass state, observable and boundary criteria at 128/160. Their total amplitudes are 0.00633143, 0.02015968 and 0.01023622. The [second-preparation record](../results/sg_independent/check/summary.json) includes six circuit/cutoff outputs and an independent DOP853 reference check. Total time is 219.94 seconds and peak memory is 2.27 GiB. These data support two specific preparations and do not establish uniform input-family performance.

A descriptive grid-scope sentence initially overstated the cutoff-state sampling. The [erratum](../results/sg_independent/check/grid_scope_erratum.json) preserves all pre-correction JSON copies. The cutoff-state comparison uses the 21 outer-step times. Observable cutoff differences and all independent-solver differences use those times plus the 257-point diagnostic grid. Numerical arrays, values and verdicts did not change. A separate [saved-state check](sg_reference_common_times.json) confirms that the original preparation's observable cutoff differences on the common circuit times do not exceed its archived diagnostic-grid estimates.

## Duffing iteration and training coverage

The [new protocol](../duffing_refinement_protocol.json) defines two arms before training and fixes twelve fresh coherent amplitudes by seed and SHA256. All original ten inputs become development data for this study. Both arm selections are complete before either is evaluated on the fresh inputs.

The iteration-only arm repeats r10/L8/seed 240732 with the same training states, initialization and objective, increasing the iteration limit to 5000. Its initial vector and objective at iteration 1200 exactly match the archived run. It reports solver success at iteration 2891, with objective reduced from 2.12268e-5 to 4.10337e-6. Each route passes only 5 of 22 individual inputs, so both full-family criteria fail. This result confirms that a better original training fit does not resolve the observed family dependence.

The broader arm uses L12 and coherent radii `{1,1.5,2.25}` at eight initial phases, giving 240 exact-target training snapshots. It changes initialization and iteration allowance as well as coverage and depth. All three starts reach 5000 iterations. Seed240733 has the lowest training objective, 8.00047e-6, and is selected without evaluation feedback.

That H/S_K pair passes the primary 1e-3 criterion on all ten development and twelve fresh coherent inputs. At 96/112, the worst independent-input amplitudes are 0.02337130 and 0.02354832. The maximum boundary population is 8.0637e-7 for S_K at N112. Both routes fail the stricter 1e-4 family criterion. The [complete records](../results/duffing_refinement/comparison/summary.json) include the unsuccessful arm and all starts. Optimization takes 686.39 seconds and validation 1.53 seconds.

The new assessment checks the unchanged boundary threshold at both cutoffs. This is stricter than the original lower-cutoff-only assessment, and is distinguished from the iteration-only training intervention. It changes no verdict in this batch. Source comparison identifies one later owner edit, correcting “exact split trajectories” to “exact target trajectories” in the module docstring. Executable code matches the saved run snapshot.

## Duffing padding and resource interpretation

The N112 S_K boundary population exceeds its N96 value while remaining below threshold. A further [fixed-parameter check](../results/duffing_padding/comparison/summary.json) evaluates every input at 112/128 and 128/160, with the original 96/112 pair reproduced as a default-behavior control. It completes in 2.62 seconds. The reproduced quantities agree exactly, and both higher pairs preserve every primary verdict.

At 128/160, the worst total amplitudes are 0.02222396 for H and 0.02218808 for S_K. Maximum padding differences are 2.40e-5 and 4.19e-6. The largest monitored boundary population is 9.22e-10. Each input's time-resolved padding distances are saved. Parameters, physical inputs, reference cutoffs and thresholds do not change. This is additional numerical-resolution evidence on the same inputs.

H uses 120 SNAP, 120 displacement and 20 Gaussian primitives, plus 11 merged native waits. S_K uses 131 SNAP with the same displacement and Gaussian counts and no native waits. Thus the programmable primitive counts are 260 and 271. Counting waits as operations gives 271 for both, so “11 extra” refers specifically to programmable SNAP gates. Physical durations require the rates and pulse durations in the manuscript's resource model.

The saved N96 Kerr-replacement tail bound ranges from 1.69e-5 to 0.02056 across the 22 inputs. Its maximum, added to the corresponding H error, does not certify the primary S_K criterion. That pass is supported by direct numerical validation. The jointly changed broader arm does not isolate the causal contribution of training coverage, depth or initialization, and none of these searches establishes global optimality.

## Closure and remaining scope

Independent checks cover the raw numerical aggregates, selections, input roles, continuation identity, reference grids and resource vectors. E0 verifies the affected finite algebra and implementation interfaces. Lean supports the scoped finite statements listed in the manuscript's theorem map. The original sources and comparison records remain separate from new artifacts.

The numerical exploration is complete within these recorded budgets. The SG SNAP baseline's failed state criterion is a reported comparison outcome. Hardware pulse synthesis, device noise, continuum guarantees and uniform accuracy outside the tested input sets are not established by these records. Final language editing, Overleaf synchronization and assignment of a public archive identifier remain separate publication steps.
