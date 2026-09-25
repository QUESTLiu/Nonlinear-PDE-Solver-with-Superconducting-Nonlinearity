# Decisions following the compiler diagnosis

The consultation in `cc-refinement-opinion-2026-09-24.md` identifies useful mechanisms but contains several conclusions stronger than its evidence. The original comparison records remain unchanged. New runs use separate output directories and the original acceptance thresholds.

## Accepted evidence and interpretation

The Duffing training snapshots start from three real coherent amplitudes. In the original independent evaluation, 45 of 48 H circuit/input rows have larger common-grid distance than the Gaussian-only control, and 43 exceed native Kerr. This count was checked directly against `../results/formal/E2/summary.json`. For r=10 and L=4, the worst-input H distance is about 0.85014, compared with a same-parameter 96/112 padding difference of 2.01e-4. The large failure persists under that padding comparison. Training coverage is a supported explanation for the observed family dependence. The effect of additional iterations and the attainable accuracy of the ansatz remain unmeasured. Successful development inputs do not by themselves exclude either effect.

For SG, parity makes the displacement gradient zero at every point with all displacements zero. The SNAP-phase gradient need not vanish there. A full stationary point results when the remaining phase variables also solve their restricted optimization problem. The archived near-diagonal solutions are numerical approximations to such points. Two selected circuits leave that region, so the original comparison also contains evidence about nonzero-displacement fits. Its failure cannot be attributed entirely to the stationary point.

The paired-frame probe compares two parameterizations of the same diagonal unitary. At N=48, their unitary difference is 8.86e-15, while the gradient norms are 4.52e-16 and 4.81e-3. This probe uses zero regularization to isolate the fit geometry. The new optimizer comparison keeps the original regularizer and records the resulting objective. A lower loss under this initialization would establish a limitation of the old search. Failure to improve under another finite search would not prove a capacity limit.

The SG reference population outside the first 16 levels of either mode is approximately 4.21e-3 at the final time. This motivates checking training-space coverage. A large restricted-action leakage upper bound is inconclusive evidence, not a lower bound on the circuit error. In addition, the local compiler acts after a native half step, so an outer-step reference population is not the population at every local correction location.

The manuscript already describes the high-cutoff trigonometric counts as representative and does not claim their ratio is an optimum. Searching further outer-step counts is unnecessary for that scoped comparison. The fixed configurations will instead receive a second-input check after development selection. The new input is declared before the initialization refinement and will not be used for tuning.

## Executed decision

`../sg_refinement_protocol.json` fixes an 18-restart SG comparison with the original dimensions, objective, optimizer limits, selection rule and numerical thresholds. Only initialization changes. The seed generates bounded opposite displacement pairs, with the first SNAP at the diagonal phase fit. Independent finite-matrix multiplication verifies the initial unitary, and a finite-difference derivative checks its nonzero tangent direction. The optimizer records every iteration without additional objective evaluations. Each completed fit is also checked at training cutoff 96 with unchanged parameters, separately from the full-circuit padding comparison.

The optimizer cap is 3600 seconds and deterministic validation has a separate 1800-second cap. The batch uses one BLAS thread, at most 4 GiB of memory and 500 MiB of new output. Any subsequent training-space or depth adjustment will have its own recorded dimensions and stopping rule. The second coherent input has requested field 0.75 and momentum 0.25 on site 0, with zero means on site 1. It will be evaluated once after the development choices are complete.
