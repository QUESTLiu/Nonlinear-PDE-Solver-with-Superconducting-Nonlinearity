# 请评估下一轮 compiler refinement

项目为 `/Users/zhen002/GitHub/nonlinear-PDE-simulator`。用户希望完成以 PRA 为目标的 native-plus-correction bosonic dynamics 稿件，并在未达标的实验上调整必要的 size、提高研究质量。请独立判断现有失败的原因，以及哪组最小的新实验能够改变结论。当前还没有启动新的 Duffing 或 SG SNAP refinement。你的角色是提供科学和 numerical diagnosis，不需要实现改动，也不需要认可已有判断。

## 工作边界与入口

先读 `manuscript-implementation-handoff-2026-09-24.md`、`manuscript-revision-plan-2026-09-24.md` 的相关部分、`manuscript/main_pra.tex` 和 `manuscript/pra_revision_notes.md`。旧 `main.tex`、`ref.bib` 和旧 figures 均只读。当前原稿 SHA256 仍为 `21cb92539b7e775e94bc637ae0dcfdc742ecb91d1e76d5e6121fd3836ad7160b`。新稿与 research code 已在当前 working tree 实施，无 commit、push 或 Overleaf 操作。

请只读 code 和 records，可以做小规模 record arithmetic。不要启动 trajectories、optimizers、大矩阵、全套 tests 或新的外部调研。若需要额外 evidence，请明确指出最小 discriminating check、成本和可能的两个结论。可以只写一个意见文件 `pra/records/cc-refinement-opinion-2026-09-24.md`，其他文件不要修改。若运行任何 Python，先核对项目 declarations，再使用 `/Users/zhen002/miniconda3/envs/cvdv/bin/python`，当前版本 3.13.0，NumPy 2.3.4、SciPy 1.16.3。不得加入 `Co-Authored-By`、`Generated with` 或任何 AI authorship marker，作者是用户。

用户已批准后续计算，并明确允许提高相关 size。我们仍保持逐批 time、memory 和 storage caps，不能把“质量过关”理解为放宽 threshold、改变 target 或隐藏失败。新的调参若利用已经看过的 validation inputs，它们对后续工作就属于 development，需要另定独立 validation 或相应缩小 claim。

## 已完成的科学对象与验证

主线是 finite unitary quantum evolution。Classical first-moment validity 有独立的 closure 和 cutoff residual。T1 是 native-reference variance integral 的 sufficient bound，T2 是同一 finite space 内 common-phase exact-orbit action sum。Physical hardware、continuum 和全局最优 compiler 不在已建立的结论内。

E0 的独立 checks 包含 physical polynomial sandwiches、finite CCR、实际 finite-exponential Fréchet derivative、common-phase objective、actual Kerr replacement locations、SG retuning、single-ancilla signs/order/两-qubit equivalence、full-route generators 和 mixed-state metrics。全部通过。Lean full import graph 已 build，新增 composition lemmas 没有 proof placeholders 或新增 scientific axioms。新稿仍在 final integration，不能从 build 推出科学结论。

## Duffing：development 准确，完整 family 失败

固定 `beta=0.1,T=1,omega0=1`。训练使用 real coherent amplitudes `{1,1.5,2.25}` 的 exact-target intermediate states，经固定 Kerr/V2 half steps 后拟合 exact V4 action。全部 columns 共用一个 phase，objective 为 action MSE 加 `1e-8 mean(|alpha|^2)`。SLSQP 使用 actual finite derivatives，`maxiter=1200,ftol=1e-12`，phases 在 `[-pi,pi]`，displacement 的 real/imag components 在 `[-1.5,1.5]`。

正式范围为 `r={10,20},L={4,8,12}`，每组 seeds `{240731,240732,240733}`。Training dimensions 为 `N_pad=96,d_SNAP=48`，validation 使用 reference 64/80 和同参数 padding 96/112。18 次 optimizer 全到 iteration limit，累计 optimizer time 429.56 秒。每组只按 training loss 选择一个 restart，然后构造同参数的 Kerr-to-SNAP replacement `S_K`。

十个 evaluation inputs 包括 development coherent states `{1.75,2}`，另外四个 coherent states `{1.75i,2i,1.75 exp(i pi/4),2 exp(i pi/4)}`，以及 `|0>、|1>、(|0>+|1>)/sqrt(2)、(|0>+i|2>)/sqrt(2)`。后八个是在原 protocol 固定及 restart selection 之后才构造和评估的。它们现在已被看过。

六组 H 在 development inputs 上的 amplitude estimates 约为 0.00983–0.03032，但完整 family 上为 0.8503–1.3776。`S_K` 对应为 0.01054–0.03025 和 0.8502–1.7615。每组 worst input 都是 `|2i>`，完整 family 均未达到 `sqrt(0.01)=0.1`。部分 circuits 的 boundary population 超标，但 `r=10,L=4` 的 H 最大 boundary population 仅约 `9.35e-8`，其 full-family amplitude 仍约 0.8503。因此不能把全部失败归因于 padding。

Native/Gaussian ablation 的最大 distance 约 0.5096/0.4713，exact split 在 r10/r20 为 0.01387/0.003474。最大 reference margin 为 0.0006172，independent solver difference 小于 `6.72e-14`。Same-space T2 的 120 个 circuit/input rows 与每个 prefix 对应正确，但很多 sums 很大。Secondary rectangular norm 约 1.967，没有给出 uniform guarantee。

主要 records 是 `pra/results/formal/E2/summary.json`、各 restart 的 `optimizer.npz` 和 `optimization.json`，以及 `same_space_action_check.json`。Semantic owners 是 `duffing/duffing_padded_hybrid_validation.py`、`duffing/native_comparison.py` 和 `pra/metrics.py`。

## SG：强 trigonometric baselines 已通过，SNAP 未通过

固定两 sites、open boundaries、`E_J=1,lambda2=0.3,kappa=1`，`omega_b=Omega_time=sqrt(3)`，`T=5.2/sqrt(3)`，同一 projected coherent preparation，requested means `(1,0)`、momenta `(0,0)`。使用 finite spectral Hamiltonian，并另做 physical-sandwich comparison。

Native 和 curvature-matched retuned native 的最大 state distance 约 0.4108、0.5283，C12 errors 约 0.1152、0.1381。Reference 的 96→128 state change 为 0.001315。用户要求提高 cutoff 后，128→160 change 降至 0.0002301，independent DOP853 差约 `6.85e-9`。

SG SNAP local training 为 `N_pad=48,d_SNAP=24,N_acc=16`，用前 16 个 Fock basis columns 拟合 local `exp[-ih lambda2(1-cos(2phi_N))]`，两个 sites 共用同一个 block。`r={10,20},L={2,4,6}`，每组 seeds `{20260729,20260730,20260731}`。Zero start 与两个 `Normal(0,0.025^2)` cold starts 的规则与原计划一致。18 restart slots 中 17 个 solver success，1 个 iteration limit，optimizer time 共 25.87 秒。六个 training-selected circuits 全部未通过 state 和 C12 criteria，amplitude estimates 约 0.1034–0.1444。

可复核的具体线索是多数 optimized displacements 几乎为零。例如全部 r20 circuits 的 training loss 约 `8.5198e-4`，多个 alpha 最大模长低于 `1e-5`。r10 的 zero starts 给出相同 loss 约 `0.00340644`，alpha 近零。r10 L4/L6 的某些 random starts 离开该点，loss 降到约 `0.00171–0.00195`，但 full-circuit accuracy 仍失败。

Trigonometric comparison 已完整执行 72 个 configurations，包含 H_trig、G_trig、H_trig_ret、两种 order、`s={1,2,4,8,16,32}` 和两个 r。G_trig 的全部 r20 configurations 在 96/128 下满足 distance 和 observables，单独的 N96 boundary population 略高于 `1e-6`。用户批准的三例 128/160 同参数 refinement 已解决这个问题，结果如下。

| Circuit | Amplitude estimate | C12 error + margin | S2 error + margin | Maximum boundary population | Ideal programmable primitives |
|---|---:|---:|---:|---:|---:|
| H_trig, r10, order2, s1 | 0.00651723 | 0.000819184 | 0.00106737 | 3.419e-8 | 262 |
| G_trig, r20, order1, s1 | 0.0244965 | 0.00368006 | 0.00339716 | 3.924e-8 | 784 |
| G_trig, r20, order2, s1 | 0.0109371 | 0.00306078 | 0.00255555 | 3.505e-8 | 1104 |

三例均通过原 criteria。State amplitude threshold 是 `sqrt(1e-3)`，C12 error 加 margin 至多 0.01，S2 至多 0.02，另有 numerical resolution 和 boundary 条件。Counts 包含 exact pulse merging 及跨 system-only separator 取消 `2(r-1)` 个 Ry。它们不是 duration ratios。H 添加 native waits，G 添加 Gaussian pulses，双方都 coherent reuse 一只 qubit。物理 transient ancilla excitation 与 logical per-step-frame population 不混用。

主要 records 是 `pra/results/sg_formal/comparison/summary.json`、`pra/results/sg160/summary.json`、`pra/results/sg128/convention/summary.json`。Owners 是 `sine_gordon/native_correction.py`、`pra/run_pilot.py`、`pra/run_sg_formal.py`。全部三例高-cutoff records 经独立 arithmetic 复核，N128 是既有参数与 records 的复用，未重新训练。

## 请重点判断

1. 从 records 和 actual objective 看，Duffing 失败主要由 training coverage、迭代不足、ansatz capacity、active levels 还是 padding 决定？哪些证据支持或反驳各解释？不要仅根据 aggregate loss 推断原因。
2. SG 的 even-parity target 是否让 near-identity SNAP+D initialization 进入 displacement gradient 退化的 stationary region？若是，扩大 N/L 与改变 initialization 哪个先做更有区分力？一种待检验想法是 zero SNAP phases 加 paired opposite displacements，令初始总 circuit 仍为 identity，但 SNAP derivatives 作用在 displaced frames。这只是 hypothesis，请指出可能的问题。
3. 请给一个最小 bounded refinement，列清需要改变的 size、initialization 或 optimizer setting，保留不变的 target/criteria，成本上限和停止规则。优先区分原因，避免同时扩大所有 axes。如果要改 training coverage，如何重建后续独立 evaluation，使旧的失败 family 不再被称为新的 held-out evidence？
4. 现有科学结论是否已经由 H_trig/G_trig 的可复核 matched-accuracy comparison 支撑？让 H_SNAP 或 Duffing compiler 通过，分别会增加哪项实质结论？如果单纯追求每一路线都通过并不合理，请直接说明，同时指出必须完成的质量检查。

请把支持事实、未验证机制和建议分开。返回应引用实际文件位置或 record fields，允许否定上面的初步解释。
