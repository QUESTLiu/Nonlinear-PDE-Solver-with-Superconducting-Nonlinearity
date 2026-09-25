# Duffing 与 SG SNAP refinement 意见

日期 2026-09-24。本文回答 `pra/records/cc-refinement-brief-2026-09-24.md` 的四个问题。

证据基础。数字来自 records 字段、对这些字段的算术，以及 `pra/records/sg_parity_stationary_probe.json`。该 probe 由 `pra/probe_sg_stationary.py` 在 brief 之后生成，它在 r=10 的 local step 上、regularization 为零时，对每种参数化只做一次 objective 与 gradient evaluation。所用的 L=4 与 displacements 0.5、−0.5、0.5i、−0.5i 只写在脚本中。算术使用 `/Users/zhen002/miniconda3/envs/cvdv/bin/python`（Python 3.13.0，NumPy 2.3.4），与 `.python-version` 和 `requirements-pra.txt` 一致。当地时间 19:30 之前核对时，`duffing/duffing_padded_hybrid_validation.py`、`duffing/native_comparison.py`、`pra/metrics.py` 与 `sine_gordon/sglib.py` 的 SHA256 与 E2、SG formal、sg160 records 的 provenance 相同，`sine_gordon/native_correction.py` 与 SG formal、sg160 records 相同。`pra/run_pilot.py` 当时已与 SG records 的 hash 不同，且没有存档，因此 SG 部分只依赖 record fields 和 hash 一致的 owners。此后 `duffing/duffing_padded_hybrid_validation.py`、`pra/run_pilot.py`、`pra/run_duffing_formal.py`、`manuscript/main_pra.tex` 与 `manuscript/pra_revision_notes.md` 都有修改，下文对 manuscript 的引用以 19:51 读取的版本为准。标为"未核实"或"估计"的内容没有 record 或计算直接支持。

## 结论

1. Duffing 的失败主要由 training coverage 造成。Compiled residual 只在训练 snapshots 所在的 phase-space arc 附近准确，离开该区域后在多数情况下比完全省略 V4 更差。Records 排除了 padding。Active levels 不是失败的必要条件，只在最差的几行中与大幅 excursion 同时出现。更多迭代不能约束 training span 之外的作用，这一点是推理，records 没有 loss 历史可以直接显示。Ansatz capacity 能否覆盖全部 phase，现有数据无法回答。
2. 在 SG 的 local objective 中，α=0 这个集合上的 displacement gradient 恒为零，这由 parity 对称决定，与 N 和 L 无关，从该集合出发的 gradient 方法不会离开它。18 个 restarts 中 15 个停在该集合的 diagonal optimum，6 个 selected circuits 中 4 个只实现了 number-diagonal correction。另两个 selected fits 是真正的 SNAP+D 解，但只提供弱证据。应先改 initialization，而不是先扩大 N 或 L。
3. SG 的 training space 不能为末时刻 state 提供保证。Reference 末时刻有 4.21e-3 的 population 位于 P_16⊗P_16 之外，restricted T2 bound 的 leakage 项为 0.421，是 state budget 0.0316 的 13 倍。这是最坏情况上界，只说明 P_16 上的训练无法 certify state criterion，并不证明 tail 造成了现有误差。
4. Paired opposite displacement 能打破一阶退化。应把 undisplaced SNAP 放在已知的 diagonal optimum，并按 d_SNAP 限制 frame 幅度。
5. 覆盖全部 phase 的 Duffing training family 若包含 antipodal initial states，也会有同样的 parity 对称，zero start 会停在 α=0。Duffing 的下一轮 training 必须使用打破该对称的 initialization。
6. H_trig/G_trig 的 records 已在一个 instance 和一个 development input 上支撑 SG 的比较结论。H_SNAP 或 Duffing compiler 通过会增加有限的新内容，但不改变 native interaction 可用性（H_trig 对 G_trig）或 Kerr replacement 的结论。在同一 development input 上调参直到每条路线都通过，没有科学理由。
7. 推荐的最小下一步是 SG-A。它只改 initialization，沿用原 18 个 restart slots，optimizer 预计合计数分钟。Duffing D1 可在其后按需运行。

## 问题 1：Duffing 失败的原因

### 支持事实

- 训练数据只覆盖一段 phase-space arc。`pra/run_duffing_formal.py` 用 `df.build_training_snapshots(96,[1.,1.5,2.25],r,...)` 生成训练 columns，三个 real coherent amplitudes 沿 exact target 演化。按只含 Kerr 项的解析估计，r=10 时 residual block 输入的相位从约 −0.06 rad 转到 −1.09、−1.18、−1.38 rad（半径 1、1.5、2.25），V2 与 V4 会带来修正（估计，未核实）。Objective `||e^{-iγ}WX−Y||_F^2/N` 只约束 W 在这些 columns 的 span 上的作用。
- Development inputs 全部达标。六组 H 在 real 1.75、2 上的最大 amplitude estimate 为 0.0098–0.0303，低于 √1e-3=0.0316。
- 逐步误差随输入相对 arc 的位置变化。`pra/results/formal/E2/evaluation_r*_L*/<input>.npz` 的 `hybrid_action_errors` 是 exact target 中间态上的逐步 action error。Diagonal inputs 从 π/4 顺时针转入 arc（估计终点 −0.52 至 −0.59 rad），逐步误差由第一步的 0.067–0.640 降到最后一步的 0.0016–0.0081，与 real2 最后一步的 0.0033–0.018 相当。imag2 从 π/2 转到约 0.20 rad（估计），始终在 arc 之外，逐步误差由 0.33–1.56 降到 0.047–0.37。Vacuum 与 |1⟩ 的 phase-space 分布不随时间转动，它们的逐步误差也不下降。
- Off-arc 作用比省略 V4 更差。48 个 independent (pair, input) rows 中，H 的 `max_common_distance` 有 45 行大于 Gaussian-only control，43 行大于 native Kerr。r10 L4 的 vacuum 为 0.2140，Gaussian control 为 0.0133。r20 L12 的 imag2 逐步 action error 平均 0.97，最大 1.56，上限为 2。
- Depth 的影响随输入而异。L 从 4 增到 12 时 development 误差下降，而 off-arc coherent inputs（imag 与 diagonal）的误差在两个 r 下都单调增大。例如 imag2 的 H amplitude estimate 在 r10 为 0.8503、1.3633、1.3775，在 r20 为 1.1067、1.3163、1.3667。低 Fock inputs 不单调，例如 r10 的 vacuum 为 0.214、0.157、0.152。
- Padding 与 active levels 不能解释低 Fock inputs 的失败。r10 L4 的 vacuum 与 |1⟩ 在全部 gate locations 上的 above-active population 最大为 5.6e-9 与 1.6e-7，boundary population 为 3.3e-15 与 1.4e-13，distances 仍为 0.214 与 0.483。全部 H rows 的 padding margin 不超过 0.0101，independent inputs 上的 H distance 不小于 0.115。
- 18 个 runs 的 message 都是 `Iteration limit reached`，单次用时 13.1–35.8 s。r10 的 training MSE 在 L=4、8、12 分别为 1.5e-4–1.7e-4、2.1e-5–4.3e-5、1.5e-5–1.9e-5。Zero start 与 random starts 的 losses 属同一量级。原 training family 只含正实数 amplitudes，在 parity 下不封闭，因此没有问题 2 所述的 stationary trap。
- H 与 S_K 在所有 rows 上的 `max_common_distance` 相差不超过 0.0058。S_K 在 r20 L8 imag2 的 1.7615 等于 common distance 1.3153 加 padding margin 0.446，高出 H 的部分来自 padding margin，该 row 的 S_K boundary population 为 0.0875。

### 判断

| 解释 | 判断 | 依据 |
|---|---|---|
| Training coverage | 主要原因 | 逐步误差随输入相对 arc 的位置变化，off-arc 作用在多数 rows 中比省略 V4 更差 |
| 迭代不足 | 推断不是失败原因 | Development 已达标。更多迭代只降低 span 内的误差，不增加 span 外的约束。Records 没有 loss 历史，这是推理 |
| Ansatz capacity | 对 training family 足够，对全 phase family 未检验 | 对 off-arc coherent inputs，L 增大使误差单调增大，符合参数相对训练约束过多的情形 |
| Active levels | 不是必要条件 | r10 L4 的低 Fock inputs 几乎不进入 level 48 以上，误差仍为 0.2–0.5。最差几行的 above-active population 达 0.238（r20 L8 imag2）与 0.185（r10 L8 imag2），可能放大这些行的误差 |
| Padding | 被 records 排除 | H 的 padding margin 最大 0.0101，失败误差不小于 0.115 |

Brief 中"不能把全部失败归因于 padding"的判断成立，并且可以加强。Records 排除了 padding 对 H 任何失败 row 的解释。

### 未核实

- 训练 snapshot matrix 的 numerical rank，以及每个输入落在其 span 外的比例，没有计算。它能量化 coverage 解释，不会改变判断的方向。
- 上文的相位值是只含 Kerr 项的解析估计。
- SNAP+D 在 L≤12 时能否拟合覆盖全部 phase 的 family，以及拟合后能否泛化，需要新的 optimizer 数据。
- Runs 没有 loss 历史，无法判断在 1200 次停止时 loss 是否仍明显下降。更难的 training family 可能使 iteration cap 成为限制。

## 问题 2：SG 的 stationary region 与先做什么

### 支持事实

对称性结论（纸面推导，在有限维中精确成立）。记 Π=(−1)^n。有限矩阵满足 Π a_N Π=−a_N，所以 Π D_N(α) Π=D_N(−α)，而 SNAP 为 diagonal，满足 Π S(θ) Π=S(θ)。设 training inputs 为 X，targets 为 Y。若存在一个作用在 columns 上的 unitary Q，使 ΠX=XQ 且 ΠY=YQ，则 Frobenius objective 满足 f(θ,−α,γ)=f(θ,α,γ)。Regularizer ν·mean|α|²（ν=1e-8）也是偶函数。SG 的 X 是前 16 个 Fock columns，Y=U_hX，U_h=exp[−ihλ2(1−cos 2φ_N)] 与 Π 对易，Q 是 ±1 的 diagonal，条件成立。于是对所有 θ、γ、L、N_pad 和 d_SNAP，α=0 这个集合上 ∂f/∂α 恒为零，从该集合出发的 gradient 方法不会离开它。这个结论不用 CCR，也不要求 SNAP phases 接近零。

Records：

- `sg_parity_stationary_probe.json` 在 N_pad=48、d_SNAP=24、16 columns 下给出 diagonal optimum 的 loss 0.0034064443 和 gradient norm 4.5e-16。
- Zero starts（seed 20260729）在 4–6 次 iteration 后以 solver success 结束（重用的 pilot 记录 10 次 evaluation）。r10 的 loss 为 3.40644431e-3，r20 约为 8.5198266e-4，max|α|≤1.1e-7。两者之比 3.998，与 diagonal residual 随 ε=hλ2 平方缩放一致。
- 18 个 restarts 中 15 个的 loss 与 diagonal optimum 至少六位有效数字相同，max|α|≤7.4e-3。r20 的六个 random starts 全部回到这一点附近。只有 r10 L4 的 seeds 20260730、20260731 和 r10 L6 的 seed 20260730 离开，loss 为 1.71e-3、1.95e-3、1.77e-3。
- 6 个 selected circuits 中，r10 L2 与三个 r20 circuits 是 diagonal 解。它们在 96/128 的 Bures 为 0.1414–0.1417，C12 error 为 0.0443–0.0447，S2 error 为 0.0390–0.0395，不随 r 和 L 变化。误差不随 r 下降，与缺失的 off-diagonal correction 一致，与 outer splitting 误差不一致。
- 两个 SNAP+D selected circuits 是 r10 L4 与 r10 L6。它们的 Bures 为 0.1198 与 0.0948，C12 error 为 0.0199 与 0.0162，S2 通过，boundary population 在 N=96 为 4.2e-6 与 1.26e-5，在 N=128 为 8.0e-7 与 2.8e-6。r10 L4（seed 20260730）的四个 displacements 为 −0.004+1.433i、0.208+1.5i、−0.194−1.435i、−1.5i，两个 components 等于 bound 1.5，displacements 部分和的最大模为 2.94，运行中离开 P_16⊗P_16 的 population 达 7.4e-2。r10 L6（seed 20260730）的最大 |α| 为 0.62，运行中离开 P_16⊗P_16 的 population 为 2.9e-3，与 diagonal 解的 3.3e-3–3.4e-3 相近，但它在 1200 次 iteration 时终止。
- Reference tail 超出 training space。由 `pra/results/sg128/reference/E4_reference_lambda0.3_N128.npz` 的 `target_step_states` 计算，t=0、T/2、T 时每个 site 的平均占据数不超过 0.9，但 Fock tail 随时间增长。下表的 ℓ_k 是 r=10 grid 第 k 个 step 时刻落在 P_M⊗P_M 之外的 amplitude。

| M | 末时刻 joint population outside | 末时刻 site 1 / site 2 在 level M 以上 | 2Σ_{k=0}^{9} ℓ_k |
|---:|---:|---:|---:|
| 16 | 4.21e-3 | 1.64e-3 / 3.99e-3 | 0.421 |
| 24 | 2.01e-3 | 6.6e-4 / 1.94e-3 | 0.199 |
| 32 | 1.01e-3 | 3.0e-4 / 9.8e-4 | 0.108 |
| 48 | 1.66e-4 | 4.6e-5 / 1.6e-4 | 0.031 |

M=16 的末时刻值与 record field `reference_access_population` 的最大值一致。最后一列是 `main_pra.tex` 中 Eq. (t2restricted) 的 leakage 项，对 M≤32 超过 √1e-3=0.0316，r=20 时约加倍。

### 判断

- Brief 的假设成立，并且可以更精确。退化来自整个 α=0 集合，与 SNAP phases 是否接近 identity 无关。"17 个 solver success" 多数是在这个集合内收敛。
- 现有 H_SNAP 结果主要检验了 number-diagonal correction。两个 SNAP+D fits 的 loss 只降到 diagonal optimum 的约一半，r10 L4 的 displacements 触到 bound 且有大幅 excursion，r10 L6 未收敛，因此它们对 SNAP+D capacity 只提供弱证据。
- 先改 initialization 更有区分力。扩大 N 或 L 不改变 α=0 集合的性质，r20 的 L=2、4、6 已全部停在同一 loss。SG-A 能把这个 trap 与其他限制分开，但单一 initialization family 的失败并不单独证明 capacity 不足。
- Size 在 initialization 之后才成为主要问题，而且涉及的是 training space。N_acc=16 对末时刻 amplitude 约 0.065 的分量没有约束，这部分作用只能由 full-circuit validation 检验。Validation cutoffs 96/128/160 已经足够，reference 的 128→160 变化为 2.3e-4。

### Paired opposite displacement 设想

Probe 检验了一个 anchored 版本。L=4，第 0 层 SNAP 取 target diagonal 的相位，displacements 依次为 0.5、−0.5、0.5i、−0.5i，相邻 pair 之间的 SNAP 为零。它的 unitary 与 diagonal optimum 相差 8.9e-15，loss 相同，gradient norm 为 4.81e-3，且在报告精度内全部来自 SNAP components。设想因此能打破一阶退化，但需要注意以下问题。

1. Paired 点上 displacement gradient 仍为零，第一步只由 displaced frames 中的 SNAP phases 推动。
2. Identity 只在相邻 pair 之间 SNAP phases 为零时精确成立。这些 phases 若随机初始化，起点就不再是 identity。
3. 所有 SNAP phases 取零时，起点 loss 高于 diagonal optimum，下降过程仍可能回到 α=0 的 basin。把第 0 层 SNAP 设为 arg⟨n|U_h|n⟩ 后，起点 loss 恰为 diagonal optimum，gradient 非零。α=0 集合上的最小 loss 就是 diagonal optimum，所以 loss 一旦下降，单调下降的迭代不会再回到该集合。
4. Frame 幅度受 d_SNAP 限制。n≤15 的 columns 在 displaced frame 中仍位于 d_SNAP=24 以内，要求大致 (√15+|β|)²≤24，即 |β|≤1.03。考虑 displaced Fock state 的宽度，|β|≲0.8 较稳妥（估计，未核实）。r10 L4 的部分和 2.94 会把 n=15 的 column 推到 level 约 46，接近 N_pad=48（估计，未核实）。
5. 它不处理 tail coverage。离开 α=0 后，block 对 level 16 以上分量的作用可能比 diagonal 解更大。
6. 这个 rule 是在看到该 physical input 的结果后设计的，由此得到的 SG 结果仍是 development evidence。

### 未核实

- α=0 在 r20 是否为 α 方向的严格局部极小。六个 r20 random starts 都回到 α≈0，但 selected 的 r20 L4 seed 20260731 停在 |α|=7.34e-3，loss 比 zero starts 低 1.3e-10，而 zero starts 之间只差约 1e-12。这更像是平坦或微弱下降的方向在 ftol=1e-12 下提前停止。Hessian 未计算。
- r10 L4 是否依赖 N_pad=48 的 truncation。部分和 2.94 与 N=96 的 boundary population 提示这一点，但没有比较同一参数在 N_pad=48 与 96 的 P_16 objective。
- Tail 对现有 H_SNAP 误差的实际贡献。上表只给最坏情况。
- 由三个 r10 数据点得到的经验关系 Bures≈(2.3–2.9)√loss 能否外推。若成立，state criterion 约需 r10 loss≤1e-4、r20 loss≤3e-5。

## 问题 3：最小 bounded refinement

### SG-A：只改 initialization（首选）

| 项目 | 设置 |
|---|---|
| 保持不变 | Local target、N_pad=48、d_SNAP=24、N_acc=16 的 uniform objective、ν=1e-8、phase 与 displacement bounds、SLSQP maxiter 1200 与 ftol 1e-12、按 training loss 选择、96/128 validation（boundary 需要时做 128/160 同参数复核）和全部 acceptance criteria |
| 唯一改变 | 第 0 层 SNAP 对全部 24 个 active levels 取 arg⟨n\|U_h\|n⟩，其余 SNAP 为零。Displacements 在相邻两层成对取 (β_k,−β_k)，\|β_k\| 与方向由 PCG64(seed) 从预先声明的分布抽取，例如 \|β\|∈[0.3,0.8]、方向均匀。Common phase 从零开始 |
| Grid | r∈{10,20}、L∈{2,4,6}、seeds 20260729–20260731 全部用新规则，共 18 restarts，与原 grid 一一对应 |
| 运行前检查 | 用修改后的 owners 对一个原 selected parameter vector 做一次 objective evaluation 和一次 H_SNAP validation step，确认复现原 record 数值。新 records 带上修改后 owners 的 hash |
| 新增记录 | 每次 iteration 的 loss、最终 max\|α\| 与触到 bound 的 components 数、训练中 level 24 以上和 top 8 levels 的最大 population、同一参数在 N_pad=96 的 P_16 objective |
| 上限 | 单次 15 min，optimizer 合计 1 h，validation 30 min，内存 4 GiB，新输出 500 MB。原 batch 的 optimizer 合计 25.87 s，validation 平均约 8 s 每个 circuit/cutoff record |

判定规则。记 L_d 为 diagonal optimum（r10 为 3.41e-3，r20 为 8.52e-4），L_t 为上面经验关系给出的目标 loss（r10 约 1e-4，r20 约 3e-5，未核实），L* 为每个 r 的最佳 selected loss。

1. 按原 protocol 验证全部 selected circuits。任一 circuit 通过全部 criteria 即停止并报告。
2. L*≤L_t 但 full circuit 失败时，把逐步 action error 分成 P_16⊗P_16 内外两部分。外部分量占一半以上则运行 SG-B，否则报告 local objective 与 full trajectory 不一致，停止。
3. L*>L_t 时，若同一参数在 N_pad=96 的 P_16 objective 与训练值相差超过 10%，training N_pad 是限制，运行 SG-B。否则运行 SG-C，并报告触到 bound 的 components 数，因为 bound 不变时，更多 layers 可以在 bound 内组合出更大的位移。
4. SG-A 之后最多再运行 SG-B 或 SG-C 之一，仍未通过则把 H_SNAP 记为 budget-limited。Target、thresholds、bounds 与 objective 定义始终不变。

SG-B（training space）取 N_acc=32、d_SNAP=48、training N_pad=96，L 与 initialization 同 SG-A，此时 (√31+0.8)²≈41，仍在 d_SNAP 以内。M=32 的 leakage 项为 0.108，是 budget 的 3.4 倍，所以 SG-B 能否通过取决于 tail 的实际贡献。SG-C（capacity）取 r=10、L∈{8,12,16}，尺寸与 initialization 同 SG-A，相邻 L 的 loss 改善小于 2 倍时停止。

另有三项可选的无 optimizer 检查，各需秒至分钟级。第一项计算 r10 与 r20 diagonal optimum 处 α-Hessian 的最小特征值，第二项比较 r10 L4 在 N_pad=48 与 96 的 objective，第三项用已存 exact states 把 diagonal circuit 的逐步 action error 分成 P_16⊗P_16 内外两部分。第三项能在 SG-A 之前判断 N_acc 是否已是约束。

### Duffing：D1

| 项目 | 设置 |
|---|---|
| 保持不变 | Target、N_pad=96、d_SNAP=48、reference 64/80、padding 96/112、objective 与 ν、bounds、ftol、SLSQP maxiter 1200、按 training loss 选择、全部 controls、S_K 构造和 thresholds |
| 改变 1：training coverage | 半径仍为 {1,1.5,2.25}，初始 phase 取 kπ/4（k=0,…,7），共 24 个初始态，r=10 时 240 个 columns。每条 trajectory 在 [0,T] 内转过约 1.15–1.45 rad（估计），大于相邻初始 phase 的间隔 0.785 rad |
| 改变 2：initialization（由改变 1 强制） | 该 family 含 antipodal pairs。`coherent_state` 满足 Π\|ξ⟩=\|−ξ⟩，H_D 的各部分与 Π 对易，所以存在 column 置换 Q 使 ΠX=XQ、ΠY=YQ，objective 对 α 为偶函数，原 seed 240731 的 zero start 会停在 α=0。三个 seeds 都改用 anchored paired init：第 0 层 SNAP 取 closed-form diagonal optimum θ_n=arg Σ_j conj(X_nj)Y_nj，displacements 成对取 (β,−β)。运行前用一次 evaluation 确认起点 loss 等于 diagonal optimum 且 SNAP gradient 非零 |
| Grid | r=10，L∈{8,12}，三个 seeds，共 6 restarts |
| 迭代记录 | 记录每次 iteration 的 loss。若最后 200 次 iteration 的 loss 下降超过 10%，按预先声明的 continuation 把同一 run 延长到 5000 次，并单独报告 |
| 上限 | 单次 15 min，optimizer 合计 1.5 h，validation 30 min，内存 4 GiB，新输出 1 GiB。原 runs 在 30 个 columns、1200 次时用 13–36 s，240 个 columns 的用时未测 |

原 arc family 在 parity 下不封闭，原 zero start 也没有被困，所以新旧两组都是非退化起点，比较的主要差别仍是 coverage。

判定规则。原 r10 arc 拟合中，MSE 1.5e-4 对应 development 误差 0.0295，MSE 2.1e-5 与 1.5e-5 对应 0.0163 与 0.0153，后两者受 exact split 的 0.0139 限制。据此：

- L=12 时 covering MSE 仍不低于 1.5e-4，且 loss 历史已平稳，判为 capacity 不足，停止并报告，L>12 另行决定。
- MSE≤5e-5，且原 imag、diagonal inputs（现为 development）的逐步 action error 降到 √MSE 量级时，coverage 解释得到确认，可以冻结新的正式 protocol 并加入 r=20。
- 介于两者之间时，最多再运行一个加密半径的 coverage arm。
- 低 Fock inputs 不在该 family 内，作为 stress test 单独报告。

不需要训练的替代方案是把 claim family 缩小为 E1 使用的 real-displacement coherent states，并在新的 real amplitudes（例如 1.25、2.1）上评估现有六组 pairs。这是事后缩小的 claim，必须如此标注，也不回答泛化问题。

### 重建独立 evaluation

1. 任何新训练之前，冻结新的 protocol 文件，写明 claim family、training design、development set 和 held-out set。
2. Held-out set 按声明的 seed 从 claim family 抽样，例如 |α|∈[1,2.25]、arg α∈[0,2π) 的 12 个 coherent states，记录 SHA256，并排除原有十个 inputs。
3. 原十个 inputs 全部标为 development，只用于 regression 与诊断。原 E2 结果作为 arc-trained protocol 的 negative result 原样保留。
4. Restart selection、go/no-go 与 L 的选择只使用 training loss 和 development set。Held-out set 在最终 selection 之后打开一次，此后即成为 development。
5. 若需要检查新 family 的 coverage，使用从同一分布另抽的 design sample，不使用 held-out states。
6. SG 的 local training 不使用 physical input，但 initialization rule 与尺寸依据该 input 的结果选择。要得到 held-out evidence，需在 SG-A 之前声明第二个 physical input（同一 preparation family，requested means 或 momenta 不同），并在全部选择完成后只评估一次。

## 问题 4：现有结论的支撑与质量检查

### 支撑情况

这里的 SG 比较结论指 H_trig（native SG 加 trig residual）与 G_trig（全部 programmable）都能在不改变 criteria 的情况下达标，前者使用更少的 ideal programmable primitives，同时增加 native waits。

- 这一结论有可复核的支撑。`pra/results/sg_formal/comparison/summary.json` 与 `pra/results/sg160/summary.json` 中三个 128/160 rows 的 amplitude、C12、S2 与 boundary 数值与 brief 表格一致。H_trig 在 r=10 的 12 个 configurations 中有 11 个于 96/128 通过，o1 s1 只在 N=96 boundary（1.03e-6）上略超。G_trig 在 r=10 全部失败，amplitude estimate 为 0.046–0.065，s 增大后停在 0.046，由 outer splitting 主导。G_trig 在 r=20 的 12 个 configurations 的 amplitude 与 observables 全部通过，但 N=96 的 boundary population 为 1.03e-6–1.08e-6，均略超门槛。只有 r20 的 o1 s1 与 o2 s1 做了 128/160 同参数复核，两者通过。
- 支撑范围是 E_J=1、λ2=0.3 与一个 development physical input。两条路线通过同一 acceptance，但 amplitude estimates 不同（0.0065 对 0.0245 或 0.0109）。
- Counts 的数值依赖所选 configuration。按 manuscript 表格的同一 counting rule，由 `E4_H_trig_o1_s1_r10_N96_schedule.json` 可得 H_trig r10 o1 s1 为 182 个 primitives（60 CD、82 Ry、40 Rz），少于 262，但它没有做 128/160 复核。G_trig 的 784 与 1104 各含 102 个 merged Gaussian primitives。G_trig 只有 r∈{10,20} 两个取值，中间的 r 是否能以更少 primitives 达标未核实。262 与 784 适合称为 representative counts，不宜写成 count ratio。

### 让其他路线通过会增加什么

- H_SNAP 通过后，可以在同一 native route 上比较不用 qubit 的 residual compiler 与 trig residual。若通过需要 N_acc 约 32–48 和更大的 L，这本身是资源结论，说明 Fock-local compiler 与 φ-local residual 在该 state 上不匹配（未核实）。它不改变 H_trig 与 G_trig 的比较结论。
- Duffing compiler 通过后，可以得到在声明 family 上达标的 native-plus-correction 实例和同精度的 H/S_K pair。S_K 比 H 只多 r+1 个 SNAP，占全部 SNAP 的 (r+1)/(rL+r+1)，r=10、L=8 时约 12%。Kerr capability 结论由解析构造决定，不因此改变。
- 在同一个 development input 上扩大 size 或 optimizer，直到每条路线都通过，会把比较变成调参，结论也只对这个 input 成立。比较的价值在于正确归因失败，并在 held-out inputs 上检验。SG-A 值得运行，因为成本以分钟计，并能把一个信息量很低的失败变成有信息量的检验。D1 只在需要正面 Duffing 结果或需要确认 capacity 时再运行。

### 必须完成的质量检查

1. 19:51 读取的 `main_pra.tex` 在 H_SNAP 段落中已说明 parity 机制与 training-space coverage。Conclusion 中 "Direct trigonometric correction succeeds, while all six tested SNAP corrections fail." 仍把六个失败并列为 SNAP correction 的结果，`pra_revision_notes.md` 的 formal SG 段落也仍写 "All six SNAP circuits fail"。两处应说明其中四个是 zero-displacement 集合上的 number-diagonal 解，并把 H_SNAP 结果称为受 search 限制的结果。这不需要新计算。
2. 同一版本的 Duffing 段落（以 "The controls locate a separate source of error" 开头）已加入 45/48 与 training-coverage 解释。可以再补充逐步误差随输入转入训练 arc 而下降这一直接证据，并说明原来的八个 inputs 已成为 development。
3. SG counts 只称 representative，或者按预先声明的"grid 内 count 最少的通过配置"做 128/160 同参数复核。后者是 deterministic 计算，参照 sg160 三个 circuits 的 163.72 s，约需数分钟。
4. SG 的全部 circuit evidence 来自一个 development input。能实质改变信心的最小检查，是在预先声明的第二个 physical input 上用固定的 trig configurations 复核。成本约为一次 reference 加几个 circuits，参照 96/128 reference 记录的 357 s，约 10–20 分钟（估计，未核实）。通过则结论不依赖这一 input，失败则须缩小结论。
5. 任何 rerun 都报告全部 restarts、initialization rule、bound activity、loss 历史与 training-cutoff consistency，并带上修改后 owners 的 hash 与一次 regression evaluation。已打开的 inputs 不再称为 held-out。

## 推荐的最小下一步

先运行 SG-A。它只改 initialization，沿用原 18 个 restart slots 以及全部尺寸与 criteria，optimizer 与 validation 预计各需数分钟。运行前在新的 protocol 文件中声明 initialization 分布和第二个 physical input，并用修改后的 owners 对一个原 selected parameter vector 做 regression evaluation。SG-A 的结果决定是停止、转 SG-B（training space）还是转 SG-C（capacity）。Conclusion 与 revision notes 中 H_SNAP 的表述修正不需要新计算，可以同时进行。Duffing D1 在 SG-A 之后按需运行，它的 covering family 必须配合打破 parity 对称的 initialization。
