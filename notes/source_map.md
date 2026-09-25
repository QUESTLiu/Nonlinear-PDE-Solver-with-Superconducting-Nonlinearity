# 新稿与原稿 main.tex 的对应关系

本文件说明 `manuscript/pra_v2/main.tex` 如何由 `manuscript/main.tex` 整合而来，包括原稿每一部分的去向、改动原因、新增内容及其数据来源，以及需要作者决定的事项。原稿 SHA256 为 `21cb92539b7e775e94bc637ae0dcfdc742ecb91d1e76d5e6121fd3836ad7160b`，注释掉的内容按惯例视为已删除。

2026-09-25 根据两份外部审稿意见又改了一轮，改动逐条列在第八节。改动前的版本（源文件、PDF、本文件和画图脚本）保存在 `previous/2026-09-25_before_external_reviews/`。

## 一、编译与复现

- 编译：在本目录运行 `/usr/local/bin/tectonic main.tex`（REVTeX 4.2，`pra`，双栏），参考文献为 `references.bib`，共 37 页。
- 环境：`/Users/zhen002/miniconda3/envs/cvdv/bin/python`，即 Python 3.13.0、NumPy 2.3.4、SciPy 1.16.3、Matplotlib 3.10.7。
- 重画全部数据图：在仓库根目录运行 `/Users/zhen002/miniconda3/envs/cvdv/bin/python manuscript/pra_v2/scripts/make_figures.py`。该脚本只读 `pra/results` 和 `data/` 中的记录，不跑模拟。Fig. 1 是 TikZ，源文件为 `figures/fig_overview.tex`。
- `scripts/` 中的计算脚本及其输出（均在 `data/`）：
  - `duffing_amplitude_scan.py`：Duffing 振幅细扫描，秒级。
  - `derived_numbers.py`：SG 经典校验和多项式阶数表的数值，秒级。
  - `class_diagnostics.py`：η_H 与 C_corr 随 cutoff 的变化（附录 B），秒级。
  - `kerr_apriori_bound.py`：Kerr 先验界 Eq. (df-apriori)，秒级。
  - `splitting_baselines.py`：双 sine-Gordon 各种 product formula 的外层 splitting 误差，含步数扫描、cutoff 64/80 对照和 λ2 = 0.15 检查，约 95 秒，输出 `splitting_baselines.json`，对应 Fig. 6(b)。
  - `splitting_sweep.py`：λ2 ∈ {0.1, 0.2, 0.3, 0.45, 0.6}（E_J = 1）与 κ ∈ {0.5, 1, 2} 的 splitting 误差比，local cutoff 64，约 100 秒，输出 `splitting_sweep.json`，对应 Fig. 6(c) 的曲线。
  - `dsg_step_scan.py`：三条路线（hybrid、Gaussian-block、kinetic–potential）的编译电路在 cutoff 128/160 上按步数扫描并按同一验收规则判定，输出 `dsg_step_scan.json`（λ2 = 0.3，明细在 `dsg_step_scan_parts/`），对应 Table III、Fig. 5(c) 和 Fig. 6(a)。阶数代码 1、2 表示所有谐波用一阶或二阶 block，代码 3 表示 Josephson 谐波用二阶、λ2 谐波用一阶（只用于 programmable 路线）。参考轨迹缓存在环境变量 `DSG_SCAN_CACHE` 指定的目录。
  - `dsg_step_scan_lambda.py`：用同一套电路和验收规则在其他 λ2 上扫描，输出 `dsg_step_scan_lambda0.1.json` 和 `dsg_step_scan_lambda0.6.json`，对应 Table III 和 Fig. 6(c) 的星号。
  - 三个 λ2 的全部编译扫描、混合阶数扫描和第二初态检查合计约 40 分钟单机计算（多进程并行，每进程 2 到 3 个 BLAS 线程），峰值内存约 2.3 GB。
  - `dsg_chain.py`：任意格点数的开链版本，逐个格点作用算符，时间演化用自适应 Lanczos 传播子。它在两格点上复现了 `dsg_step_scan.py` 的结果（`data/dsg_chain_parts_N2_cut128_160/`），三格点结果在 `data/dsg_chain_N3_cut128_160.json`。三格点参考轨迹在 cutoff 128 和 160 上分别用了 12 分钟和 26 分钟，programmable 电路每个约 1 分钟，hybrid 电路每个约 1 小时（native 步要对 4.1×10⁶ 维的态做 Lanczos 传播）。参考轨迹约 9 GB，缓存在 `DSG_CHAIN_CACHE` 指定的目录。
  - `dsg_protocol_checks.py`（2026-09-25 新增）：两项检查。`common` 在共同读出时间 t_k = kT/20 上重新判定电路，每个读出时刻用 ⌈t_k/h⌉ 个等长外层步单独构成一条电路，输出 `dsg_common_times.json`（明细在 `dsg_common_times_parts/`）。`gauss` 在 Gaussian-block 电路实际进入 Gaussian 因子的态上，比较分解形式 𝒢_φ𝒢_Π𝒢_φ 与截断矩阵的精确块，输出 `dsg_gauss_factor_check.json`。hybrid 电路每个约 4 分钟，其余每个几秒。
  - `dsg_thresholds.py`（2026-09-25 新增）：读取全部步数扫描记录，给出阈值收紧到 5×10⁻⁴、2.5×10⁻⁴、10⁻⁴ 时各路线最便宜的电路，输出 `dsg_thresholds.json`，对应附录 K 的 Table VII。为此在三个 λ2 上补跑了更大的步数，programmable 电路每个约 2 秒，hybrid 每个约 30 秒，加上每个新步数约 1 分钟的参考轨迹。
  - `dsg_curvature.py`（2026-09-25 新增）：沿 native、target、retuned 三条轨迹平均的 onsite curvature，输出 `dsg_curvature.json`，约 1 分钟。
  - `dsg_boundary_check.py`（2026-09-25 新增）：在 cutoff 160/192 上重新判定 λ2 = 0.6 时只因 padding 128 边界判据不通过的 Gaussian-block 电路，输出 `dsg_boundary_check_lambda0.6.json`，约 5 分钟。
  - `trig_counts.py`（2026-09-25 新增）：三角门的新计数。把 qubit 换基 B = R_y(π/2)R_z(π/2) 移到整条电路首尾后，每个基本因子只需 2 个 controlled displacement 和 1 个 R_y。`verify` 检查合并后的门序列与精确三角块一致（3×10⁻¹⁵），并检查新计数等于 “原 CD + 原 R_z + Gaussian + 4”。`recount` 重写 `data/` 中全部已存电路的计数，原计数保留在 `primitive_counts_conjugated` 字段。`dsg_step_scan.py` 和 `dsg_chain.py` 的计数函数已改用它。秒级。
  - `dsg_fock_tail.py`（2026-09-25 新增）：两格点与三格点 target 在 cutoff 128 上 “某个模式超过 level 60” 的布居，输出 `dsg_fock_tail.json`，约 1 分钟。

## 二、蓝色标记

`main.tex` 中凡是原稿正文里没有的文字都标成蓝色，包括新写的句子、改写过的句子、新增或改动的公式、全部图注和表格，以及从 backups 取来的内容。原稿原句保持黑色，只移动了位置的原句也保持黑色。标记由 `scripts/mark_changes.py` 生成。它逐句、逐个行间公式与原稿正文比对，比对时忽略空白、公式里的排版间距，以及行间与行内公式的格式差别。只要措辞有改动，哪怕只是把连字符改成 en dash 或把 n̂ 改成 N̂，整句都会标蓝。

- 去掉全部标记：把导言区的 `\newcommand{\revcolor}{\color{blue}}` 改为 `\newcommand{\revcolor}{}`。
- 重新生成标记：`python manuscript/pra_v2/scripts/mark_changes.py manuscript/pra_v2/main_unmarked.tex manuscript/main.tex manuscript/pra_v2/main.tex`。`main_unmarked.tex` 是不带标记的干净源文件，以后的修改应在它上面进行。
- 删除的原句不会显示为蓝色，下面第五节单独列出。

## 三、原稿各部分的去向

| 原稿部分 | 新稿位置 | 处理 |
|---|---|---|
| 标题 | 标题 | 改为 “Native nonlinear interactions versus programmable emulation in hybrid oscillator–qubit simulation”，原因和原题见第七节 |
| 作者与 “PNNL” | 作者与 affiliation | 写成 “Pacific Northwest National Laboratory, Richland, Washington 99354, USA”，需作者确认 |
| 摘要（占位） | 摘要 | 新写 |
| 目录、TODO before submission | 删除 | 这是工作清单，其中可以落实的事项已在正文完成 |
| Introduction 前三段 | Sec. I | 原文保留。人名连写改为 en dash（Koopman–von Neumann、Fokker–Planck、oscillator–qubit），补了 Gan 2025 的相关工作 |
| 研究问题 | Sec. I | 中间一句改为 “哪些部分必须由 programmable correction 提供”，并在末尾加上 “native 演化何时比 programmable 模拟更省”。M_hw(f,ε) 保留在附录 A |
| 字典介绍段 | Sec. II A 开头 | 原文保留，从引言移到 Sec. II A |
| Framework：约瑟夫森哈密顿量、exact 与 trig 字典、Kerr 与 number 字典、multimode 字典、字典并集 | Sec. II A | 原文保留 |
| Cooper-pair 平移、transmon 展开、polynomial 字典、flux-biased 字典、mixing 字典 | 附录 A 1 | 原文保留，移到附录以缩短 Sec. II |
| classical hardware dictionary 与向量场展开 | 附录 A 2 | 原文保留，修正 f ≈ f_j 的笔误 |
| Classes of nonlinear DEs | 首段和 boxed 方程类在附录 A 2，六个小节在附录 A 3 | 原文保留。正文 Sec. II A 保留新写的 Hamiltonian realizability 一段，它说明 boxed 类是字典可以表示的类，闭合电路直接实现的是其中的 Hamiltonian flow |
| Extension to linear dissipative sectors（LCHS） | 附录 C，正文 Sec. II A 末尾留一段指引 | 基本保留原文（29 句中 13 句逐字不变，其余多为公式排版或引用写法的差别），修复断开的 LaTeX。几处实质改动列在第五节。新增：postselection 概率在无驱动时随光子数指数下降，归一化后误差会被输出范数放大 |
| Encoded field and validity | Sec. II B | 原文保留。a†a 的记号统一为 N̂，Kerr 坍缩公式加 Milburn 1986 与 Yurke–Stoler 1986 引用。Gronwall classical-transfer 界在附录 E，正文留一段说明 |
| Operator/numerical/resource conventions | η_H 与 C_corr 在 Sec. II C，calibration、model uncertainty 和 tolerance 推导在附录 B，resource accounting 在 Sec. II E | 均为原文。新增 I*_traj = 10⁻³、三分之一分配只用于分类阈值，以及 class 条件依赖 cutoff、η_H 不衡量修正对具体输入的影响 |
| Resource-bounded simulability 定义与报告清单 | 附录 D | 定义原文保留。Sec. II E 留一句指引 |
| Classes of native match | 正文 Sec. II C 用一段话给出两类的条件，正式定义在附录 B | 定义原文保留。原表 Class I 写 “ΔH = 0”，与定义 η_H ≤ ε_nat 不一致，已按定义改正 |
| Unitary benchmarks 引言段 | Sec. III 开头 | 原文保留，节标题改为 “Benchmarks”，“两个例子” 改为 “三个例子”，并说明电路比较采用理想 gate model |
| SG：模型与电路 | Sec. III A 1 | 原文保留。新增 compact 与 extended phase 的适用条件，以及 shunt 电感带来的 onsite 二次项 |
| SG：quantization 与 encoding | Sec. III A 2 | 原文保留。补充：两格点开边界下 √(E_J+2κ) 是反对称 normal mode 的频率，对称模频率为 √E_J。删去 “No bound-state-count estimate is used in the analysis.” |
| SG：classical validation | Sec. III A 3 | 叙述原文保留，数值按存档更新。改正三处与数据不符的说法（见第五节）。新增阈值敏感性（ε* 从 0.10 到 0.11 时 τ_E 从 2.47 移到 3.13）、固定方差对照的 cutoff 变化，以及 Fig. 2(d) 的 residual 分解 |
| SG：polynomial synthesis | 附录 F，正文 Sec. III A 3 末尾留一段结论 | 原文保留，阶数表与记录一致 |
| Duffing：约化与分解 | Sec. III B 1 | 原文保留，a†a 记为 N̂。新增：负 Kerr 也可以通过复共轭实现同一动力学，因此不再需要正 Kerr 假设 |
| Duffing：benchmark 与修正大小 | Sec. III B 2 | 原文与数值保留。新增三个 “inconclusive” 点的直接模拟 infidelity 均低于 10⁻²，以及 Duffing d_MF 的明确定义 |
| Duffing：compiling | Sec. III B 3 | 原文保留，补出 SNAP+D 公式。22 个测试输入明确分为 10 个 development inputs 和 12 个 held-out 输入，并说明 held-out 振幅在训练范围内。新增 composition bound 为 0.053（不能认证该电路）、Kerr 节省按全部 primitives 计为 271 个中的 11 个，以及替换不增加编译成本 |
| 附录：Duhamel 推导 | 附录 E 最后一段 | 原文保留 |

## 四、新增内容

先说明来源。下列结果在原稿 `main.tex` 里没有，但它们来自 Codex 阶段存档在 `pra/results` 的计算，Codex 稿 `backup_codex/main_pra.tex` 已经包含：Theorem 1 的 native-reference residual bound、composition bound 和 classical transfer bound，Duffing 在 22 个输入上通过的编译电路，120 个 SNAP 对 131 个 SNAP 的替换及较高 padding 下的稳定性检查，以及 SG 的固定方差对照。本版改写和重新组织了这些结果，但它们不是本版新算的。本版新算的是下面各项。

- **Sec. III C 双 sine-Gordon。** 这是改动最大的部分。
  - 三条路线在同一验收规则（cutoff 128/160）下扫描步数和 block 阶数，得到各自最少的 primitive 数（Table III）：

    | λ2 | hybrid | Gaussian-block | kinetic–potential | 对 Gaussian-block 的节省 | 对 kinetic–potential 的节省 |
    |---|---|---|---|---|---|
    | 0.1 | 54 | 181 | 330 | 3.4 倍 | 6.1 倍 |
    | 0.3 | 64 | 209 | 330 | 3.3 倍 | 5.2 倍 |
    | 0.6 | 94 | 293（见下） | 357 | 3.1 倍 | 3.8 倍 |

    λ2 = 0.3 时 controlled displacement 数的节省为 2.3 倍和 4.0 倍。九个最便宜电路在第二个初态上全部通过。

    以上计数都已按第九节的新门序列重算，被选中的电路（步数与阶数）没有变化。λ2 = 0.6 的 293 需要说明。混合阶数的 Gaussian-block 电路在 r = 10 到 12 时满足状态和观测量判据，只是在 padding 128 上顶端四个能级的布居略超 10⁻⁶（padding 160 上约 1.5×10⁻⁷）。在 cutoff 160/192 上重新判定，这三条全部通过（`scripts/dsg_boundary_check.py`，`data/dsg_boundary_check_lambda0.6.json`）。为了对 baseline 公平，Table III 和 Table VII 采用这些电路并以星号标注。r = 10 这条在 160/192 上也通过了第二初态和共同读出时间的检查，r = 9 在共同读出时间上仍不通过。只用 128/160 规则时，λ2 = 0.6 的最便宜 Gaussian-block 电路是二阶、r = 9、301 个 primitive，节省为 3.2 倍。
  - Gaussian-block 电路是新加的更强 baseline。它把 kinetic 与 bond 作为一个两模 Gaussian unitary 实现，分解为 𝒢_φ𝒢_Π𝒢_φ，每一步计 6 个 Gaussian primitive（附录 J）。programmable 电路还允许两个谐波用不同的 block 阶数，这使它们在 λ2 = 0.1 和 0.3 时更便宜。
  - 机制：hybrid 的 splitting 误差约与 λ2 成正比。kinetic–potential split 的 7 倍误差大多来自把 kinetic 和 bond 拆开，最强 programmable split 的误差比只有 2.1 到 2.2。
  - 参数依赖：splitting 误差比从 λ2 = 0.1 时的 4.7–4.9 降到 0.6 时的 1.55–1.59，几乎与 κ 无关。但编译层面的节省几乎不随 λ2 变化，因为主要来自 programmable 电路每一步都要编译 Josephson 谐波（同步数下三个 λ2 都约为 2.8 倍），步数差最多一步。
  - 只在 r = 10 和 20 上的比较放在附录 J 的 Fig. 9。在 cutoff 128/160 下，这两个步数的最便宜电路是 hybrid 104 和 kinetic–potential 466。
  - 三格点开链（λ2 = 0.3，初态 ⟨φ⟩ = (1, 0, 0)，cutoff 128/160）：hybrid 94（r = 6），Gaussian-block 382（r = 8），kinetic–potential 540（r = 13），节省 4.1 倍和 5.7 倍。hybrid 的状态误差 0.0315 只比阈值低一点。2026-09-25 补算了多一步的 hybrid 电路（r = 7）：109 个 primitive，状态误差 0.0187，比阈值低约 40%，节省为 3.5 倍和 5.0 倍，与两格点接近。正文现在以 r = 7 为代表，r = 6 作为贴近阈值的边界结果。三格点没有做第二初态检查。
- **2026-09-25 为回应审稿新做的检查**（数据见第一节列出的脚本）：
  - 阈值收紧：把 infidelity 阈值从 10⁻³ 收紧到 5×10⁻⁴、2.5×10⁻⁴ 和 10⁻⁴，其他判据不变。hybrid 在每个阈值上都是最便宜的路线，对 Gaussian-block 的节省在 2.6 到 4.6 倍之间。λ2 = 0.1 和 0.3 时节省随阈值收紧而增大，在 10⁻⁴ 时分别为 4.6 倍和 3.7 倍。λ2 = 0.6 时节省降到 2.6 到 2.7 倍，因为这时两种 splitting 误差只差 1.6 倍，而且 hybrid 要改用二阶 block。λ2 = 0.6 在 10⁻³ 到 2.5×10⁻⁴ 三个阈值上的 Gaussian-block 电路用的是 cutoff 160/192 的判定（见上）。完整结果见附录 K 的 Table VII。
  - 共同读出时间 t_k = kT/20：三个 λ2 上，Table III 中最便宜的电路在共同读出时间上仍然通过，少一步的电路仍然不通过，所以最便宜的计数不变。λ2 = 0.3 时三个最便宜电路的状态误差为 0.0271、0.0286、0.0288，在各自网格上为 0.0271、0.0298、0.0307。三格点没有做这项检查。
  - Gaussian 分解在实际进入态上的误差：λ2 = 0.3 最便宜电路的 14 个进入态上，分解与截断矩阵精确块之差在 cutoff 128 时最大 6.7×10⁻⁴，在 cutoff 160 时最大 2.6×10⁻⁴。λ2 = 0.1 和 0.6 在 cutoff 128 时最大 5.9×10⁻⁴。原稿写的 3×10⁻¹⁵ 只对初态成立。电路执行的就是分解形式，所以 Table III 的误差已经包含这一差别。
  - onsite curvature：沿各自轨迹对时间和两格点平均，native、target、retuned 分别为 0.66、1.02、1.63，与 |C12| 和 S2 的排序相反。这支持 “onsite 势越硬、bond 的相对作用越弱” 的直观解释，但逐点的 curvature 并不按这个顺序排列，所以正文只把它写成观察。
- Sec. IV Discussion：硬件模型一段说明 native lattice 演化只有几十到几百 ps，而 ECD 的脉冲为 24 ns 和 44 ns，因此 Sec. III C 是理想 gate model 下的比较。本轮又把这一假设提前写进引言。
- Sec. V Conclusion、Data availability。
- 附录 B 到 L：native-match 的正式定义与截断依赖、LCHS、simulability 定义、证明、多项式阶数、数值方法、Duffing 训练覆盖、SNAP+D 失败的宇称原因、trig gate 与 Gaussian block 的构造和计数、参考解收敛与验收检查、Lean 形式化范围。
- 新引用（均已核对 DOI 或 arXiv）：Milburn 1986、Yurke and Stoler 1986、An–Fang–Lin 2021、Şahinoğlu–Somma 2021、Burgarth et al. 2024、Crane et al.（PRX accepted）、Gao–Pfister–Bekiranov 2025、Park and Light 1986、Hochbruck and Lubich 1997。

## 五、删除或改正的原句

以下原句在新稿中被删除，或因与数据不符而被改写。删除的内容在蓝色标记中看不到，搬运到 Overleaf 时需要手动处理。

- 删除 “No bound-state-count estimate is used in the analysis.”，因为正文从未引入这一估计。
- 改正 “This choice ensures that each parameter point is evolved over the same number of characteristic oscillation periods rather than the same laboratory time.”。它只对反对称模成立，对称模在窗口内只走完 0.28 到 0.80 个周期。
- 改正 “The corresponding physical times decrease with increasing E_J …”。数据显示 E_J 从 2 到 4 时物理时间从 1.14 升到 1.87。
- 改正 “The evolution time is scaled by the local harmonic frequency so that each benchmark covers the same number of oscillation periods …”，理由同上。
- “Each benchmark must report …” 与 “The diagnostics (η_H, C_corr) are reported with every benchmark.” 改为陈述句，去掉规定语气。
- “exact-V̂4 symmetric-split control” 一段改写。原句中的 “control” 与 control operation 冲突，原句的分号也拆开了。
- Definition 1 中的 “only if the stronger condition … holds” 改为 “if”。原句只给出必要条件，定义因此不完整。“declared” 改为 “stated”。
- Definition 3 中的 “declared accessed space” 改为 “truncated system space”，因为前者在全文没有定义。
- 附录 D 的 (i)–(ix) 报告清单改写为两句正文。原清单中的 “fixed-scale map” 和 “conditional state error” 没有定义。
- 引言中 “follows a density-based route [10]: it maps …” 在冒号处拆成两句。
- LCHS 一节（现附录 C）：原句 “This includes an important class of spatially discretized nonlinear wave and reaction–diffusion models for which the local nonlinear interactions are represented natively while diffusion, attenuation, or other linear dissipative processes are treated through the one-oscillator dilation.” 改为 “Spatially discretized nonlinear wave models with linear loss, such as the lossy Kerr array below, have a generator of this form.”，并补一句说明经典读出需要另外论证。原因是 reaction–diffusion 的反应项本身是非线性耗散，不属于 “非线性 Hermitian 部分加线性半正定耗散” 这一类。
- LCHS 一节：原文重复写了一遍 H_native = Σ c_μ D_μ 的分解和 ΔH 的定义，改为引用 Sec. II C 的 Eq. (native-hamiltonian-dictionary)。“Consequently, it belongs to the Hermitian sector …” 并入前一句，“Throughout this subsection” 改为 “Throughout this appendix”。
- 本轮：三个 benchmark 小节标题去掉了 “Class I–U Example:” 与 “Class II–U Example:” 前缀，Sec. III 的标题 “Unitary benchmarks within the general framework” 改为 “Benchmarks”。
- 本轮：原句 “In the transmon regime E_J ≫ E_C, fluctuations of the phase are localized near a minimum of the Josephson potential.” 在正文中与一个新从句合并，因此标蓝。它后面的展开式移到附录 A 1。
- 本轮：“A native-match criterion is meaningful only relative to …” 所在的 tolerance 段落移到附录 B，其中开头一句和最后三句保留在正文 Sec. II C，附录里不再重复。
- 本轮：SG 小节的 “The contribution studied here is the comparison between native analog evolution … and programmable synthesis of the same nonlinear sector.” 改为说明本小节校准经典读出、比较放在 Sec. III C，因为 Table I 标明 SG 没有做这一比较。
- 本轮：附录 A 中 “for the proposed hardware” 改为 “for Josephson hardware”，两处 “for/to the proposed architecture” 改为 “native Josephson simulation”，因为新标题和引言不再把本文写成提出一种硬件架构。

## 六、图

所有图都是新画的。

| 图 | 内容 | 数据来源 |
|---|---|---|
| Fig. 1 | 概念示意（TikZ），含 E0 项和 H_kin 标记 | 无数据 |
| Fig. 2 | SG 经典校验，含固定方差对照，(d) 为 E_J = 1 时的 closure、non-Gaussian 与 finite-cutoff residual | `pra/results/formal/E3` |
| Fig. 3 | Duffing native 误差、界和网格 | `data/duffing_amplitude_scan.json`，`formal/E1_retry` |
| Fig. 4 | Duffing 修正，10 个 development 与 12 个 held-out 输入 | `duffing_refinement/comparison` |
| Fig. 5 | 双 SG 观测量（\|C12\| 与 S2）及 λ2 = 0.3 时三条路线最便宜电路的误差 | `sg128/reference`，`data/dsg_step_scan.json` |
| Fig. 6 | 步数扫描的资源对比（含混合阶数）、splitting 误差随步长变化、splitting 误差比随 λ2 与 κ 的变化，以及三个 λ2 上编译电路的 primitive 比 | `data/dsg_step_scan*.json`，`splitting_baselines.json`，`splitting_sweep.json` |
| Fig. 7（附录 F） | 多项式阶数 | `pra/results/formal/polynomial` |
| Fig. 8（附录 H） | Duffing 训练覆盖 | `formal/E2`，`duffing_refinement` |
| Fig. 9（附录 J） | 只限 r = 10、20 的比较，s 从 1 到 32 | `pra/results/analysis_summary.json` |

所有图例都已逐张渲染检查，不遮挡数据或参考线。Fig. 2(d) 用曲线旁的文字标注代替图例。原稿的 figures/ 和 figures_sg/ 未改动，新稿不再使用它们，因为这些图的原始数据没有存档。

## 七、需要作者决定的事项

1. 标题。原题为 “Hardware-native hybrid oscillator–qubit simulation of nonlinear dynamics”。两份审稿都指出，“hardware-native” 许诺的硬件层次高于文章实际使用的理想 gate model（native 演化可在门之间开关），因此改为现题，作者已于 2026-09-25 同意。若以后要恢复原题，需要补一个 native 项、可控项、能标和时序都自洽的有效控制模型。
2. 四位作者的 affiliation 是否都写 PNNL，以及地址格式。
3. Data availability 按作者要求写的是本仓库的 GitHub 地址 https://github.com/ElinRanjan/nonlinear-PDE-simulator ，并列出仓库内容。2026-09-25 检查时，该仓库未公开（未登录访问返回 404），`.gitignore` 忽略了整个 `manuscript/` 目录，`pra/`（其中 `pra/results` 约 1.2 GB，单个文件都小于 50 MB）和 `sine_gordon/native_correction.py` 也还没有提交。投稿前需要提交并推送这些文件，至少包括 `pra/`、`manuscript/pra_v2/scripts`、`manuscript/pra_v2/data`、`sine_gordon/` 的改动和 `cvdv/`，并把仓库设为公开，否则审稿人打不开链接，也找不到声明里说的脚本和数据。GitHub 链接不是持久标识，可以考虑从某个 release 生成 Zenodo DOI 附在后面。
4. 缺少 Acknowledgments 和 funding 信息。
5. 篇幅：共 37 页，正文约 21 页。本轮已把字典推导、native-match 的细节和多项式比较移到附录，第一张数据图从第 13 页提前到第 10 页，资源图从第 21 页提前到第 18 页。
6. 硬件假设：lattice 比较假设 native Hamiltonian 能在门之间开关，而固定电容的 Josephson 阵列做不到。引言和 Sec. IV C 都已写明这是理想 gate model 下的比较，作者可以决定是否补充一个可行平台的讨论。
7. 编译电路已在两格点的 λ2 = 0.1、0.3、0.6 和三格点的 λ2 = 0.3（κ = 1）上验证，κ 的依赖只来自 splitting 误差。三格点的第二初态检查和三格点的共同读出时间检查没有做，主要成本在 hybrid 电路，每条约 1 小时。
8. 附录 L 的 Lean 描述：已在源码中确认没有 `sorry`、`admit`、`axiom` 和 `native_decide`，但本次没有重新编译 Lean 项目，所列 formalized statements 的清单也请作者核对。

## 八、2026-09-25 外部审稿意见的处理

两份意见分别来自 ChatGPT 6 Pro（对比本稿与 Codex 稿）和 Codex（审阅本稿）。下表列出每条意见的处理。

| 意见 | 来源 | 处理 | 位置 |
|---|---|---|---|
| 标题和引言的硬件承诺高于理想 gate model | 两份 | 改标题。引言新增一段，前置说明 native 演化可开关、理想 primitive、extended phase、Kerr 的符号与 rotating frame，并指出这是 gate model 内的资源比较 | 标题，Sec. I |
| 收紧开篇的一般非线性方程框架，结果前移 | 两份 | 引言加 Table I，第 2 页即列出三个 benchmark 的 native 部分、其最便宜的 programmable emulation 和结果。字典推导、native-match 细节和多项式比较移到附录 A、B、F，图放到首次讨论处附近 | Sec. I，Sec. II，附录 A、B、F |
| 与 Roy 等、Rainaldi 等、Zoufal 等的关系 | ChatGPT | 引言分两段写：已有 analog 实现和 programmable 构造各自存在，本文的新内容是相同精度下的成本比较，以及 “节省由最便宜的 emulation 决定” 这一结论。Theorem 1 写成对 Zoufal 等 Duhamel 估计的应用 | Sec. I |
| Class I/II 与 η_H 篇幅过大 | ChatGPT、Codex | 正文只留一段条件和一张表，正式定义、calibration 变体和 tolerance 推导移到附录 B | Sec. II C，附录 B |
| 不同电路在不同时间网格上取最大误差 | ChatGPT | 在共同读出时间 t_k = kT/20 上重新判定，最便宜电路不变，每个更便宜的电路仍不通过。正文写明验收只在有限个时刻检查 | Sec. III C，附录 K |
| 需要一个有明确误差余量的工作点 | ChatGPT | 阈值从 10⁻³ 收紧到 10⁻⁴ 的扫描，见第四节 | Sec. III C，附录 K |
| 三格点 3.8 倍贴近阈值，不应承担 scaling 结论 | 两份 | 补算三格点 hybrid r = 7。摘要、讨论和结论改为给出范围，并说明不是规模趋势 | 摘要，Sec. III C，IV A，V |
| “0.1 到 0.6” 应写成三个测试强度 | 两份 | 摘要、讨论和结论都改为三个测试值 | 摘要，Sec. IV A，V |
| “only when the initial phase spread shrinks” 超出两个对照点的支持 | Codex、ChatGPT | 改为陈述两个观察：主扫描中窗口随 E_J 变长，固定方差的两个较大 E_J 上窗口变短 | 摘要，Sec. V |
| V''_target − V''_native = 4λ2 cos 2φ 会变号，stiffness 解释过强 | 两份 | 改为：两个观测量随沿轨迹平均的 curvature 增大而减小（0.66、1.02、1.63），这与直观解释一致但不构成证明，Eq. (retuning) 本身不决定排序 | Sec. III C 2 |
| Gaussian shear 分解是 canonical 恒等式，不是截断矩阵恒等式 | 两份 | 改写，并在实际进入态上检查分解误差（见第四节） | Sec. III C 3，附录 J |
| LCHS 的 first-commutator 论断 | Codex | 限定为 C_LCHS 这一领头诊断，并指出二阶 product formula 的 nested commutator 含 Kerr 项 | 附录 C |
| 保留旧稿的 residual 诊断图 | ChatGPT | Fig. 2 新增 (d)：closure residual 最大 0.18，non-Gaussian 部分 0.061，finite-cutoff 部分低于 4×10⁻⁵ | Sec. III A 3 |
| S2 在有 ancilla 残余关联时不能单独解读为纠缠 | ChatGPT | 在 ancilla 一段补一句 | Sec. III C 3 |
| 复现声明应具体 | ChatGPT | Data availability 改为本仓库的 GitHub 链接并列出内容。仓库需在投稿前公开并提交相关文件（见第七节第 3 条） | Data availability |
| 编译（优化）成本未计入 | 作者本人 | 资源一节说明 primitive 计数不含 classical compilation。Kerr 替换的 SNAP 相位是解析的，U_comp 两条线路共用，所以替换不增加编译成本 | Sec. II E，Sec. III B 3 |

未采纳或只部分采纳的意见：ChatGPT 提出的另一条路，即给出一个同时满足全部假设的物理实现模型，本轮没有做，改为调整定位（见第七节第 1 条）。Framework 已压缩，但字典一节、引言的一般性开头和四部分研究问题仍保留原文，是否再删由作者决定。

### 独立审稿（2026-09-25，改稿后）

改稿后又请一位独立审稿人按 PRA 审稿人的角度通读全文，并把正文、摘要、Table I、III、VII 和附录 K 的每个数字对照数据文件。所有资源数字都与数据一致。它指出的问题和处理如下。

| 问题 | 处理 |
|---|---|
| Data availability 指向的仓库未公开，且 `manuscript/` 被 `.gitignore` 忽略 | 按作者要求保留 GitHub 链接，所需的提交和公开见第七节第 3 条 |
| 摘要和结论说窗口 “随 E_J 变长”，数据并非单调（E_J ≤ 2 时 τ_E = 2.3–3.2，E_J ≥ 4 时 4.5–4.9） | 摘要、结论和 Sec. III A 3 改为给出这两段数值和固定方差对照的 1.9、1.8，删去 “require” 的说法 |
| r = 10 处出现 182 和 262 两个 hybrid 计数，没有说明 | 写明 182 是 cutoff 128/160 下的一阶电路，502 对 262 是 cutoff 96/128 下的二阶对二阶 |
| 附录 J 说 128/160 下所有电路都满足边界判据，λ2 = 0.6 时不成立 | 改正，并在 160/192 上重新判定三条只因边界不通过的 Gaussian-block 电路，全部通过，节省改为 3.0（见第四节） |
| 讨论中 “机制取决于对易子结构而非格点数” 与正文矛盾 | 改为：大部分节省来自每步少编译一个谐波和 Gaussian 门，较小部分来自 splitting 误差，二者都不决定随格点数的变化 |
| 引言 “节省由最便宜的 emulation 决定” 说过头，“strengthen the baseline” 像改稿叙述 | 改为 “在两个带修正的 benchmark 中，节省随所测最好 emulation 的成本而定”，baseline 直接描述其构造 |
| Emulation cost 一段说 Gaussian block “exactly” | 限定为 canonical operators |
| Table I 图注对 Duffing 行说 “cheapest” | 改为只对双 sine-Gordon 说 cheapest found |
| 重组后的残留（附录 D 的 “below”、正文引用附录方程、SG 小节提前宣布比较、附录 A 的 “proposed architecture”） | 逐处改正 |
| 正文与附录几处近乎逐字重复 | 删去正文的耗散段重复句、引言的皮秒时间尺度句和附录 K 的重复句 |
| Fig. 2(c) 图例越过右边框，Fig. 1(b) 画的是 kinetic–potential 而非 Gaussian-block | 图例改为 “fixed variance”，Fig. 1(b) 改画 Gaussian block 𝒢 |
| 三格点 “level 60 以上布居小于 6×10⁻⁸” 无数据可查 | 用并集定义重算为 8.7×10⁻⁸（`scripts/dsg_fock_tail.py`），正文改为小于 9×10⁻⁸，附录 G 补上该数和 cutoff 变化 1.7×10⁻⁶ |

## 九、2026-09-25 第二轮 6 Pro 意见与 Codex 复核的处理

6 Pro 对改后版本又审了一轮，Codex 复核了它的意见。两者的主要结论一致，处理如下。

**三角门编译的冗余（必须修，已修）。** 原构造把每个基本因子 E_± 写成 U e^{ivZ} U†，并把 U 分解为 R_y(π/2) CD R_y(−π/2)，所以每个因子都要重复做 qubit 换基。取 B = R_y(π/2)R_z(π/2)，它在共轭下把 Z 变为 X、把 Y 变为 Z，于是 E_±(v) = 𝓑 D^{±1} R_y(−2v) D^{∓1} 𝓑†，其中 D 就是原来的 controlled displacement。相邻因子之间的 𝓑†𝓑 相消，𝓑 又与只作用于系统的 native 和 Gaussian 块对易，所以整条电路只需在首尾各做一次换基。这一改写不改变电路的 unitary，已有的全部误差数据仍然有效，只需要重新计数。
- 验证：有限矩阵上恒等式成立到 10⁻¹⁵ 量级。合并后的门序列（一阶、二阶、混合阶数，s = 1 和 2）与精确三角块一致到 3×10⁻¹⁵（`scripts/trig_counts.py verify`）。同一脚本还检查了两格点和三格点上的整条电路：换基只在首尾各做一次，外步之间插入一个随机的系统 unitary 代替 native 和 Gaussian 块。48 种组合（阶数、s、系数、r = 1 和 3）与原构造一致到 1.5×10⁻¹⁴（去掉整体相位），脉冲数也都与 `trig_counts()` 的计数相同。附录 J 相应补了半句。
- 重计：`data/` 中 258 条电路全部重算，每条的新计数都等于 “原 CD + 原 R_z + Gaussian + 4”。每条路线被选中的电路（步数和阶数）都没有变，所以 6 Pro 给出的 64、209、330 正是新的最便宜电路。
- 新的两格点计数公式：hybrid 一阶 10r+4、二阶 14r+4，programmable 的三角部分一阶 18r+4、二阶 26r+4、混合 22r+4。三格点为 15r+4（hybrid 一阶）和 33r+4（programmable 混合）。Gaussian 计数不变。
- 结果：λ2 = 0.3 时为 64 / 209 / 330（原 110 / 347 / 568），节省 3.3 倍和 5.2 倍。三个 λ2 上对 Gaussian-block 的节省为 3.1 到 3.4 倍。阈值从 10⁻³ 收紧到 10⁻⁴ 的整个范围内为 2.6 到 4.6 倍，恰在 10⁻⁴ 时为 2.7 到 4.6 倍。三格点为 3.5 倍（r = 7，有余量）到 4.1 倍（r = 6，贴近阈值）。controlled displacement 数不变，所以 CD 的节省（2.3 倍和 4.0 倍）不变。r = 10/20 的对照为 hybrid 104、kinetic–potential 466，retuned 与 hybrid 的二阶对照为 264 对 144。
- 已更新：摘要、Table I、III、VII、Sec. III C、IV A、V、附录 J 的构造说明和计数公式、附录 K，以及 Fig. 5、6、9。原计数保留在数据文件的 `primitive_counts_conjugated` 字段中。

**其余意见。**
- 附录 A.2 从分量展开直接跳到 “可以通过调节电路参数实现”，与正文的 Hamiltonian 条件不一致。已在 A.2 补充：对闭合电路，这一说法只适用于 Hamiltonian flow（f_M = J∇h_M），一般的分量展开还需要编码、耗散机制或其他物理构造。原句 “then the hardware implements an approximate nonlinear flow …” 改为 “then, for a realizable f_M, …”，因此标蓝。
- Sec. IV C “有硬件误差就用更多步” 改为：硬件误差需要单独占用误差预算，增加步数虽然降低 splitting 误差，却增加门和切换次数，所以最优步数会变。
- 时间记号：原句 “The dimensionless simulation time and laboratory time are related by τ = 8E_C t_lab/ħ” 改用 t，避免与 τ = Ωt 冲突。Sec. III A 3 的 “physical time” 改为 “simulation time”。
- 摘要从约 390 词压到约 290 词，只保留主要对照、一个代表性资源结果（64 对 209）和适用条件，SG 窗口的具体数值留在正文。
- 引言补了两句，说明理想 gate model 的物理意义。这个模型分离出 native 演化所省掉的那部分编程成本，即 native 相互作用本身的编译，以及它与其余项的对易子迫使 product formula 多用的步数。切换成本只加在每次 native wait 上，残余 drift 则影响两条路线（措辞经下面的独立复核修正）。
- Codex 提醒 PRA 审稿人仍可能追问这一 access model 的物理意义，以及相对已有 digital–analog 思路的新认识。上面补的那句是正面回应，但它不能证明这些能力可以在某个具体平台上同时实现。这仍是作者需要判断的风险（第七节第 6 条）。

**对数值结果的影响。** 改写前后电路的 unitary 完全相同，模拟本来就直接作用精确因子 E_±，所以状态误差、C12 和 S2 误差、boundary 布居和 purity loss 都不变，也不需要重算。分别按新旧计数在三个 λ2 和四个阈值上挑最便宜电路，12 组选择完全相同。三格点通过的电路和每条路线的最便宜电路也相同。唯一与参考系有关的量是 qubit 在各外步时刻的激发布居（小于 5×10⁻⁴），它定义在因子 E_± 的参考系中。编译后的脉冲在首尾两次换基之间把 qubit 留在 𝓑†|0⟩ 附近，所以正文和附录 J 都注明了这个参考系。qubit 偏离其参考态的程度与参考系无关，所以这个数值本身不变。Duffing 电路和 SNAP+D 路线不用三角门，不受影响。

### 独立复核（第二轮改后，2026-09-25）

独立审稿人（Opus）逐项核对了重计后的数字和新句子。Table III 和 VII 的全部计数都与重新生成的数据和计数公式一致，按 route、λ2 和阈值重新挑选最便宜电路（含 160/192 的重评）也复现了 Table VII 的全部 12 行。旧计数和旧比值都没有残留，附录 J 的构造正确。问题和处理如下。

| 意见 | 处理 |
|---|---|
| Sec. IV A 和 Conclusion 说阈值严格十倍时节省为 2.6 到 4.6 倍。恰在 10⁻⁴ 时实际是 4.6、3.7 和 2.7，2.6 出自 2.5×10⁻⁴（λ2 = 0.6） | 两处改为阈值最多严格十倍的整个范围内为 2.6 到 4.6 倍，与摘要和附录 K 的说法一致 |
| 引言新句说切换和残余 drift 给两条路线都带来成本，与 Sec. IV D 矛盾，因为切换只发生在每次 native wait 前后，programmable 路线没有 wait。“编译对易子”的说法也不对，对易子产生 splitting 误差，迫使增加步数 | 改为 native 演化省掉的是 native 相互作用的编译和它的对易子迫使 product formula 多用的步数。切换成本加在每次 native wait 上，残余 drift 影响两条路线 |
| Sec. III C 说 qubit 在每个 potential block 之后回到 \|0⟩ 附近。换基只在首尾各做一次后，编译后的脉冲在块之间把 qubit 留在 𝓑†\|0⟩ ∝ \|0⟩ + i\|1⟩ 附近，而这一句没有指向附录 J 的参考系说明 | 句首加上 “In the frame of the elementary factors, which differs from that of the compiled pulses by the qubit basis change (Appendix J)” |
| 常数 4 包含最后的 𝓑，它只在 qubit 被 trace out 前把 qubit 转回计算基。去掉它，λ2 = 0.3 的节省为 207/62 = 3.34 倍，现为 209/64 = 3.27 倍 | 保留这两个门（结论更保守），附录 J 加一句说明计入最后的 𝓑 会略微降低每个节省倍数 |
| Sec. III C 的 “subdivisions up to s = 32 lower neither count” 依据的是 96/128 的扫描，而在那里 104 个门的 hybrid 电路和 r = 20 的 kinetic–potential 电路都超过 boundary 限 | 句中注明 “in the same comparison at cutoffs 96/128”。已核对：在 96/128 下，比 104 或 466 更便宜的 s > 1 电路只有 r = 10 的 kinetic–potential 电路（s = 2 时 396 个门），它们的状态误差为 0.046 到 0.051，远超 0.032 的阈值，所以结论成立。retuned 电路那句相应改为 “evaluated only in that comparison” |
| 引言的 “The strongest programmable circuits” 含义不清 | 改为 “The cheapest programmable circuits tested” |
| Data availability 的仓库链接仍返回 404，且 `.gitignore` 第 6 行排除了 `manuscript/`，所以 `trig_counts.py` 等脚本和数据不在仓库中。审稿人建议引用带版本的公开存档（DOI 或 tagged release） | 正文未改，作者已决定用 GitHub 链接。投稿前需要提交 `manuscript/pra_v2/scripts`、`data` 和 `pra/` 并公开仓库，也可以考虑为 tagged release 申请 DOI（第七节） |
| 正文与附录的两处近乎重复：正文和附录 C 都说 H_native 是固定的 Hermitian 算符，正文和附录 A 都给出 Kerr 形式 | 未改。附录 C 需要这一前提才能得出 H_native 属于非幺正生成元的 Hermitian 部分，附录 A 只用一句话引用正文公式，而正文的 “Retaining …” 是原稿原句 |

改后重新生成了标蓝版本（标蓝 616 句，未改 349 句），附录 J 补了整条电路的验证（见本节开头的“验证”）。两个版本都是 37 页，没有未定义引用，也没有 overfull box。
