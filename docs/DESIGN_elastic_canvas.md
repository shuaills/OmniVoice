# 弹性画布（Elastic Canvas）masked-diffusion TTS — 设计文档 v0.1

> 状态：草案。目标分支：`design/elastic-canvas-2026070x`。产出物：本文档 + RESULTS.md（按仓库家规，一实验一分支，可合并性以数值证据为准）。

## 1. 问题与动机

离散 masked-diffusion TTS（OmniVoice 形态）要求在生成开始前确定目标画布长度 T，且生成过程中 T 不可变：

- **偏短不可恢复**：文本被挤压（语速失真、吞字）；偏长靠尾部静音裁剪部分补救。
- **时长估计是规则**：OmniVoice `utils/duration.py` 用 Unicode script 字符权重 + prompt 语速折算（Eq.4），对数字、缩写、中英混排、异常语速 prompt 系统性失准。
- **表达通道被掐死**：时长本身是韵律载体（情感语速、戏剧停顿、拖长音）；副语言（笑声、叹息、换气）**在 transcript 中没有字符对应，因此不占画布预算**——instruct 请求了笑声，画布却是按平铺念字裁的。
- 邻居的自供：LLaDA-TTS（arXiv 2603.26364）局限第一条原文 *"requires specifying output length in advance (via a token-text ratio)"*。

**主张**：把时长从"生成前的全局参数"变成"生成中的局部决策"。将 DreamOn（arXiv 2602.01326）的 `[expand]`/`[delete]` 机制适配到 T×C 多码本声学矩阵，模型在去噪过程中自行伸缩画布。

## 2. Related work 定位（切割线）

| 工作 | 长度谁定 | 何时定 | token 体制 |
|---|---|---|---|
| OmniVoice (2604.00688) | 外部规则 | 生成前，冻结 | T×C 声学（Higgs 8cb） |
| LLaDA-TTS (2603.26364) | 外部规则（生成与编辑均是） | 生成前，冻结 | 单码本语义（CosyVoice） |
| DreamOn (2602.01326) | **模型**（expand/delete） | **生成中** | 1D 文本 token（代码 infilling） |
| **本工作** | **模型** | **生成中** | **T×C 声学** |

与 LLaDA-TTS 编辑功能的区别：它是**用户发起的离线画布手术**（外部编辑意图→对齐头定位→固定画布重生成）；本工作是**模型发起的在线长度自修正**（无外部意图，生成过程自己发现、自己改）。引用绕不开，定位切干净。

相对 DreamOn 的增量：① 编辑单位从 token 升到时间列（全码本联动）；② 语音特有的句中帧复制→delete 增强（局部语速修正，DreamOn 只在末尾删）；③ 与层级去噪（layer penalty）、CFG 的耦合设计；④ expand-k（>1→2）在语音上的必要性论证（插入十几帧笑声）。

## 3. 机制设计（底座无关）

### 3.1 编辑单位与词表

- 编辑单位 = **时间列**（一帧的全部 C 行）。
- `[expand]`/`[delete]` 作为 **codebook-0 词表的 +2 个类别**（head 与 embedding 各加两行，随机初始化——LLaDA-TTS 的新 [MASK] embedding 随机初始化先例可循）。
- 列状态机：`masked → {real-token | [expand] | [delete]}`。cb0 落定为真实 token 前，该列 1..C-1 行不参与预测（layer penalty 天然保证低层先解，此门控是顺水推舟）。
- 执行语义（照抄 DreamOn）：`[expand]` 当场裂变为两个全 mask 列；`[delete]` 当场移除该列。**deletion broadcasting**：delete 右侧全为 mask 时整段剪掉（DreamOn 实测步数 122→52）。

### 3.2 训练数据增强（collator 层，不动抽取管线）

对每个样本以概率 `p_elastic`（初始 0.5）施加画布腐蚀；其余样本保持原始精确画布（保底：推理禁用 special 类即回退 baseline 行为）：

1. **合并→expand**：随机把相邻真值帧列合并为一列，目标 cb0=`[expand]`，该列其余行不计 loss。合并概率用 DreamOn 的 static + dynamic-inverse 双调度器 1:1 混合。
2. **末尾附加→delete**：目标区末尾附加 0..K 个 `[delete]` 列（K~画布的 20-30%）。
3. **句中复制→delete（语音特有）**：随机复制一帧作为输入列、目标=`[delete]`。相邻帧近似连续，复制帧是声学合理扰动；给模型句中局部语速修正能力。
4. 画布总长扰动等效于 `T_canvas = T_true × U(0.7, 1.3)`，模拟规则估计误差分布（可用规则在训练集上的真实误差分布替代 U）。

### 3.3 Loss

标准 masked CE 基础上：`[delete]` 目标降权 1/N_delete（DreamOn 消融：去掉配平掉 6.2 个点）；与现有 per-codebook loss 权重（OmniVoice `normalized_audio_codebook_weights`）相乘合成。

### 3.4 推理循环修改

在 OmniVoice 32 步循环内：

- 每步 unmask 决策时，cb0 位置的候选含 `[expand]`/`[delete]`；被选中即**当场执行**结构操作并更新位置索引。
- **长度冻结阈值 θ**：仅当全局 mask 率 > θ（初始 0.4）允许结构操作；之后锁画布专心填细节。τ=0.1 的 time-shifted 调度前期只提交少量高置信位置——结构期与细节期天然分相。
- **CFG 旁路**：special 类 logits 不做 CFG 外推（无条件分支看不到文本，对长度的意见是噪声）；仅声学类别走 `c + s(c−u)`。
- `L_max` 护栏 = 规则估计 × 1.5，达到即禁用 expand。
- **Serving 兼容**（step 级批处理引擎）：按规则估计 × 1.3 预分配、mask 控有效长度，伸缩零 realloc；broadcasting 即尾部截断。位置编码每步按当前有效列重算（无 KV cache，本来就是全量 forward）。

## 4. 底座路径考量（本次新增的核心维度）

两条候选路径，机制设计对两者均适用；差异在集成方式与实验排序：

### Path A：自研从头预训练
- special 类从 day-0 进词表，弹性增强并入预训练数据管线 → 模型**原生弹性**，无 retrofit 分布错位。
- codec 可选自研（如 12.5Hz×16cb）：帧率减半降低画布长度与步数压力，但 16 列门控与 layer penalty 要重调；配方偏离 OmniVoice 越远，复现风险越大。
- 风险：违反"发版主线一次一个变量"——**缓解**：`p_elastic` 采样式混合训练保证 baseline 行为可完整回退（推理关掉 special 即纯 OmniVoice）；即便如此，弹性是否上主线仍以 E1/E2 数值门槛决定，不默认上。

### Path B：OmniVoice 公开 ckpt 继续预训练
- 词表手术：cb0 head/embedding +2 行随机初始化，其余全承接。模型已会固定画布去噪，弹性=行为微调。
- **已知风险**：底座只见过精确画布，有"画布即正确长度"的强先验；需要足量弹性增强数据冲淡。LLaDA-TTS 证据（AR→diffusion 转换 6000h 即反超）表明这类行为迁移收敛很快，但那是更大的分布切换，本处是更温和的增量，预期更容易。
- 锁定 Higgs 8cb codec → 微调数据需 Higgs 重抽。**只需子集**（千小时级足够行为微调），非全量 100kh。
- 附带收益：646 语言底座白拿；tech report 叙事为"OmniVoice + 我们的弹性画布 + 内部数据强化 zh/ja/ko"。

### （备考 Path C：自研 AR talker ckpt 转换，LLaDA-TTS 路线）
label shift + 双向化，从现有 AR ckpt 出发。多码本双流如何套 label shift 需单独设计，暂不入主矩阵，留为 E3 探索项。

### 决策解耦原则
**E1（在公开 OmniVoice ckpt 上做弹性微调）在两条路径下都是正确的第一个实验**：它同时是 Path B 的直接验证和 Path A 的机制预演（配方验证后平移进预训练管线），且不阻塞底座决策。底座选择的 gate 放在 E1 出结果之后。

## 4.5 与 block diffusion 的关系（为什么不直接上）

Block diffusion（块间 AR + 块内并行）通过重新串行化**溶解**定长问题（EOS 增量决定长度），并白送流式与 KV cache——是更完整的终局形态。本工作不选它作 v1 的理由：

1. **底座不兼容**：OmniVoice ckpt 是全双向，bidir→block 转换在语音上无先例、无公开 block 语音底座；这是换主线的体量，不是并行 arm。
2. **质量论点**：全局双向上下文是 OmniVoice SIM/鲁棒性主张的根；block 交还一半给串行（跨块误差累积回归）。离线质量优先的 v1 形态下，全双向+弹性可能是质量轴上更优的点。
3. **定位**：block 语音化是拥挤的收敛车道；弹性画布是无人占位的缝。
4. **资产平移**：collator 增强、special token 机制、时长鲁棒性评测在 block 世界复用；且**编辑场景**（插入所需 mask 数未知）是 block 因果结构解不掉、弹性画布正好解的长期用途。

**明示的赌注**：若 v2 全面转 block diffusion 流式化（建议下季度立项），生成侧长度自修正被溶解，弹性机制余值 = 编辑 + 句中语速修正 + 全部数据/评测资产。接受此代价。

## 5. 实验矩阵

| ID | 内容 | 底座 | 数据 | 周期 | Gate |
|---|---|---|---|---|---|
| E0 | 固定画布 baseline 复现（对照锚点） | OmniVoice ckpt 原样 | Emilia-zh 子集（Higgs 重抽） | ~2 天（含重抽） | CER/SIM 对齐论文 ±10% |
| E1 | **弹性微调**：§3 全量机制 | OmniVoice ckpt +2 类 | 同 E0 + 弹性增强 | 1-2 天/轮 | 见 §6 主图：0.7×/1.3× 初始画布下 WER 不塌 |
| E1-abl | no-expand / no-delete / 句中复制关 / θ 扫描 / p_elastic 扫描 | 同 E1 | 同 E1 | 各 ~1 天 | — |
| E2 | 弹性并入从头预训练（若 Path A 胜出） | 自研 | 全量 | 主线排期 | E1 复现 + 无主线质量回退 |
| E3（探索） | AR ckpt 转换（LLaDA-TTS 路线）可行性 spike | 自研 AR talker | 子集 | 3-5 天 | 转换后 CER 不劣于 AR 基线 2× 以内 |

## 6. 评测协议

1. **鲁棒性主图**：初始画布 ∈ {0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5}×oracle，WER/SIM/语速失真曲线，弹性 vs 固定。预期形状：弹性平、固定两头塌（DreamOn：90.8% vs 55.3% 的语音版）。
2. **时长精度**：|T_final − T_oracle| 分布 vs 规则估计；结构操作发生位置 vs 涌现对齐头（LLaDA-TTS L11 型探针，MAE 基准 1.29 token）的一致性。
3. **副语言表达**：instruct 驱动笑声/叹息/停顿测试集；画布增长事件计数 + 表达力 CMOS；"笑声不占预算"badcase 前后对比 demo。
4. **效率**：去噪步数开销（broadcasting 开关）、RTF 对比固定画布。
5. 常规：Seed-TTS-Eval / 内部 zh-ja-ko 口径，与 eval_any.sh 管线对齐。

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| expand 仅 1→2，长插入需多轮 | v1 接受；v2 引入 expand-k（DreamOn 作者自认的改进方向，语音更需要） |
| 画布震荡不收敛 | θ 冻结 + L_max + 每步结构操作数上限 |
| special 类置信度校准差（低频类） | loss 配平 + 监控 unmask 顺序日志 + CFG 旁路 |
| Path B 底座"精确画布"先验过强 | 提高 p_elastic / 延长微调；E1 的 gate 就是测这个 |
| LLaDA-TTS/DreamOn 团队先发 | 窗口按月计：E1 本周立项，两周内出主图 |
| 主线发版风险隔离 | 弹性只以并行 arm 存在；p_elastic 混合训练保证可整体关断 |

## 8. 里程碑（并行于发版主线）

- W1：collator 增强 + 词表手术 + 推理循环改造；4090 debug 冒烟；Emilia-zh 子集 Higgs 重抽。
- W2：E0 + E1 首轮；鲁棒性主图初版。
- W3：E1-abl 关键消融；副语言测试集构建与评测。
- W4：结果定稿 → tech report 章节（成）或 future work + 独立论文规划（不成也有 E0/E1 数据资产）。
