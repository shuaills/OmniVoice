# CFG + Band-4 training receipt（2026-07-16）

状态：**候选训练合同、代码与 launch manifest 已冻结，并于 2026-07-17 发车。** first-100 已否决 `guided` EOS 默认化，所有训练 A/B 固定使用 `legacy` EOS。未验证项仍按本文门禁执行，不因作业已启动而提前记为成绩。

冻结 manifest：

- source commit：`deb6cb16c4ce59344e79909e13df26377b0249b3`（施帅）；
- source root：`/opt/gpfs/users/shuai/work/cfg-band4-pretrain-20260717/OmniVoice`，启动时要求 clean tree，并校验 `omnivoice.__file__` 位于该目录；
- image：`registry.luna.ogpu.cloud/luna/ysyb-dev:latest`；
- init checkpoint：`/opt/gpfs/users/shuai/work/block-loss-design/OmniVoice/exp/blockcausal_splitloss_emilia_300k_lx20/checkpoint-300000`，`model.safetensors sha256=d72a01f60f01e2a6432779993981d2bdd2100266bdffc0a46a3cb957bf8e1908`；
- data config：`examples/config/data_config_emilia_full_blockparity.json`，冻结副本 sha256 `000ad8ebfb26e40511b87f503794768a3581800784192f56bfa94ac6832733c2`；
- candidate config：`examples/config/train_config_cfg9055_band4_10k.json`；
- self-terminating launcher：`cfg9055_band4_train.sh`，无 `sleep infinity`；
- H100 主作业：`shuai-cfg9055-band4-10k-h2-v2`，2×H100、GA=2、10k steps，输出 `/opt/gpfs/users/shuai/experiments/cfg-band4-9055-20260717/main10k_shuai-cfg9055-band4-10k-h2-v2`；
- 4090 工程作业：`shuai-cfg9055-band4-smoke-r2-v2`，2×RTX4090、300 steps，输出 `/opt/gpfs/users/shuai/experiments/cfg-band4-9055-20260717/smoke300_shuai-cfg9055-band4-smoke-r2-v2`。

发车后证据：4090 工程作业已于 `2026-07-17 01:59 CST` 完成 `300/300`，`rc=0`，最终窗口 `loss=3.8898`、`audio=4.4603`、`eos=0.0190`、`void=0.0054`，checkpoint 已写入 `checkpoint-300`。H100 主作业已进入稳定训练，step 50 窗口 `loss=4.0152`，所有 acoustic/EOS/void 统计有限；该数字只证明工程链路工作，不作为质量成绩。

## 目标

在保留 Band-4 晚停长尾收益的前提下，让训练真正覆盖推理时的两种 CFG 无条件分支：

- `U_shared`：无文本/style，但保留参考音频；
- `U_drop_ref`：无文本/style，也删除参考音频，target 时间线从 0 重启，并允许首块不足 32 帧。

首要验收不是“平均 WER 看起来不错”，而是消灭强 CFG 下的英文单点复读，同时保住可观测到的 SIM 空间。

## 已有证据

- 当前 300k block 全集相对 R1：zh `WER 1.07 / SIM .705` vs `0.89 / .768`；en `2.43 / .640` vs `1.65 / .716`。
- Band-4 first-300、同训练时长控制组：zh 时长 p95 `1.540→1.419`、`>2x 6→1`；en `2.358→1.905`、`24→13`。这是可信的终止长尾收益；WER/SIM 只按“未明显回归”表述，不宣传为已证明的产品增益。
- 交叉剂量探索里，`drop_ref g1` 相对 `shared g2` 的 SIM 差为 zh `+.032`、en `+.061`，WER 为 `.93 / 3.50`；它同时混入了 reference policy 与 guidance 变化，不能当成 drop-ref 的因果增益，正式 A/B 必须同剂量比较。
- 强引导 `drop_ref g2` 的英文 `57.93% WER` 几乎由一个样本造成：该样本 per-utt WER 为 `55.5`；其余 99 条均值约 `2.46%`。这说明 CFG 的整体质量空间存在；该失败与窄首块/U 合同失配假设一致，但因果仍需训练 A/B 验证。
- 已定位样本 `common_voice_en_17429166-common_voice_en_17429168` 的首个 U target block 只有 3 帧；`g0.5/g1` 正常，`g1.5/g2` 复读。
- 同 checkpoint、同 first-100 的 shared `guidance_scale=0` 已直接否决“现权重关掉 CFG 即可”：zh WER `.46→8.60`、SIM `.744→.680`、时长中位数 `1.041→1.641`；en WER `2.77→10.94`、SIM `.624→.586`、中位数 `1.262→1.892`。退化分布在大多数样本上，不是单点离群。
- 代码在 `guidance_scale=0` 时确实完全跳过 U cache/forward；任意非零 guidance（无论 `.25` 还是 `2`）仍支付相同双分支计算。因此小剂量只用于质量定标，不是延迟优化；若最终需要单分支速度，CFG teacher 蒸馏是另一份训练合同，不能冒充本 receipt 的普通微调。

## 当前合同为何不够

现有 `drop_cond_ratio=0.1` 会同时去掉文本、语言/style 和音频 prompt，并把 `prompt_ratio` 强制为 0。训练 U 因而总从音频帧 0 开始，按固定 32 帧分块。

推理 `drop_ref` 会物理删除参考音频，将 U 的 target RoPE 时间重置为 0。若参考长度 `S` 不整除 32，首块宽度为：

```text
q = 32 - (S mod 32), q ∈ [1, 32]
```

随后才恢复 32 帧块。当前训练没有覆盖 `q=1..31` 的 prompt-free 首块；cache/recompute 一致性只证明两条推理实现彼此一致，不证明它们与训练分布一致。

## 首轮训练配方

基线为 [train_config_ft10k_band4.json](examples/config/train_config_ft10k_band4.json)，从同一个 300k x20 checkpoint 初始化。除下述 CFG 样本合同外，LR、seed、数据顺序、Band-4、split loss 和训练步数全部锁死。

### 样本分支

| 分支 | 比例 | 文本/style | 参考音频 | target 几何 |
|---|---:|---|---|---|
| `C` | 90% | 保留 | 保留 | 现状不变 |
| `U_shared` | 5% | 删除 | 保留切点 `S` 前的 reference | 现状 shared-U 几何，不能把 target 强行移到新块 |
| `U_drop_ref` | 5% | 删除 | 物理删除 `audio[:S]` | 位置从 0 重启；首块 q 帧，之后每块 32 帧 |

无条件**样本比例**仍为 10%，但 suffix 长度改变后，监督 token 数、audio loss mass 和梯度剂量不会自动保持相同；训练必须逐分支记录 target 帧数、supervised cells 与有效 loss mass。新功能关闭时必须与现有 processor 的张量及 RNG bit-identical。

`U_shared` 的切点 `S` 复用条件样本现有的 prompt-cut 分布，并满足 `1≤S<T`；`audio[:S]` 无 target loss，target 从 `S` 开始。它必须逐列复现推理：`floor(S/32)*32` 个参考帧进入 committed clean history；`S mod 32` 个余数帧留在当前 noisy block 的前缀，target mask 紧接其后并与余数共享同一个 block。把所有参考帧都塞入 clean copy，或让 target 另起一个新块，都不等价。

### `U_drop_ref` 相位覆盖

- 在 `U_drop_ref` 内先用一次确定性 Bernoulli draw 选相位桶：50% 从 `q=1..32` 均匀采样，50% 从 `q=1..4` 均匀采样。因此 q=1..4 各占该分支 14.0625%，q=5..32 各占 1.5625%；
- draw 顺序固定为 branch → phase bucket → q → legal cut `S`；功能关闭时不消费这些新增 draw。给定 q 后，在所有满足该 q 的合法 S 中均匀选一个；
- 目标 suffix 至少保留一个 acoustic frame；先构造该样本所有合法 `(S,q)`。请求桶或 q 无合法 cut 时，在同一 `U_drop_ref` 分支内按上述固定混合分布、限制到合法 q 后重新归一并重采样，同时累计 requested/actual/rebucket 计数；不能跳过样本、改回 `C/U_shared` 或退回固定 32 帧；
- 首 q 帧的 `block_idx=0`，此后每 32 帧递增；clean/noisy 两副本共享 target-time position；
- 令 suffix 长度 `L=T-S`。ragged canvas 终点是最小的 `q+32k>L`；不能替换为普通的 `ceil(L/32)*32`；
- noisy block `b` 只能看 clean `<b` 与 noisy `b`，保持现有 block-causal attention 语义；
- `U_drop_ref` 中参考 token 泄漏数必须恒为 0。

### EOS / void / loss 保持项

- ragged canvas 终点必须是严格大于 target 长度的下一个块边界；target 恰好落在边界时仍追加纯 EOS block；
- `eos_band_k=4`；EOS 只在 cb0，原子 hard-mask；靠近 canvas 右边缘时允许明确裁成 1–3 格，并记录实际 band 宽度；
- void 从 Band-4 之后开始；
- `split_gamma=.8718`、`lambda_eos=.03728`、`lambda_void=.14516`；
- Band-4 仍按每文档一个 EOS event 求均值，不能把 EOS 剂量乘四；
- 训练 A/B 的主评测统一锁死为 `legacy`；`guided` 只保留显式研究开关，不能进入产品晋级矩阵。

## 实现门禁

1. 默认关闭：旧条件样本、labels、positions、tags、block ids 与 RNG bit-identical。新增 branch/q/S 使用按样本 keyed 的独立随机流，不能扰动后续全局 RNG；control/candidate 中匹配的 `C` 样本、mask 与张量必须逐位一致。
2. 90/5/5 分支频率与确定性 seed 测试。
3. `q=1..32` 全覆盖；`q=1..4` 过采样符合配置；requested/actual/rebucket 守恒，无 silent skip 或分支回退。
4. 对每个 `S mod 32`，比较推理实际存在列上的 positions/block ids，以及当前 noisy query 的可见性矩阵；训练独有的未来 clean/其他 noisy 列必须不可见。不能要求训练两副本与推理单时间线的完整 tags 张量相等。
5. ragged-block dense/flex 的布尔可见性 mask 精确一致；固定 fp32 tiny fixture 的数值输出按预先锁定的 `rtol=1e-4, atol=1e-5` 比较。
6. 每文档恰好一个 EOS event；Band-4 event 归一不变；audio/EOS/void loss 均有限。
7. 现有 cache/recompute、vocab、legacy fixed-canvas fail-fast 与 guided/legacy scorer 测试全部继续通过。
8. processor → collator/packer → model 的端到端 fixture 覆盖多文档 pack，并证明无跨文档 attention 泄漏。
9. 分布式聚合 branch/q/rebucket/target-frame/supervised-cell/loss-mass 计数守恒。
10. 冻结 repetition detector、WER/SIM scorer 版本与阈值，并用已知复读样本和正常样本做正反 fixture。

## 执行顺序与晋级线

### 300-step smoke

- 单独、自退出 OMS 作业，从 Band-4 相同初始化点启动；
- 核对三分支与 q 直方图、有限 loss、吞吐和显存；
- 固定复读样本只作诊断；300 步不承担“必须修好”或“不许恶化”的质量门槛；
- smoke WER 不作为晋级成绩。

### 10k A/B

- control 为现有 90/10、prompt-free、固定 32 帧 U；candidate 为完整 90/5/5 新 U 样本合同。二者使用同 checkpoint、数据顺序、seed、LR 和 Band-4；变化项不能简写成“只改几何”；
- 若要单独归因于窄首块覆盖，增加同为 90/5/5、但固定 `q=32` 的几何对照；
- first-100 通过后再跑 first-300；
- 先由同 checkpoint 的 shared-reference 强度扫描预注册主剂量 `s*`；主比较固定为 `shared s*` 对 `drop_ref s*`，`g1.5/g2` 只作压力测试。repo 的 `s` 公式是 `C+s(C-U)`，报告同时标注 conventional weight `w=s+1`；
- shared 回归限：WER `≤ +0.3pp`，SIM `≥ -0.005`，时长 p95 `≤ +0.1`；
- 训练增益：candidate 必须在同 seed、同 guidance 的 control-drop_ref 上改善已知 q=3 case，并降低 q=1–4 桶的 repetition rate；
- 产品门槛：candidate drop_ref first-300 不允许出现复读型灾难样本；g1.5/g2 英文 WER `<5%`，并保留相对 candidate shared 至少 `+.03` 的 paired SIM；
- 任一强 guidance 仍被单条复读污染，则不晋级。

first-100 只做灾难筛查，不承担细小数值收益的晋级判断；first-300 与 fullset 报告 paired bootstrap 置信区间。

### Fullset

- 固定 Seed-TTS zh 2020 / en 1088、bf16、16 steps/block、相同 seeds；
- repetition detector 与人工复核分别报告，英文复读为 0；runaway 作为独立指标报告，不能替代复读检测；
- candidate 相对同 seed control-drop_ref 必须继续改善 q=1–4 桶；产品 WER 不比同 checkpoint 的 candidate shared Band-4 差 `>0.5pp`，英文 paired SIM 至少 `+.03`；
- duration median/p95 不退化，并按 `stop_position mod 32` 分层检查早停；
- 通过后才讨论更长训练或新的 300k recipe。

## 方向性假设，不是成绩

- 高置信：任意非零 CFG 仍执行双分支，不能靠减小 guidance 获得单分支延迟；
- 中等置信：补齐 `q=1..4` 覆盖可能降低 `drop_ref` 强引导复读；
- 低置信：drop-ref 的 SIM 空间能否在不损 WER 的情况下保留。

不预报具体 WER/SIM/时长区间。Band-4 与 CFG 训练会共同改变 EOS/语音分布，收益不能线性相加；`p95<1.3` 只保留为长期目标。

## 与去 CFG 蒸馏的边界

本 receipt 修复的是**仍保留双分支 CFG**时的训练/推理几何。若产品目标是单分支速度，必须另立蒸馏 receipt：冻结 teacher scale 与解码器，定义单分支 student、teacher logit/trajectory 目标、自生成历史、EOS/Band-4 处理和单分支速度门禁。它不是普通 10k SFT；在 pilot 前不承诺“便宜三十倍”。

## 不确定性

“窄首块触发复读”目前由一个灾难样本、guidance 剂量响应和明确的训练/推理几何失配共同支持，置信度中等。90/5/5 与 q 过采样比例是首轮诊断配方，不是已证明最优值；失败时先调整 U 比例和 q 分布，不改写已独立验证的解码结论。
