# B2（真块因果 + 精确块 KV cache）实验记录

分支：design/block-b2-causal-20260706（基于 B1 分支 4270b6b）
工作区：/opt/gpfs/users/shuai/work/block-conversion-b2/
设计：docs/DESIGN_b2.md

## 2026-07-06 实现 + 全门数字

CPU（tests/test_block_dual_cpu.py）：
- gate1 规则真值表：50×50 packed 3 文档+pad，dense == flex mask_mod == 独立 spec 重实现，无空可见集行
- gate1b 处理器不变量：300/300（clean 拷贝逐位=真值、监督仅 noisy、EOS fill、共位 RoPE、tag/block 元数据）
- gate2 训练-推理可见集：4/4 块逐位相等；训练侧对推理缺席列（未来 clean、他块 noisy）全黑；cache 全可见假设成立
- gate3c tiny-fp64 cache 等价：cond/CFG token 全等，max|Δlogit| = 0.000e+00

GPU（tests/gpu_gate_b2.py，4090 sdpa fp32，真迁移 ckpt 0.8B）：
- gate3g cache 等价：cond max|Δlogit|=2.708e-04、cfg 1.373e-04，token 逐位相同，id 全法
- gate5 端到端训练前传：loss 8.452 有限，635 监督格；换全双向 mask loss 差 0.342 —— 块因果 mask 实际起作用（load-bearing 证明）

回归：B1 CPU smoke 与 elastic smoke 全过（共享接线未破坏既有路径）。

成本实测（400 样本，emilia 拟真长度分布，bs=32）：
- 序列长 dual/official = 1.911×；有效 batch 20480 tokens：82.2 → 43.0 条
- 每样本监督格：official 800 vs dual 817（1.02×，不缩水）
- 每 packed token 监督密度：3.212 → 1.717（≈折半 = 双拷贝的真实代价）

## staged（未提交训练）

train_config_block_b2.json（=B1 + block_scheme=dual，init 同一迁移 ckpt）、
b2_launcher.sh。提交模板：
saisuan_oms.sh submit-longrun --name shuai-block-b2 --queue queue-h100-4n \
  --image lunalabs-acr-registry.cn-guangzhou.cr.aliyuncs.com/luna/pytorch-ddp-example:latest \
  --gpus 8 --cpus 64 --memgb 512 --shmgb 64 \
  --launch-command "bash /opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice/b2_launcher.sh"

## 风险排序

1. flex create_block_mask 编译我们的 mask_mod 未实测（H100 训练第 0 步即验）
2. 无右侧上下文的质量代价未知（B1 vs B2 同起点 10k 探针定量）
3. 前缀自闭注意力 vs 官方前缀全可见的分布偏移（转换微调内容之一）

## 2026-07-06 B2 提交记录
- shuai-block-b2 Running @ysyb-gpu1（e11a 让位）。**flex 块因果 mask H100 编译
  一次通过**（最后未实测险退役），91 步 loss 4.25–4.55，1.11 it/s，零 Traceback。
- 10k 探针（~2.5h）判决项：① 逐块生成 CER 是否可听；② EOS 停长 vs 文本长度
  相关性；③ 同起点 vs B1@5k/9.3k 的 loss 曲线（cache 兼容几何的代价）；
  ④ cache 推理路径在真生成里的 RTF 实测。

## 2026-07-07 B2 判决探针：10k/20k vs 零点（block dLLM 成立）

口径：blockwise_probe.py，16 步/块，块 32 帧，max_blocks 12，cache 路径，4090 sdpa。
（fp16 cache/重算断言已降级为 match_rate 指标——fp32 门是正确性锚点，fp16 argmax
在近平手 logits 上翻面属浮点尘埃；本轮 parity/RTF 表未采样，30k 探针前修。）

| 指标 | 零点（init） | 10k | 20k |
|---|---|---|---|
| CER mean/max | 0.977 / 1.0（碎渣） | **0.006 / 0.045** | **0.006 / 0.045** |
| EOS 触发 | 0/10（全兜底） | **10/10** | **10/10** |
| 停长/规则时长比 | —（截断值 1.22） | 1.12 | 1.18 |
| 停长-文本 spearman | 0.0 | 0.058 | **0.391** |

**判读：**
1. **可懂度 10k 步即回到官方水位**（CER 0.006 = elastic 臂 fixed 路径同一水平，
   max 0.045 也是同一条 t4 的 ASR 侧怪癖）。块因果几何 + 单向生成的质量代价
   在这个测试集上≈0。
2. **EOS 全触发、零兜底**——模型自决长度成立，停长比规则估计长 12–18%
   （规则本来就只是估计，不作为真值）。
3. **停长-内容相关性在长出来**（0.058→0.391，10 样本 spearman 粗但趋势明确）
   ——"说多长"开始跟内容走而不是跟惯性走。
4. 诚实条款：10 条中文小集；英文 WER、SIM、MOS 未测；cache RTF 收益待
   长序列/H100 实测（4090 短序列反慢 10%）。50k 后按 §全量评测补。

## 2026-07-07 cache 质量对照裁决（fp16 分叉定性）

背景：greedy 下 cache/重算 token match 仅 2–30% 且 20k 低于 10k，不能当浮点尘埃
放过。裁决法：token 不可比（迭代生成蝴蝶效应），改比质量。
**结果（ckpt-20000，t0/t5）：cache CER 0.0 = recompute CER 0.0。**
判：fp16 首个近平手 argmax 翻面后的合法采样分岔，非实现缺陷（fp32 门本就
整段逐位相同）。cache 路径放行。probe 新增 parity_cer 栏目常态化此对照。
附：match_rate 指标保留仅作监控异动用，不再解读绝对值。

## 2026-07-07 三列对照（原版 vs B2-30k 双模式）+ 30k 判决

| | ①原版ckpt·定长 | ②B2-30k·定长 | ③B2-30k·blockwise |
|---|---|---|---|
| CER mean/max | 0.006/0.045 | **0.442/0.625** | 0.013/0.062 |
| 时长 | 4.37s（规则4.29s） | 3.41s（塌） | 4.77s |
| RTF（4090 sdpa） | 0.265 | 0.396 | 0.992 |
| EOS/spearman | — | — | 10/10；ratio 1.121；**spearman 0.161** |

**判读：**
1. **"双模式通吃"假设被证伪**：B2 权重在原版定长模式下 CER 0.442——纯截断画布
   训练**置换**了全画布去噪能力（训练分布里从不存在"当前块在中间、后面还有
   掩码块"的状态）。产品上无碍（blockwise 即目标形态），报告措辞从"增量能力"
   改为"能力转换"；若要双模式，B3 配方可掺 ~10% 全画布官方样本（待议）。
2. **spearman 曲线打脸警告**：0.058→0.391→0.161 非单调。n=10 的秩相关本来就
   极噪（两个名次互换即大幅摆动），此前"0.39 在爬"的读数不可作为报告 claim。
   **行动：扩建 100 条变长文本的长度语义评测集**，40k/50k 用大集出曲线。
   （又一次低 n 读数教训——与 E1 的 10k 克制假象同性质。）
3. CER 0.013 vs 前两点 0.006：多一条 ASR 滑落，量级仍 fixed 水位，40k 复核。
4. RTF：blockwise 0.992 vs 官方 0.265（4090/sdpa/单流），当前配置下慢 ~4×。
   流式的正确卖点是**首块延迟**（~0.5s 出首个 1.28s 音频 vs 官方全画布 4s+ 后
   才出声），下轮探针加 time-to-first-block 指标；吞吐收益等 H100 flex + 批量。
数据：probe_compare_30k/、probe_b2_30000/。

## 2026-07-07 音色锚定（voice prompt）接入 blockwise
- 锚：Seed-TTS zh 官方参照 10002287-00000095（4.74s，probe_assets/ref_anchor.*，
  token 预编码 8×119）。YODAS 试点片段太脏（ASR 转写不知所云），弃用。
- 病根修复链：audio_mask 语义=音频区域（非待生成）→ 探针 a0 切分丢参照 token
  → decoder 新增 seed_audio（整块预填 + 部分块预填 + 双 cache 预热，
  返回自动剥 seed）。全程 seed=None 时原路径字节不变。
- 验收：30k ckpt 锚定冒烟 CER 0.0/0.0，无参照泄漏；prompt 跟随忠实
  （尾裁实验里模型自动补念被裁文字——负例反证）。
- 后续探针统一带锚：听感可比 + 解锁 SIM 指标；首块延迟(prefix+block0 wall)
  已在 per_block_wall_s 里可直接报。

## 2026-07-07 40k 大集判决（55 条 × 音色锚，首次真统计口径）

**头条：spearman(生成帧数~字符数) = 0.81（55 点）**——10 点小集的 0.06/0.39/0.16
锯齿确系噪声，长度语义实为强相关。停长/规则比 1.05。首块延迟 1.37s
（prefix+seed 预热+首生成块，4090）vs 官方全画布 ~4.4s 才出声 ≈ **3.2× 提速**。
cache 质量对照：三组 cache==recompute CER 逐位相同（含坏例——坏得都一样，
恰证两路径等价）。

**长度分桶（zh 40 条）：**
| 桶 | CER | EOS |
|---|---|---|
| 短 ≤15 字 | 0.742 ❌ | 10/10 |
| 中 16-40 字 | **0.000** | 10/10 |
| 长 41-70 字 | **0.002** | 9/10 |
| 超长 71+ 字 | 0.375 ❌ | 0/10（全兜底） |
| en 15 条 | — | 15/15 |

**两个失败模式的病根（都是训练分布边界）：**
1. **超长崩 = max_sample_tokens 2000**（≈10s 音频/样本）：模型没见过 >10s 的
   画布，71+ 字文本需要 ~20s+ → EOS 行为未定义 + 撞 max_blocks 截断。
   解法：serving 侧按句切分（标准做法）；或下一轮训练放开长样本。
2. **短文本+长锚崩 = prompt_ratio_range [0,0.3]**：训练中 prompt 最多占画布
   30%；锚 119 帧 + 目标 ~30 帧 → prompt 占 80%，分布外 → 模型续着锚的
   势头瞎说（greedy 下更糟，parity 短文本 CER 3-6.5）。解法：短目标时锚
   裁短（如 ≤2× 目标估计帧数），或训练放宽 prompt 比例。
3. **甜点区 16-70 字（≈1-10s 目标）无可挑剔**：CER 0.000/0.002，EOS 95%。
   50k 全量评测（Seed-TTS 句长多在此区）不受影响。

RTF（EOS 行）3.4（4090+锚+16 步口径，工程优化空间照旧记 H100/flex/批量）。
数据：probe_b2_40k_full/。

## 2026-07-07 Seed-TTS 全量评测生成管线（staged，未提交 job）

- 生成器：tests/seedtts_blockwise_gen.py——可分片（--shard i/n）、断点续跑
  （跳过已存在 wav）、逐条 meta/failures jsonl、输出命名 {utt_id}.wav 与打分
  栈对齐。锚定路径照抄 blockwise_probe（prompt wav → encode → seed_audio）。
- 验证（zh 前 5 条，ckpt-40000，4090）：5/5 生成、全 EOS 停止、时长 4-10s
  目标量级无参照泄漏；ASR 抽查 10002290-00000011 逐字正确
  （自动驾驶将大幅提升出行安全效率）。
- **打分栈用法（yinfeng 的 tts_eval，自带 venv 与评测模型）**：
  WAV_DIR=/path/to/wavdir TESTSET=seedtts_zh bash /opt/gpfs/users/yinfeng/work/tts_eval/eval.sh
  要求布局 WAV_DIR/seedtts_zh/<utt_id>.wav；en 同理 TESTSET=seedtts_en；
  TESTSET=seedtts 出 zh+en 合并报告。指标：SIM-o（WavLM）+ WER avg/wgt
  （zh paraformer / en whisper-large-v3）。测试清单 id 与 test.tsv 逐一对齐
  （zh 2020 / en 1088）。官方基线对照：exp_results.md（zh 0.778/0.89，
  en 0.741/1.65）。
- launcher：b2_eval_launcher.sh（8 分片×8 GPU，env: TSV/CKPT/OUT）。
  **提交模板（50k 收官后由主会话发）**：
  zh: OUT=results_b2_50k/seedtts_zh（默认即 zh）
  en: TSV=.../seedtts_testset/en/test.tsv OUT=results_b2_50k/seedtts_en
  oms: saisuan_oms.sh submit-longrun --name shuai-b2-eval-{zh,en}
  --queue queue-h100-4n --image <广州ACR> --gpus 8 --cpus 64 --memgb 512
  --shmgb 64 --launch-command "TSV=... OUT=... bash .../b2_eval_launcher.sh"
  预估：zh 2020 条/8 卡，H100 单条 ~3-5s → ~20-30 min；en 同量级。

## 2026-07-07 50k 收官探针（55 条 × 锚，B2 训练终点）

| 指标 | 40k | **50k（终点）** |
|---|---|---|
| spearman | 0.81 | **0.791（平台）** |
| 停长/规则比 | 1.05 | **1.006（几乎完美校准）** |
| 中桶 16-40 字 CER | 0.000 | **0.000** |
| 长桶 41-70 字 CER/EOS | 0.002 / 9/10 | **0.002 / 10/10** |
| 短+锚 CER | 0.742 | 0.75（未自愈） |
| 超长 EOS | 0/10 | 0/10（未自愈） |

**结论：** ① 甜点区终态完美，长度语义平台在 ~0.8，停长校准 1.006；
② 两个分布外红格再训 1 万步纹丝不动——**"病根在训练数据分布、不在步数"
诊断坐实**（短+锚 = prompt_ratio≤0.3 上限；超长 = max_sample_tokens≈10s 上限），
修复归训练配方（B3/下轮），不归推理。③ parity 例行满分（含坏例逐位同）。
B2 训练线正式收官，主表交给 Seed-TTS 全量评测（生成中）。

## 2026-07-07 Seed-TTS zh 首轮打分 + 截断验尸（主表第一战）

首轮：SIM 0.536 / WER 30.56%（官方基线 0.778/0.89）——但错误构成 12017 del
vs 73 ins = 没说完，不是说错。验尸链：
1. 帧/字双峰：70% 健康（p50 6.1，与探针甜点一致），30.6% 截断（p5 0.06）；
2. prompt 时长假说阵亡（截断组 4.76s vs 健康组 4.67s，无差）；
3. **定论：618 条截断中 616 条停在首个生成块内（中位第 5 帧）**——
   部分块预填状态（prompt 尾列+掩码列混居）中 [eos] 误击发，逐条 ~30%。
   与 162 条"empty generation"同族，只是击发位置略后。
修复：min_gen_frames=0.3×规则估计（commit 8780f2f），观测截断全部 ≤14 帧，
地板 37+ 帧 → 全覆盖。**en 场次自带地板（时序上的天然 A/B）**：en meta 截断
归零 → 修复即证 → zh 全量重生成（results_b2_50k_v2）+ 重打分。
健康 70% 的存在意味着修复后 WER 应回落到个位数量级；SIM 0.536 里截断的
贡献占多少，重打分后再判是否另有克隆保真度问题。

## 2026-07-07 Seed-TTS 主表终版（B2-50k，blockwise + 逐条 prompt 克隆 + EOS 地板）

| | SIM-o | WER% | 基线 SIM/WER | 样本 |
|---|---|---|---|---|
| zh（v2，地板修复后） | **0.693** | **4.16** | 0.778 / 0.89 | 2020 |
| en | **0.641** | **3.68** | 0.741 / 1.65 | 1088 |
| zh 首轮（无地板，作废留档） | 0.536 | 30.56 | — | 2020 |

地板一刀：zh WER 30.56→4.16、SIM +0.157（截断同时吃 WER 和 SIM 的实证）。
剩余缺口跨语言一致：SIM −0.085~−0.10，WER 2.2~4.7×。定性框架：官方 =
581kh 基座原生形态 vs 本表 = ~100 H100·时流式转换形态，缺口即"流式质量
成本"的 v1 报价，推理侧优化（每块步数、CFG scale）尚未动刀。
下一战：① SIM 缺口主攻（块间音色漂移/步数/fp16 三嫌疑）；② WER 便宜刀
（steps 16→32、CFG 扫描）；③ 数据 results_b2_50k_v2/（zh）、results_b2_50k/（en）。

## 2026-07-07 优化战开场：SIM 缺口诊断（免费分析）+ 便宜刀扫描提交

**SIM 缺口两个免费诊断（现有 v2/en 逐条日志复算，解析器与报表精确对齐 0.693/4.16）：**

1. **块间音色漂移排除**：SIM 随生成块数不降反升——zh spearman(SIM~n_blocks)=+0.10，
   en=+0.30（en 6块0.584 → 11块0.688 单调爬）。若跨块漂移存在长句应更差，实际相反
   （短句低 SIM 更像说话人嵌入在短音频上的固有低估）。漂移假设✗。
2. **缺口是弥漫性的，不是坏尾拖的**：zh WER=0 的 1022 条平均 SIM 仅 0.708（基线
   0.778，差 0.07）；WER>0 组 0.678。spearman(SIM~句WER)=−0.163（弱耦合）。
   最低-100-SIM 尾部确实高 WER（句均 0.218 vs 全体 0.041）但量太小补不了缺口。
   → 剩余嫌疑：每块采样质量（steps/CFG/fp16）或转换训练上限。

**便宜刀扫描（双作业已提交，commit 96b022f）：**
- `shuai-b2-sweep-zh`：zh 前 304 条子集（LIMIT=38×8 shard，seed 与 v2 全跑逐条一致），
  5 配置顺跑+each 自动 eval.sh 打分：steps32 / gs1.0 / gs3.0 / gs0(无CFG) / fp32。
  **锚点（steps16 gs2.0 fp16，从 v2 现有数据复算）：SIM 0.7134 / WER 3.45%（n=304）**。
  产出 results_b2_sweep/<tag>/seedtts_zh.report.md。
- `shuai-b2-s32-en`：en 全集 1088 条 steps32（主表级对照行，直接量"WER 缺口里步数值多少钱"）。
  产出 results_b2_steps32/seedtts_en.report.md。
- 代码新旋钮：seedtts_blockwise_gen.py --guidance-scale/--dtype；launcher 环境变量
  STEPS/GS/DTYPE/LIMIT 透传。

## 2026-07-07 便宜刀扫描·波 1 收官（zh 前 304 条子集，锚=steps16/gs2.0/fp16）

| 配置 | SIM | WER% | 判决 |
|---|---|---|---|
| 锚点 | 0.7134 | 3.45 | （v2 现有数据复算，同种子） |
| steps32 | 0.713 | 3.79 | 空刀：块内去掩码 16 步已饱和，算力翻倍零增益 |
| gs1.0 | 0.695 | 7.50 | CFG 减弱两头输 |
| **gs3.0** | 0.710 | **2.89** | **免费午餐：WER −16%，SIM 噪声级（−0.003）** |
| gs0 | 0.594 | 25.80 | CFG 完全承重；"关 CFG 省算力"阵亡 |
| fp32 | 0.705 | 3.90 | 平刀：fp16 无罪 |

**SIM 缺口三案并审结案**：漂移✗（SIM 随长度上升）、坏尾✗（WER=0 句同样低）、
推理侧✗（steps/CFG/精度全平）→ **归因=转换训练上限**（0.07~0.10）。
推理侧对 SIM 无剩余子弹；候选训练侧杠杆：更长转换（>0.5 epoch）、prompt_ratio
配方、数据质量子集、向双向老师蒸馏。均需真卡，待拍板。

**WER 侧**：CFG 单调陡降后趋缓（0→25.8，1→7.5，2→3.45，3→2.89），最优点
在 3~4 之间，波 2（2.5/3.5/4.0）钉死后主表配方免费升级。

## 2026-07-07 en 全集 steps32 复核：steps 刀从"空刀"升级为"反刀"

| en 全集 1088 | SIM | WER% |
|---|---|---|
| 主表 steps16 gs2.0 | 0.641 | 3.68 |
| steps32 gs2.0 | 0.640 | **6.03（+64%）** |

- SIM 与步数无关（第三次确认）。WER 显著恶化且与 zh 子集方向一致（+10%→+64%）。
- 失败模式排查：中位帧 136 vs 137、EOS 100% vs 100%、截断率 1.5% vs 1.3% →
  **非长度/EOS 问题，是正常生成内的发音退化**。机理：32 步时每步仅解 1 帧，
  置信重掩码轮次翻倍，早期正确承诺被反复改写。
- serving 叙事升级：16 步不是妥协是甜点——更少步数又快又好。

## 2026-07-07 CFG 波 2 收官：最优点判决 gs3.0

| gs | 0 | 1.0 | 2.0锚 | 2.5 | **3.0** | 3.5 | 4.0 |
|---|---|---|---|---|---|---|---|
| SIM | 0.594 | 0.695 | 0.7134 | 0.704 | **0.710** | 0.693 | 0.683 |
| WER% | 25.80 | 7.50 | 3.45 | 3.22 | **2.89** | 3.55 | 3.07 |

- WER 在 3.0 后进入噪声平台（±0.3pp 抖动），SIM 在 3.0 后开始单调付费
  （0.710→0.693→0.683，过度 guidance 压音色）。**gs3.0 = 甜点**。
- 终版主表已发射：shuai-b2-final-gs3（zh 2020 + en 1088 全集 @ gs3.0，合并打分）。

## 2026-07-07 大鱼：句首语气词 = 转换数据教坏（官方 0.6% vs B2 35.6%）

用户听感发现（"所有样本开头都有啊啊啊呃呃呃"）→ 量化钉死：
- zh 句首语气词率（ASR 下限）：**官方 0.6%（12/2020）vs B2 gs2.0 35.6%（718/2016）**。
- 插入错误对账：35.6%×1字 ≈ 1.6pp ≈ 实测插入差（1.83−0.02）。**句首口癖 ≈ 半数 WER 缺口**。
- 推理旋钮压不掉：CFG 2→4 只降到 ~23% 地板；steps/fp32 无关（21-27% 区间）。
- 归因：Emilia-YODAS YouTube 切段天然以"啊/呃/气口"开头，0.5 epoch 转换教会口癖；
  官方 58 万小时配方无此分布。
- **修复路径（训练侧，待拍板）**：转换数据过滤（YODAS 自带转写，文本级过滤
  句首语气词条目近乎免费）+ 重转换 ~100 H100·h。预期 zh WER 2.89→~1.7
  （逼近官方 0.95），且数据质量过滤可能同时抬 SIM（同一杠杆双击两个残余缺口）。

### 补充：插入错误构成拆解（口癖的双语镜像）
- zh 插入 715 条：**96% 句首 filler**（634 首 + 51 首尾）——开口"啊/呃"。
- en 插入 134 条：**88% 句尾 filler**（100 尾 + 18 首尾）——停止前拖"uh"（EOS 前垫口癖，
  数据习惯直接塑造停止行为）。
- 同一病根（YODAS 口语切段）双语两种表型；文本级过滤应同时滤"句首+句尾 filler"条目。

## 2026-07-07 终版全集 gs3.0 落地：子集甜头缩水，主表维持 gs2.0

| 全集 | gs2.0（主表） | gs3.0 | 官方 |
|---|---|---|---|
| zh | 0.693 / 4.16 | 0.680 / 3.67 | 0.778 / 0.89 |
| en | 0.641 / 3.68 | 0.631 / 3.59 | 0.741 / 1.65 |

- zh：WER −12% 但 SIM −0.013；en：WER −0.09pp（白拧）SIM −0.010。
- 304 子集读数（−16%/SIM−0.003）到全集缩水——**战役第三次小样本过甜教训**
  （前两次：E1 10k 克制读数、10 条 spearman 抖动）。子集定方向，全集定去留。
- **决策：主表维持 gs2.0**；gs3.0 全集行入消融表。CFG 曲线的分析价值保留
  （承重证明 + 3.0 后 SIM 付费）。WER 主杠杆 = 口癖数据修复（重转换，待拍板）。
- 今日推理侧优化战收官：steps 反刀、CFG 微利、fp32 平刀、SIM 归因训练侧、
  口癖=半数 WER 缺口。数据侧重转换是下一仗。

## 2026-07-07 B2F 开战：口癖过滤重转换（用户拍板"直接跑"）

- **设计**：与 B2 严格单变量对照——同 init（OmniVoice-block 迁移 ckpt）、同 50k 步、
  同配方，唯一差异 = 数据运行时过滤 `filter_edge_fillers`（zh/ja/ko 句首、en 句首+句尾
  filler 条目整条丢弃）。丢弃率抽样：zh 4.7% / en 6.1% / ja 6.4% / ko 4.7%（时长同量级）。
- **7 倍放大效应**（报告素材）：数据仅 ~5% 句首口癖 → 生成 35.6%。块因果首块无未来
  上下文、文本证据最弱，声学先验过度表达。也因此文本过滤的实效必须探针实测
  （转写不到的气口是盲区）。
- 代码：branch design/block-b2f-fillerfix-20260707，commit ad82307
  （edge_filler_filter.py + builder/config 接线 + b2f_launcher.sh；单测 13 例过）。
- 作业 shuai-block-b2f 已提交（8×H100 50k 步 ~13h）。**10k 探针判决项：探针生成句首
  口癖率 vs B2@10k 基线**；次判决：CER 不回退、EOS 校准不回退。

### B2F 判决协议 + 口癖的风格随迁新发现
- **新发现**：锚定探针（干净播音腔锚）口癖仅 7.7%（4/52 @50k）vs 测试集逐条真实
  prompt 续写 35.6%——**口癖强度是 prompt 风格条件化的**（YODAS 风随意语音诱发，
  干净锚拉不出）。报告素材：数据口癖不仅被首块先验放大，还被 prompt 风格门控。
- **B2F@10k 判决工具**（锚定探针基率太低不够用）：① 锚定探针 55 条（CER/EOS 回归
  检查）；② zh 测试集前 100 条续写（4090，--limit 100）→ ASR 口癖率。
  **B2@50k 前 100 条基线 = 18.2%（18/99）**（低于全集 35.6%——前 100 条偏"好说话人"段，
  再证风格门控）。通过线（n=99 二项显著）：B2F 口癖率 ≤8% 且 CER 不回退。

## 2026-07-07 战略升级：盘家底发现官方数据可用 → B2G 顶上主攻（用户问"数据不行为啥不换"）

**YODAS 判决书（准确表述）**：不是全不行——block-AR 重学、EOS 校准都是它教会的。
是带三个特定病：① 口癖（句首/句尾 filler → 半数 WER 缺口）；② YouTube 音质
（SIM 天花板头号嫌疑）；③ 段子短（均 9.2s → >10s 红格的真因更可能是数据长度分布，
而非此前猜的 max_sample_tokens：官方同用 2000 但数据长）。

**家底盘点**：`data/20260622` = speech_master_v1_616 生产数据湖（zh 40.2 万 h 全量
+ en 57.5 万 h 部分，97.7 万 h 落盘）。其中 **higgs_tokens_436k = zh/en 各 21.8 万 h
已 token 化**（25 tok/s 同帧率、同 tar+jsonl 格式、data_config_internal.json 正是
官方 500k 训练所用 = exp_205000/250000 基线出处）。**句首口癖率 0.9%**（YODAS 1/5），
长书朗读式干净内容，平均单条 ~45s（顺带攻 >10s 红格）。

**新棋局**：
- **B2G（主攻）**：同 init 迁移 ckpt + 官方 436kh 数据 + 同 50k 步转换（launcher
  b2g_launcher.sh，data_config_internal_b2g.json，filter_edge_fillers 也开着
  兜 0.9% 残渣）。一次攻三病：口癖（干净数据）、SIM（生产音质+与基座零域差）、
  长外推（45s 均长）。commit 23a4cc9。
- **B2F（降级为消融）**：重新排队在 B2G 后。价值=报告消融行"只靠过滤能治多少"
  （只有 YODAS 级数据的用户的廉价路径）。
- 提交顺序 B2G 先（先占节点）。判决协议沿用：10k 探针 + 前 100 条续写口癖率
  （基线 18.2%）+ CER/EOS 回归检查；B2G 另加长文本桶复查（>10s 红格是否自愈）。

## 2026-07-07 B2S 入列：官方数据从头预训练 block dLLM（用户拍板，不用 OmniVoice ckpt）

**匹配数据三角**（报告核心实验设计，全部同一批官方 436kh tokens）：
| 臂 | init | 步数/超参 | 回答的问题 |
|---|---|---|---|
| 官方双向（已有） | Qwen3-0.6B | 500k, lr 2.1e-4 | 质量上界 |
| B2G 转换 | 官方迁移 ckpt | 50k, lr 3e-5 | 流式的质量成本（vs 官方） |
| **B2S 从头** | **Qwen3-0.6B（无声学 init）** | 500k 档位, lr 2.1e-4, cosine, warmup 3% | **转换到底省多少**（vs B2G 同步数对比） |

- B2S 配置 = B2G 数据/架构 + 官方预训练超参（train_config_block_b2s.json，
  commit 4e1b717）；save 每 10k、留 20 个 ckpt 供轨迹对比。
- 速度账：B2 实测 1.11 it/s → 10k≈2.5h、50k≈12.5h、500k≈5.2 天（多日占卡）。
- 判决点：10k/25k/50k 探针 CER/EOS 轨迹 vs B2（转换臂 YODAS 版）同步数轨迹；
  若 50k 时 B2S 仍不可懂而 B2G 已成品 → "转换=千分之一成本"论点闭环。
- 队列顺序：B2G → B2S → B2F（B2F 再度让位，删除重排）。
- 风险：双拷贝块因果从零训练无先例（B2 的 flex mask 一次过是在有 init 情形）；
  前 1000 步 loss 曲线定生死，监视器盯 Traceback/loss。

### 2026-07-07 晚间上卡记录
- **B2G Running @ysyb-gpu3,7 卡**（1 卡小作业堵死整机位 → 7 卡挤位开跑；全局批
  17920 vs 20480 tokens 的 7/8 偏差,lr 不变——脚注级影响,记录在案）。
- **B2S Running,8 卡整机**（他组释放一台,顺位吃进）。两主臂并行,15 卡在训。
- B2F(消融)继续排队。

## 2026-07-07 口癖归因修正:发现机械成分(用户耳测直觉触发)

用户听感:"每条都在固定第一秒、同一个 emm 声"→ 更像伪影而非学来的口癖。切割实验:

**实验① R 相关性(免费,n=2016)**:口癖率强烈依赖首块种子占用 R=prompt帧 mod 32:
R∈[0,16) ≈ 40%,R∈[24,32) = 20.1%,R≥28 = 14.1%。**数据习惯不可能感知块网格
→ 机械成分实锤**。嫌疑机制:训练时块内可见位随机散布,推理首块却是"连续种子
前缀+连续掩码后缀"——分布外可见模式,掩码区起点被种下迟疑音。
- cache 无罪(fp32 逐位 parity + 50k parity_cer 逐条相等,cache≡重算);
  凶手在两路径共享的接缝预填逻辑/模式失配。
- R≥28 仍 14% vs 官方 0.6% → 双层结构:接缝伪影为主 + 数据习惯残余。
- 之前"官方 0.6% vs B2 35.6%"的对比混杂了数据与推理路径两个变量(官方走
  全画布推理,无接缝)——归因需修正为两成分。B2G/B2F(换数据/过滤)只治数据层;
  接缝层需推理侧或训练侧修复:
  ① 训练侧(下一轮配方):块内掩码模式增广——按概率用"连续后缀掩码"替代随机散布,
    教会接缝模式;② 推理侧带创可贴:首块掩码区前几帧的去噪偏置/后处理裁剪。

**实验②(耳测,进行中)**:8 对跨说话人"生成首块 1.3s vs prompt 尾 1s"已发用户:
若 emm 跨说话人同声=固定伪影;若在各自 prompt 音色里=数据习惯。

### 口癖伪影裁决链(续):声码器排除,锁定模型侧 token 级坍缩
- **耳测判决(用户)**:8 说话人开头"emm"= 同一个声音,与 prompt 音色无关,仅音高
  微调 → 固定伪影指纹。
- **A/B 解码实验(12 条,tests/junction_decode_ab.py)**:裸解码 vs 带种子上下文
  解码后切片——**逐字同分**(口癖 5/12 vs 5/12,转写一致)→ 声码器冷启动排除,
  "emm"写在生成 token 里。
- 现头号假设:接缝分布外可见模式(连续前缀+连续掩码)下模型的 **token 级模式
  坍缩**——cb0 输出固定"emm"音姿 token 串(与说话人无关),cb1-7 随 prompt 微调
  (解释音高微差)。验证中:tests/junction_token_sig.py(跨 8 prompt 对比首 16 帧
  cb0 token ID)。若坐实 → 拿到 token 签名,推理侧可做外科手术式创可贴
  (首块掩码区起点 ban 签名 token,与 EOS 地板同族);根治仍是训练侧掩码模式增广。
- 注:B2G(换数据)治不了这一层——B2G 评测时若伪影仍在,插入错误将继续污染 WER,
  归因时须分层读数。

### 签名揭晓:cb0=0 静音串 —— 全案重构为"话头静音模板 + 裸解码不裁剪"
- 8/8 条生成的首 10-16 帧 cb0 全为 token 0(静音/底噪),跨说话人/文本/种子恒定;
  cb1-7 带 prompt 残留 → 渲染成低哼"emm",音高随说话人微变,ASR 随机转写成
  啊/呃/嗯。**伪影覆盖 ≈100%,ASR 只捕捉到 35.6%**(用户耳测"全部都有"是对的)。
- ~~"数据话头静音模板"~~ **已证伪**:训练样本话头 cb0=0 连跑中位=0(YODAS 与内部
  数据同,Emilia 管线切得紧)。**修正机制:分布外不确定性 → 边际先验坍缩**——
  接缝可见模式(连续前缀+连续掩码)训练未见 → 后验平坦 → 置信度排序去掩码让
  边际最高频 token(句中停顿的静音 0)最先胜出锁定 → 恒定静音串+半静音哼声。
  预言检验:CFG↑ 锐化条件分布应压制坍缩 → 与实测 CFG-口癖单调负相关吻合✓。
  由于数据话头无 0 串,**裁剪首部 0 串不可能误伤正文**(合法语音不以 0 串开头)。
- 创可贴(零风险 token 级):裁掉生成段首/尾连续 cb0=0 帧再声码
  (tests/junction_trim_test.py 验证中)。en 句尾"uh"疑同源(EOS 前尾静音)。
- 注:R 相关性/耳测/A-B 声码器排除三步裁决链全部兼容此解释。

### 根因调查(用户军规:模型要发布,禁 workaround,只修/只查最本质原因)
裁剪实验降级为病理确认(TRIM 后口癖 5/12→0/12、CER 0.040→0.015、正文无损
——证明伪影=纯首部静音串),**不进产品管线**。

排除清单(全部实证):
- 数据话头静音模板 ✗(训练样本话头 0 串中位=0)
- 可见性几何 OOD ✗(处理器读码确认:prompt_ratio 机制天天训练"块内连续可见前缀
  +随机掩码后缀",m_lo=max(lo,prompt_length))
- prompt 尾部静音传染 ✗(12 条 prompt 编码尾部 0 串全为 0 帧)
- 声码器冷启动 ✗(A/B 逐字同分)、cache ✗(fp32 逐位 parity)

存活假设(终审中,tests/pause_verdict.py):
- H-语言学:完整句 prompt→新句 target 之间,真人本来就停顿;模型行为正确。
  官方管线 preprocess_prompt(去静音+补标点)+postprocess_output(裁静音)本来
  就是产品的一部分,官方 0.6% 是裁剪后数字;我们 blockwise 三步全旁路 → 差距
  =管线不对等,非模型病理。跨种子恒同 = class_temperature=0 的 argmax 确定性。
- 终审 T1:官方 ckpt 官方模式关 postprocess,量裁剪前头部静音;
  终审 T2:句中续写(prompt=前 60% 音频+全句文本)——训练分布内边界,
  停顿归零 → 语言学解释成立。

## 2026-07-07 深夜:根因裁决闭合 + 根修上线(用户军规:只修最本质,模型要发布)

**终审证据(tests/eos_displacement_test.py,句间续写 4 条 × ban∈{0,8,30}):**
- ban=0:3/4 条瞬间终止(T=0,0,9)——**"起点即终止"信念直接复现**(=当年 zh
  30.6% 首块截断的原病);
- ban=8/30:头部静音 8-16 帧,不随 ban 线性拉长 → 静音=被禁 EOS 的位移表达,
  模型固有"安静区"信念 ~0.4-0.6s。
- min_gen_frames 地板与 CFG 旁路均为同一病灶上的创可贴(压症状未动根)。

**根因(训练配方层)**:
1. EOS 区域填充 `labels[0,T:]=eos`(每样本 ~32 列,T 整除时整块全 eos)→ 教出
   "[内容|掩码后缀] 块型 = 终止区"的区域先验;推理续写首块正是此块型。
2. prompt_ratio≤0.3 → 续写起点"剩余文本量"的推理域(Seed-TTS ~50%)训练未覆盖,
   起点文本对齐弱,压不住区域先验。

**根修(commit bf1f639,已生效于 B2F/B2G/B2S 三臂)**:
- EOS 标签窗口 32→4 列(点状事件,保校准信号拆区域先验;单测:标签精确落于
  T..T+3,其余 -100);
- prompt_ratio_range [0,0.3]→[0,0.7];
- 推理输入官方对等:add_punctuation(ptext)(官方 preprocess_prompt 本有,输入
  正确性修复非后处理);
- min_gen_frames 降级为纯保险丝。
**验证判据(10k 探针新增)**:① ban=0 EOS 瞬燃率 → ~0;② 头部 cb0=0 串 → ≤3 帧;
③ 停长校准/CER 不回退(点状 EOS 信号量是否足够的检验)。
- 三臂已按新配方重发(旧 B2G ~4.5k 步弃,损失 ~4h·7 卡)。裁剪后处理确认不进管线。

### 2026-07-07 深夜续:内部数据双臂 step-1000 eval 死锁 + 老跑静默停摆真相
- 新 B2G/B2S 双双在 step 1000 的 dev eval 处 NCCL BROADCAST 挂 1 小时被 watchdog
  击毙(同 SeqNum 59967)。B2F(YODAS)无恙 → 嫌疑=内部数据(cephfs)读取在 eval
  内挂起。
- 更正历史:老 B2G(配方修复前)其实只到 step 1209 就**静默停摆 ~2.5h**(无
  watchdog、无报错,eval@1000 曾正常过)——删除时误以为 ~4.5k 步。cephfs 数据管道
  挂起有两种表象:eval 内(撞 collective→watchdog)与训练中(纯静默)。
- 处置:三臂配置 eval_steps→10^9(dev eval 本就不用于判决,探针才是);B2G/B2S
  重发;监视器 v2 加 step 停摆检测(20 分钟不走步即报警)。B2F 保持在跑作对照。
- 根因待查(若停摆复发):cephfs 慢读/坏 shard vs dataloader worker 死锁。

### 2026-07-08 B2F-5k 根修首读:部分改善,未过线
- ban=0 瞬停 3/4→2/4;说话时 lead0=0(干净);ban=8 静音 8-15 帧(未变)。
- <|denoise|> 线索排查:死路——我方推理显式 denoise=False,训练亦无此 token,两侧一致。
- 判读树:10k 若 0/4 → 5k 只是欠训,现配方够;若仍 ≥1/4 → 加第二杠杆:
  **句边界续写增广**(内部数据 turns 字段,prompt 切点按概率吸附句边界,
  显式监督"完整句后继续说"——恰是推理克隆的 junction 几何)。不叠加未验证修复。

### 2026-07-08 B2F-10k 判决:配方 v1 不足 → 杠杆 2 上线(11bd604)
- 10k 复测:ban=0 瞬停仍 2/4(同两条,5k→10k 轨迹平)→ 点状 EOS+prompt0.7 必要
  不充分。病根补全:**句边界 junction 训练分布里根本不存在**(均匀切点永远切在
  语流中间),"完整句后继续说"从未被监督。
- 杠杆 2:turn_boundary_prompt_prob=0.5(内部数据 turns 时间戳,prompt 切点按
  概率吸附句末,边界限定 [0.05T, 0.7T])。配置门控,B2F(在跑,v1)不受影响;
  B2G/B2S 排队中,启动即生效。单测过(边界帧数学 + 20 样本无损处理)。
- 验证:B2G-5k/10k 复跑 ban 扫描,判据不变(瞬停 0/4、lead0≤3、校准不回退)。

### B2F v1 轨迹收敛确认(5k/10k/15k 三点)
- ban=0 瞬停 2/4、2/4、2/4(同两条),ban 下静音 8-21 帧不变 → **收敛而非欠训**。
- v1(点状 EOS)必要不充分的消融证据链闭合;junction 监督(杠杆 2)为充分性候选,
  判决在 B2G-v2 的 5k/10k。B2F 此轴不再复测,专注其本职(口癖数据过滤消融)。

### T1 补测(官方裁剪前静音):停顿是基座天性,转换坏在两处
- 官方 ckpt 官方模式克隆,postprocess 关:头部低能量 0.38-1.08s(3/3 都有);
  postprocess 开也只部分裁除(1.08→0.20s,其余未动)。→ "续写起点该有停顿"
  是基座共有信念,非转换新病。
- 转换真正引入的两个 delta:① EOS 机制让"停顿"升级成"终止"(官方无停止决策,
  结构性免疫);② 停顿渲染成带声哼鸣而非干净静音(块内逐通道提交不一致:
  cb0=静音 + 残差码本=类语音;全画布是全局共精化)。
- 对 v2 的加持:句边界吸附的真值就是真人下一句的起始(含其真实停顿 0.1-0.4s)
  → B2G 同时学"别停"和"停顿该多长",无启发式。

### 哼鸣的分子级机理闭合(token 0 之谜)
- 真数字静音编码为 cb0=244(非 0);真实数据里 cb0=0 仅 0.134%(rank 275)——
  **token 0 不是静音簇,是分布外产物**。
- 完整链:模型把 ~全部概率押在 [eos] → 护栏 ban 掉 → 余下近平坦尾部 →
  class_temperature=0 的确定性 argmax 平局向最小索引坍缩 → **cb0=0 恒定**;
  残差码本分布不退化 → 提交任意类语音 token → 带声哼鸣、随说话人微调音高。
- "哼鸣 = 被禁止终止的解码器被迫发声"。从训练标签到平局仲裁一条链全部对账。

## 2026-07-08 B2F-50k 收官 + 消融判决:口癖过滤零效果("数据教坏"证伪)
- B2F 50k 跑满(12h49m 零事故)。first-100 克隆续写:句首口癖 20.0% vs B2 基线
  18.2%——**删光训练数据里的 filler 样本,生成口癖不降**。
- 结论:所谓"口癖"= junction EOS 位移哼鸣(v1 配方下原样存在),非数据模仿。
  当初"官方 0.6% vs B2 35.6%"是模型+推理路径双变量混杂。方法论教训入报告:
  表型统计需机理实验单变量定罪。
- B2F 报告定位:双否定消融(数据过滤零效果 + v1 必要不充分)。
- 现全部赌注在 B2G-5k 的 v2 判决(句边界吸附,~4-5h 后)。

### 2026-07-08 B2S 让卡(同事需用,用户指示杀一个)
- 选择杀 B2S:仅 ~100 步(首个 ckpt 在 10k,无进度可保),纯经济学对照实验,
  随时可原样重启(代码/配置全在 git:b2s_launcher.sh + train_config_block_b2s.json,
  配方 v2 同 B2G)。B2G(主力,5k 判决在即)保留。
- B2S 重启即完整复现(无状态丢失);日志归档 archive_b2s_v2_step100_yield.log。

### 2026-07-08 B2G-5k(配方 v2)首读:junction 信念未除,偏负面(判决延至 10k)
- ban=0:3/4 瞬停(T=0/0/6);ban=8/30:禁令一解除立即开火(T=10-17)。
- 但 5k 对 EOS 重学+新数据分布过早;v1 的特征是轨迹平(2/4×3 点),v2 看轨迹是否移动。
- 新发现:EOS 地板只覆盖首个生成块,越块禁令失效(T<ban 的读数由此而来)——护栏
  规格缺陷记录在案(此前模型过首块必说话,故未暴露)。
- 若 10k 不动 → v2 入"必要不充分"堆,下一杠杆:显式跨句对(ref=第 k 句,
  target=第 k+1 句,双文本真分离)。B2S 已按用户指示重新入队。

### 2026-07-08 性能线 CPU 侧审计收官（fork agent，GPU 实验已预排队）
- **修正早前判断**：attention 其实已编译——transformers 5.3.0 对 flex_attention 单例
  torch.compile（dynamic=False），create_block_mask 也走 _compile=True。代码比 450W
  功耗观感紧得多。trainer 无激活检查点（"关掉试试"命题不存在，69GB 是分配器缓存）。
- **4-6× 长文档减速 = 固有 tile 数学，非浪费**：flex 跳过全掩码 128×128 tile，可见
  tile 数 ∝ Σ(doc_len²)（20480 定长画布内）。45s vs 9s 文档 → tile 比 ~4.2×，与实测
  0.9→4.2 s/it 几乎吻合。**预期优化天花板：温和（胶水融合级）**，不承诺大赢。
- BlockMask 每步重建（omnivoice.py:425）但闭包三张 [L] 张量逐批变化 → 跨步缓存命中
  预期 ~0；memoization 臂的价值是给 mask 构建成本上界，profiler 说了算。
- 陷阱排掉一个：TrainingConfig.from_json 静默丢未知键——新旗标做成真 dataclass 字段，
  否则 A/B 臂会静默跑成 baseline。
- 补丁（perf 克隆 perf/step-time-20260708，默认 OFF 惰性）：perf_blockmask_cache
  （16 条 LRU，键=闭包全张量字节 sha1，带命中率遥测）+ perf_torch_compile（只编
  model.llm，mask 构建保持 eager 免 graph break）。CPU 单测通过。
- 预排队（tasks/，pod 一启动自动串行，各 timeout 90m）：01 基线 300 步 / 02 profiler
  trace / 03 maskcache A/B / 04 compile A/B。采纳规则：≥5% 且无重编译风暴才要。
- commits：8641768、960af3a（仅本地）。交付物 block-b2-perf/PERF_RESULTS.md。
- 附注：collator 恒定 padding [1,8,20480] 静态形状 → 若 compile 臂赢，下一步
  dynamic=False。
- 2026-07-08 21:0x 沙箱卡点修复：队列仅 gpu3 剩 3 卡（B2S 要整机 8 卡挤不进，不缩——
  缩了全局 batch 变，B2S vs B2G 同步数可比性破坏）。perf-sandbox 4→3 卡重提交
  （tasks 脚本 NUM_GPUS 同步改 3，A/B 各臂同为 3 卡仍自可比）→ 秒调度 Running，
  01 基线已在跑。B2S 继续排队等整机。

### 2026-07-08 23:17 B2G-10k v2 判决读数:脱离 5k 状态,但落在 v1 平台表型(N=4 证据弱,15k 扩样终判)
- ban=0:2/4 瞬停(T=0/6/160/137)vs 5k 的 3/4。**开火样本 lead0=0**——不 ban 且不瞬停
  时完全无哼鸣起手,哼鸣=被禁 EOS 位移的再次直接证据。
- fire-on-release 基本消失:ban=8 仅 1/4(T=17),ban=30 0/4(5k 分别为 2/4、3/4)。
  禁令下 lead0 仍 9-16 帧(哼鸣深度未变;但若瞬停归零,禁令惰性化,哼鸣自动不触发)。
- 关键对照:B2F(v1)15k 平台 = 同样 2/4 瞬停 + 禁令下正常长度——**B2G-10k 表型
  ≈ v1 收敛平台**。两解释并存:①v2 在途(5k→10k 在动,15k 续动);②v2 收敛于同一
  平台(必要不充分)。N=4 下一个样本翻转即 3/4→2/4,不足以裁决。
- 瞬停仍是同两条 prompt(10002430/10002481)——病灶 prompt 依赖(与风格门控一致)。
- **预注册 15k 协议**:扩样 12 条(tests/eos_displacement_test12.py,原 4 条+tsv 序
  接续 8 条,种子同构);判据:瞬停率显著下降(<2/12)且趋势向零 → v2 有效继续;
  瞬停 ≥3/12 且 10k→15k 平 → v2 入必要不充分堆,点火杠杆 3(显式任务 token/跨句对)。
  B2G 训练本身无论判决如何都继续(数据三角/主表/长文本目标独立于 junction 修复)。

### 2026-07-09 性能战役收官(12 训练臂+8 微基准,全程 parity 门禁)
- **可采纳组合(唯一既快又对)**:perf_liger+perf_fused_adamw = 3.120 vs 3.240 s/it
  (-3.7%,parity 0.25%)。低于 5% 线的诚实小赢;flags 在 perf 克隆分支,主树未合
  (B2G 在跑禁改)。下次发车窗口再决定合入。
- **大发现①**:torch.compile 在本栈(torch2.8+transformers5.3)全 scope 数值坏:
  max-autotune 1.245 s/it(**2.6×!**)但 loss 偏 42.6%,三种模式同值=inductor flex
  lowering 确定性错编译;排除 attention 后仍有第二处 glue 错编译(27.1%)。
  2.6× 是真实硬件余量;**最高价值下一步=torch 2.9+ 镜像复测**,不是继续调内核。
- **大发现②(未解 16× 异常=真正成本中心)**:flex backward 训练内 51.5ms/层 vs
  同 mask/形状/布局隔离 3.3ms。13 个假设全部 A/B 证伪(假设墓地在 PERF_RESULTS.md)。
  下一探针:真实训练内逐层 CUDA events。解开值 ~2×。
- 地面真相分解(04x 同步计时):bwd 1.3-2.4s 主导且随 pack 波动,fwd 290ms,
  optimizer 23ms(profiler 的 322ms 是发射队列噪声→fused AdamW 实测零效应),数据 ~0。
- P3 rank-balanced packing:科学安全性已代码验证(逐样本 position_ids/same_doc 门控/
  逐 token loss 权重→跨 rank 重组 loss 恒等),tile-count 成本模型+LPT 设计已写,未实测。
- P4 TE/fp8:**建议缓测**——16× 异常未解前,TE 移植可能继承同一未知上下文效应,
  10-15% 门槛的基线不稳。
- 交付物:block-b2-perf/PERF_RESULTS.md(裁决表+墓地+排序 open items);
  分支 perf/step-time-20260708(13 commits,8641768…cd19104,全 flag 默认 OFF)。

### 2026-07-09 DeepSpec(deepseek-ai,2026-07 新库)研读报告(research agent,应用户指令)
- 库=投机解码 draft 模型训练全栈(DSpark/DFlash/Eagle3),**非内核库**;但 DSpark/DFlash
  = Qwen3 上的 block-diffusion 式训练 + 数据依赖 mask_mod 闭包 flex BlockMask——
  **与我们同构**;其栈 torch2.9.1/triton3.5.1/tf5.10,全模型 compile 开着训旗舰 ckpt
  = **P2(inductor 错编译)是 2.8 时代 bug 的存在性证明**。
- 实验菜单(按杠杆排序):E1 torch2.9.1 复测(纯环境零代码,解锁 2.6× 最短路径);
  E2 SpecForge 独立编译 flex 模式(2.8 可用逃生舱:singleton compile flex+create_block_mask,
  per-shape mask_mod.__name__,recompile_limit 8→64);E3 GQA/layout 微 A/B
  (repeat_interleave KV vs enable_gqa+contiguous vs 绕开 transformers wrapper——
  他们两套刻意 workaround=flex GQA 有坑);E4 anchor 采样噪声块(固定 Q 预算→
  成本恒定+rank 方差结构性消灭;**配方改动=科学线提案**,须独立分支+收敛消融,待拍板)。
- P1 三条间接线索已转交异常猎手:①flex 模板 ~106KB smem/CTA 低占用对共驻内核极敏感
  +他们用 no_sync 让 backward 几乎无 NCCL 并发(归因试验:单卡/no_sync/NCCL_MAX_CTAS);
  ②闭包身份/autotune 缓存碰撞假说(per-shape 重命名+recompile_limit=他们的伤疤代码);
  ③GQA backward 原子操作/布局假说(E3 即测)。
- P4/P5 明确无货:无 fp8 attention(仅缓存存储压缩 PR);无 DualPipe 类通信重叠,
  他们的答案=大梯度累积 no_sync(0.6B DDP 本就不需要)。

### 2026-07-09 凌晨 性能线破案大捷:16× 异常 = 注意力一直在跑 FP32(全战役所有训练中招)
- **根因链**:accelerate bf16 混精保 fp32 master weights → transformers5.3 Qwen3RMSNorm
  `weight(fp32) × hidden(bf16)` 静默升格 q/k(fp32 RoPE 常量连带 v)→ flex_attention
  以 fp32 跑 head_dim=128,backward 模板寄存器压力灾难:**61.8ms vs bf16 5.1ms = 12×
  纯 dtype 差**(同 mask/形状/布局)。
- 破案法(判决性):节点级 CUDA-event 窗 ≈ CUPTI 内核时长(36.6 vs 36.45ms)= 内核
  真慢非 stall;28 层均匀;随 pack 36-76ms 波动。此前"隔离基准快"= 基准双重错形状
  (bf16+D64,真实 fp32+D128)的伪影。最小复现器 tools_perf_fp32_bench.py。
- DeepSpec 三线索处置:NCCL 共驻✗(单卡全量复现)、recompile 回退✗(TORCH_LOGS 零行)、
  GQA/布局✗(fp32 下 repeat_interleave/contiguous 无动)。
- **修复+验证(3×H100,300 步,parity 门禁)**:perf_flex_bf16_qkv 单旗 3.240→0.755 s/it
  (4.3×,parity 0.43%);**终局三旗组合(+liger+fused_adamw)= 0.610 s/it(5.3×,
  parity 0.46%)**。机理:flex 边界 cast q/k/v→bf16(梯度自动回 cast,softmax 累加
  仍 fp32)= 混精本来就该对每个 matmul 做的事。compile 2.6× 之谜同时解开:inductor
  赢的大头=顺手修了 fp32 病;eager 修复 0.610 已反超(数值坏的)compile 1.245。
- **跨战役含义:本栈所有 OmniVoice 训练(B1/B2/B2F/B2G/弹性各线)一直 fp32 注意力**;
  8 卡主跑预期同量级倍数。
- **决策(Plan C,保判决完整性)**:B2G 先跑完 15k 探针(v2 判决 5k/10k/15k 全程
  fp32 注意力=单变量干净),判决落地后停车→合三旗入主树→ckpt-15000 续跑;
  剩余 35k 步 ~41h→~8h。B2S 保排队位,起跑前若已合旗直接带旗;若先起跑则杀重提
  (从头训 500k 预算,5× 是必须)。
- torch2.9 复测(task15)降级为科学问题(2.8 inductor bug 归档用),非采用依赖。
- **2026-07-09 溯源判决:fp32 注意力是上游 k2-fsa/OmniVoice 原生 bug**(本地浅克隆
  3d2bd9d 实证):①上游 forward 默认 flex+create_block_mask(_get_packed_mask)训练,
  sdpa 仅 fallback;②骨干=transformers Qwen3(AutoModel.from_config),q_norm/k_norm
  fp32 升格链原生;③上游全部示例配置 mixed_precision=bf16(fp32 master+autocast);
  ④上游文件零 dtype cast。我们仅加 blockdiff_dual.py+一行 mask_mod 换接——易伤管线
  全部继承。**含义:官方 580kh 模型训练同付 4-5× 税**;值得上游 issue/PR(边界 cast
  一行,实测 4.3×),对外发布待用户拍板。

### 2026-07-09 B2G-15k N=12 判决:v2(句边界吸附)不充分,点火杠杆 3
- ban=0 瞬停 3/12(25%),全部来自原 4 条"难" prompt(样本 4 在 10k 曾恢复 T=137,
  15k 回退 T=0——junction 终止信念未消除,潜伏且随机复发);新 8 条 0/8 瞬停、
  lead0=0 全长生成——病灶 prompt 风格门控再确认。
- 预注册规则(≥3/12 且轨迹平 → v2 入必要不充分堆)触发:v1(点窗标注)+v2(句边界
  吸附)都只削弱不根除。杠杆 3 = 显式任务 token(prompted 样本训推同步)或显式跨句对。
- 决策:B2G 照常续训(数据三角/SIM/长文本目标独立于 junction 修复);B2S 保持 v2
  配方不动(三角可比性);杠杆 3 今夜实现成代码,新消融臂(B2H)排期待用户拍板
  (队列满,需与 B2S 争卡)。
- 2026-07-09 晨 杠杆 3a 已实现+单测通过(独立克隆 block-b2-lever3,分支
  fix/junction-lever3-crosssent):句边界吸附触发时,text 通道按推理路径逐字重建
  _combine_text(ttext, ref_text=add_punctuation(ptext))(join 方式逐样本自校验,
  pinyin 路径跳过,循环依赖用函数级 import)。单测 60/60 镜像命中+旗标关闭负对照。
  部署待定:B2H 消融臂需与 B2S 争卡+需先并入 perf 三旗(留待重启合并后 rebase),
  用户晨间拍板。
- 2026-07-09 t29 复测判决:torch2.9.1 不解禁 compile(maxautotune 2.450 s/it 但 parity
  仍炸 27.14%)。诊断线索:27.1% 恰等于 2.8 上"排除 attention 后"的腐蚀值 → 2.9 大概率
  修了 flex lowering(42.6% 分量),但第二处 glue 错编译(融合 embedding/码本头嫌疑)
  仍在。compile 线正式关闭;eager 三旗(+balanced packing)= 最终推荐不变。

## 2026-07-09 — Plan C restart VERIFIED (6.8× at 8 GPUs) + upstream PR #212 submitted

**Plan C (perf flags on production) verified green:**
- B2G resumed from `exp/block_b2g/checkpoint-20000` (explicit log line; old run archived at step 20098, ~98 steps discarded by design).
- Flags live in dumped config for BOTH lanes: `perf_flex_bf16_qkv=True, perf_liger=True, perf_fused_adamw=True`.
- Step time: full-era medians 4.040 s/it (fp32 era, 20k readings) → **0.625 s/it** (bf16 era, 17.7k readings) = **6.5× production speedup** (tail-vs-tail ~6.8×). Matches sandbox 0.610 s/it (3×H100) — 8-GPU DDP scaling cost negligible.
- Loss continuity across the boundary: 4.41/4.52 (steps 20097–98, pre) → 4.04/4.75 (20001–02, post) → 4.20–4.57 at 37k. Same band; 17k steps of healthy training on bf16 attention = long-horizon validation of the fix.
- Checkpoints 25k/30k/35k saved post-restart. B2G ETA 50k ≈ 16:10 CST 2026-07-09 (was ~27h at old speed).
- B2S (from-scratch 500k) Running on the freed node: 7000/500000 @ ~0.62 s/it, loss 5.55 and falling, grad_norm 0.70, lr warmup exactly on schedule (9.8e-5 = 2.1e-4 × 7k/15k). Survived the first-1000-step life-or-death window.

**Upstream PR submitted: k2-fsa/OmniVoice#212** — AttentionInterface-registered autocast cast for flex q/k/v (the fp32-attention root cause; upstream-native since the founding commit). Smoke-tested on stock transformers 5.13 (CPU repro: fp32 at boundary → bf16 at kernel; no-op without autocast). Corroboration from upstream PR #74 (FA2 inference, open since April, unreviewed): SDPA and FA2 both get cast protection inside transformers — flex is the only unprotected backend, and it is the one training uses. Evidence figures (cross-boundary timeline + kernel comparison) in progress for the PR comment thread.
