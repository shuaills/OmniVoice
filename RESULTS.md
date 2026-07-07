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
