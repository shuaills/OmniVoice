# OmniVoice Block-Diffusion 转换（B 系列）设计 — v0 精简版

分支：`design/block-conversion-20260706`（基于 design/elastic-canvas-20260705 c80576c）
工作区：`/opt/gpfs/users/shuai/work/block-conversion/`（与 elastic-canvas 完全独立）
方向拍板：块间单向、块内双向、EOS 停止；不搬 Chatterbox 校准补丁，效果说话。

## 1. 目标

把 OmniVoice 的"全画布双向 + 规则时长"改成"逐块生成 + 模型自己决定何时停"：
- 总长不再由 Eq.4 规则时长给定 —— 这是转换的全部意义；
- 为 v1 streaming 铺路（块级 KV cache 是后续独立优化，见 §5）。

## 2. 方案：单拷贝"当前块"监督（v0）

**训练**：每个样本抽一个当前块 c，画布在其右边界截断：
- 块 < c：干净（已提交上下文），无 loss；
- 块 c：逐格 Bernoulli(mask_ratio~U(0,1)) 掩码，只在掩码格算 loss（官方语义限制在块 c）；
- 内容尾部所在块：右侧补 **EOS-fill 格**（输入=mask，cb0 label=[eos]=1025，其余码本无监督），
  内容恰好块对齐时追加一个纯 fill 块 —— "该停了"是一个可达的训练状态；
- attention = 官方双向 mask 原样作用在截断后的序列上，**零 attention 新代码**。

**推理**（`generate_blockwise`）：growing canvas，逐块用官方置信度选位内环去掩码；
块提交后 cb0 扫到 [eos] → 截断停止；max_blocks 只是安全网不是目标长度。
v0 全量重算（无 KV cache），训练与推理几何完全一致。

**CFG**：普通类走官方两分支外推；**[eos] 类无条件旁路 CFG**（E1 疤痕：稀有类
logit 噪声被 c+s(c−u) 放大），条件分支单独给分，**不设配置开关**。

**词表**：1025 → 1026，[eos]=1025 仅 cb0 有语义；从**官方原版** ckpt 迁移
（scripts/migrate_block_ckpt.py，非 OmniVoice-elastic —— 与弹性臂单变量独立）。

## 3. 为什么不是 BD3-LM 双拷贝（冲突记录）

双拷贝（clean 流 + noisy 流）要求 prefix/clean 流与 noisy 流注意力隔离才可缓存，
因此**任何配置都无法复现官方双向几何**——官方里 prefix 看得到声学区，深层 KV
必然不同 → "退化等价锚点门（单块 == 官方，数值一致）"在该方案下数学上不可满足。
单拷贝方案逐字节通过该门（本仓库文化里锚点门优先级最高：offsets bug 即由此类门抓获）。

代价：每样本只监督一个块（监督密度 ≈ 官方的 1/3~1/2），部分由截断样本在
batch_tokens 下打包更密补回。**升级路径**：训练侧引入双拷贝提高监督密度
（B2 候选），推理路径不变。

## 4. 已过的四道门（数字见 RESULTS.md）

1. processor 不变量（300 样本 + prompt 守卫 100）；
2. 退化等价：block_size ≥ 画布 + eos 关 → 与官方处理器 200 seed 逐字节相同；
   经同一迁移模型 fp32 loss 十位小数一致（8.4748458862）；
3. 迁移门：1025→1026 共享词表切片 fp32 **Δlogit = 0（bit-exact）**；
4. generate_blockwise 机械：id 全法、EOS 截断正确、max_blocks 兜底、
   未迁移 ckpt 拒载。

## 5. 遗留项（按优先级）

- **B1 训练**（已 staged：train_config_block_b1.json + b1_launcher.sh，8×H100 50k 步）；
  10k 探针判决指标：① fixed-mode 质量不塌（对照 e11 的 CER 口径）② eos 停点分布
  是否合理（生成长度 vs 文本长度的相关性）③ RTF/块。
- 块级 KV cache（唯一近似：已提交块 KV 冻结在提交时刻；接口注释在 _blockwise_decode）。
- 流式声码器衔接（按块出 wav）。
- 监督密度提升（双拷贝训练侧，B2 候选）。
- eos 停点校准（若过早/过晚停：fill 长度分布或 eos 类先验调整；Chatterbox 的
  prior-calibrated scoring 留作弹药，不预搬）。

## 6. 与弹性臂的关系

同基座、不同词表扩展（elastic 1027 / block 1026）、不同工作区、互不依赖。
合流点在"长度政策"：弹性的尾部 [expand] 决策与 block 的 [eos] 决策是同一块
肌肉的两种问法；E2 若合训需先统一词表布局（预留讨论，不在 v0 范围）。
