# B2：真块因果注意力（BD3-LM 双拷贝）— design/block-b2-causal-20260706

B1（单拷贝右截断）没有改注意力，块级 KV cache 数学上不可能精确。
B2 补上这一步：**训练几何 = 推理几何 = 可精确缓存的块因果**。

## 训练布局

每样本序列 `[文本前缀 P | clean 声学拷贝 | noisy 声学拷贝]`，
clean/noisy 同列共 RoPE 位置（clean 列 j 与 noisy 列 j 都在 P+j）。

注意力规则（同 packed 文档内，padding 行保留 same-doc 兜底）：

| query | 可见 keys |
|---|---|
| 前缀 | 仅前缀 |
| clean 块 b | 前缀 + clean 块 ≤ b |
| noisy 块 b | 前缀 + clean 块 < b + noisy 块 b 自身 |

- per-block mask_ratio ~ U(0,1) 独立抽；loss 只在 noisy 拷贝
  （掩码内容格 + 尾部 EOS-fill，cb0 语义与 B1 相同）。
- clean 拷贝只含完整内容块（(T//bs)·bs 列），noisy 拷贝含全部
  T//bs+1 块。**每样本监督格数与官方几乎持平（1.02×）**——双拷贝
  的代价是序列长 1.91×（有效 batch 82→43 条 @20480），不是监督缺失
  （对比 B1 单块方案每样本只监督 ~1/3）。
- 单一规则源 `_rule()`：flex mask_mod（H100 训练）与 dense 4D mask
  （sdpa/CPU/推理）共用，规则漂移由 gate1 真值表锁死。

## 推理（cache 精确性 = 本臂存在的意义）

- 前缀单独前传入 DynamicCache（只看自己 → 表征与音频无关，可缓存）。
- 逐块生成：块内迭代去掩码，每步只前传当前块（query 看全部已缓存
  keys + 块内自身）；步间 cache crop 回滚，commit 时一次前传定格该块
  KV（clean 几何 == noisy 末态几何，可见集相同）。
- 已提交块的表征只依赖前缀+更早的块 → **cache 与全量重算完全等价**：
  tiny-fp64 max|Δlogit|=0.0；真 ckpt fp32 token 逐位相同（cond/CFG）。
- [eos] 语义/CFG 旁路/max_blocks 兜底继承 B1，无改动。

## 与 B1 的关系

同一迁移 ckpt（1026, [eos]）起步，B1 vs B2 的 10k 探针差 =
"cache 兼容几何的质量代价"，报告可直接引用。

## 已知风险（按序）

1. flex mask_mod（tensor 索引 + torch.where）在 H100 create_block_mask
   编译未实测（4090 跑不了 flex）——训练第 0 步即验，失败会立刻报错。
2. 块因果去掉右侧上下文，WER/韵律代价未知——B1 vs B2 探针定量。
3. 前缀只看自己（官方前缀看全序列）：前缀表征分布偏移，属转换
   微调要学的内容之一。
