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
