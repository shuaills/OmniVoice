# split-loss 实验台账（block-loss-design）

## 2026-07-11 Slice 1 + 1.1：loss_kind 基建 + β 矩匹配统计定稿

- **实现**：loss_kind 零 RNG 发射（纯函数 build_loss_kind）/ collator 穿线 + 混合出现保护 / 打包器超长丢弃日志 / 6 项契约测试全过。c29f0d1 + a894e19。
- **β 统计（2000 packs，500k 配方，bt 20480，seed 42，指纹全）**：
  - γ = E[β_a] = **0.9678**
  - **λ_eos^(0) = 4.24e-4**（bootstrap 95% CI [4.13, 4.36]e-4）
  - **λ_void^(0) = 3.29e-2**（CI [3.19, 3.38]e-2）
  - 5-pack 冒烟值（2.2e-4 / 1.5e-2）系小样本偏低，以 2000-pack 为准。
  - 验证链：β_a+β_e+β_v ≡ Σ w̄_c 恒等式最大误差 2.2e-16（机器精度）；D_c==0 零发生；n_docs≡n_eos 逐 pack 相等（唯一-EOS 的第二重独立验证）；超长丢弃 258 次已计数。
  - n_docs/pack 均值 5.62（p5=1, p95=13）——单 pack 文档数波动 13×，正是"局部比值漂移"问题的直接观测。
- **Codex 复审判词（切片 1）**：核心实现 solid；legacy 分支继续发 loss_kind（模式无关元数据），split trainer 对 non-decouple fail-fast；不变量断言 always-on 保留；β_e 公式确认；loader rebuild 限定表述为"确定性单进程采样"；moment matching 只匹配类别质量不匹配梯度——**梯度范数探针仍是门禁**。
- 下一步：Slice 2 = trainer 侧 W·G 窗口 loss，按 Codex 四层结构（纯计数函数 / GA 窗口协调器（不产生短窗口）/ 单次定序 all_reduce（设备随 backend）/ 每微批 W·G 可微表达式（零分母走一致分支、图连通零））。测试矩阵：W1/G1、W2/G1、W2/G3、rank 级类别为零、全局 void 为零、对照单进程拼接大批的最终梯度、sync_gradients 仅末微批、Arm A 逐字不动。HOLD 维持。

## 2026-07-12 午后 Slice 2 集群门禁：数学 4/4 + Gloo 矩阵 6/6 全过

Codex 交付 d61fd24（实现+本地 21 测全绿），集群 donor 环境复验：
- 数学单测 4/4（逐文档静音加权 / 零类别 / 零 EOS fail-fast / 图连通零分母）
- **Gloo 矩阵 6/6**：W1G1、W2G1（rank 零静音 + 全局零静音）、W2G3、原 loss 逐位一致（带/不带 loss_kind）、窗口协调器跨 epoch 无短窗
- 剩余门禁：双卡 CUDA 冒烟（G1/G3 防集合通信死锁）+ 真批量逐位一致 —— 排明早 R2 训完腾卡时，随后 R3 发车（配置：split_loss=true，γ=0.8718，λ_eos=1.864e-3，λ_void=0.14516，G=1，梯度检查点开）
跑通门禁踩掉三个分布式测试基建雷（全部入档）：
1. spawn 子进程重新 import 测试模块——父进程运行时注入（sys.modules 假 pytest）带不过去 → 依赖必须落成真文件（tools/pytest_shim/）
2. 入口脚本无 __main__ 保护 → spawn 重放主模块递归起跑（"running case" 双打印是签名）
3. 多网卡机器 Gloo 不指定 GLOO_SOCKET_IFNAME=lo 会在 TCP store 初始化处无限挂死（表现=空日志假死）

## 2026-07-16: post-speech tail campaign handover

- **Decode safety net implemented** on `campaign/silence-force-stop-20260716`:
  a blockwise fallback stops after a configurable continuous digital-silence
  run, excludes the complete reference prompt, respects the existing minimum
  generation gate, gives an earlier EOS priority, and trims a verified quiet
  run back to its onset. The disabled path is token- and RNG-identical.
  Implementation commits: `fbe08e1`, `6f8e255`.
- **Local gates:** 42 targeted tests passed, including force-stop ordering,
  block-boundary runs, prompt exclusion, disabled-path identity, band-label
  tests, split-loss math, and the Gloo matrix. Ruff passed on every changed
  Python file.
- **Generated-token calibration, en first-300, same 300k x20 checkpoint and
  seeds, 25-frame threshold:** exact codebook-0 matched a qualifying run in
  24/300 samples; exact first two codebooks in 22/300; exact first three,
  first four, and all eight in 0/300. The first-two detector retained almost
  all codebook-0 recall while being much safer against natural pauses, so the
  paired and full-set rule is **1.0 s + first two codebooks**. Raw artifact:
  `/opt/gpfs/users/shuai/work/silence-force-stop-campaign/results/pilot_en300_20260716/summary.json`.
- **Natural Emilia token-tail sample, 8,000 utterances:** the synthetic void
  vector is not a natural-tail signature. No terminal run of the first two
  codebooks reached two frames; all-eight matches were absent. Codebook-0-only
  had four English mid-clip runs of at least 25 frames, which rules it out as
  an unguarded production detector.
- **Paired waveform-tail sample, 1,000 utterances spread across all 20 Emilia
  chunks:** at the central sustained high-frequency activity threshold,
  speech-end-to-clip-end gap was mean 0.092 s, median 0.066 s, p95 0.255 s;
  only 4/1000 exceeded 0.5 s and 2/1000 exceeded 1.0 s. The 25/30/35 dB
  sensitivity bracket gives the same conclusion. Dirty long tails exist, but
  are too rare and too short to explain the distribution-wide late-stop bias;
  broad relabeling is therefore **held**, pending a stronger forced-alignment
  check. Raw artifacts:
  `/opt/gpfs/users/shuai/work/silence-force-stop-campaign/results/emilia_waveform_tail_1000/`.
- **Scheduler/compliance:** the decode pilots and waveform audit each ran as
  their own self-terminating OMS job. The k=1 control and k=4 EOS-band 10k arms
  are queued as separate single-node 8xH100 jobs; neither launcher contains a
  guardian sleep or uses pod-console injection.
