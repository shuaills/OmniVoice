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
