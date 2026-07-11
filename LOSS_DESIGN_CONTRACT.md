# split-loss 设计契约 v2（评审第二封 + Codex 写码前评审修订，2026-07-11）

L_new = γ · ( L_audio + λ_eos·L_eos + λ_void·L_void )

## 1. 统计单位（不可混用分母）
- L_audio：masked acoustic **cell** 级。L_audio = Σ_c w̄_c · (Σ_s Σ_{i∈A_sc} ℓ)/(Σ_s |A_sc|)，w̄_c = w_c/Σw。保持现行语义，不做每句先平均（长度均衡归 sampler）。
- L_eos：**utterance 事件**级。L_eos = (1/N_utt) Σ_s ℓ_eos_s。
- L_void：**句内先平均**再对"有 void 的样本"平均；实现用预计算 cell_weight = w̄_c/|V_sc|。**void_events_global==0 时 L_void=0，但该 rank 仍须进入全部集合通信**（防死锁）。

## 2. loss_kind 铁律
processor 显式输出 loss_kind: uint8[C,L]（IGNORE=0/ACOUSTIC=1/EOS=2/VOID=3）。禁止按 token 值猜类（静音帧可为句中正常内容）。断言组：IGNORE ⟺ labels==-100；每 document 恰好 1 个 cb0 EOS；padding 一律 IGNORE；loss_kind 生成不消耗 RNG。split 模式要求 eos_decouple_silence=True；与 elastic 非均匀 loss_weights 共存时 fail fast（组合语义未定义）。

## 3. DDP × GA 全局分母（Codex P1-① 修订）
每 rank 每微批送入 accelerator.backward() 的标量：
```
loss_for_backward = W · G · ( Σ_c w̄_c · audio_sum_local[c]/audio_count_global[c]
                              + λ_eos · eos_sum_local/eos_events_global
                              + λ_void · void_event_sum_local/void_events_global )
```
理由：Accelerate 1.13 对非 DeepSpeed loss 自动 ÷G，DDP 梯度 ÷W；全局分母已池化 W×G 份，两次隐式除法必须预乘补偿。**只乘 W 会使 GA=8 的梯度恰为目标的 1/8；GA=1 只是藏住 bug。**
实现要求：
1. 预取恰好 G = accelerator.gradient_accumulation_steps 个 collated CPU 批；
2. 对整个本地窗口求 int64 类别计数；
3. 窗口首个前向**之前** all_reduce 计数一次，各 rank 集合通信顺序一致；
4. 可微分子项**不做** all_reduce——梯度归 DDP 归约；
5. 每微批用上式 W·G 表达；
6. iterable 数据 epoch 翻转处保持完整 G 窗口；仅末微批断言 sync_gradients；
7. DeepSpeed/FSDP 未推导前 fail fast；运行时断言并记录 Accelerate/Torch 版本与实际 W、G（**发射器激活的是外部 donor venv，锁文件不可信**）。

## 4. λ 初始化 = legacy 矩匹配（Codex P1-② 修订，替换 raw α 方案）
在**实际被消费的 legacy pack** 上统计（非语料级占比）：
```
D_c = N_audio[c] + 1[c=0]·N_eos + N_void[c]
β_a = Σ_c w̄_c·N_audio[c]/D_c ；β_e = w̄_0·N_eos/D_0 ；β_v = Σ_c w̄_c·N_void[c]/D_c
γ = E[β_a]；λ_eos = E[β_e]/E[β_a]；λ_void = E[β_v]/E[β_a]
```
这只是类别质量矩匹配，非严格等价（void 归一化有意改变）。**B 臂更名：legacy-moment-matched initialization**（不得称"等价 λ"）。附独立梯度探针（‖∇‖ 三项，数十 batch）验证。
β 统计任务规格：走 processor→filter→packer→collator 真实路径；逐 pack 记录 N_audio[c]/N_eos/N_void[c]/β_a/β_e/β_v、超长丢弃数、filler 过滤丢弃数、RNG 与配置指纹。

## 5. Arm A 与回归/测试
- **Arm A 保持 legacy 标量归约表达式逐字不变**；sidecar 指标一律 per_token_loss.detach() 派生；bit-identity（torch.equal）只要求"同 logits/labels 下 legacy 标量改前后一致"，不跨臂、不跨独立分布式运行比较。
- 日志一律来自 detached 的全局归约窗口和；W·G 反传标量与末微批 loss.item() 均非用户面数字；若报告 split eval loss，修复 eval 端"局部均值再平均"的同类错误。
- **分布式参考测试（发车门禁）**：CPU/Gloo，W=2,G=3，各 rank/微批类别计数刻意不等 + 单 rank 类别计数为零 + 全局 void 为零用例，对照"单进程拼接等效大批"的梯度/loss；另覆盖 W1/G1、W2/G1。**四臂与 Emilia 300k 在此测试通过前 HOLD。**
- B 臂捆绑了两个变化（任务拆分 + 局部比值→全局分母）；如需因果归因，另加 legacy 任务+全局分母诊断臂，默认不加。

## 6. 四臂短 fork（ckpt-200000 起，各 2k–8k optimizer steps，同数据序）
A legacy 混合（对照，逐字不变）｜B split @ legacy-moment-matched init｜C λ_eos×2｜D λ_void×2
评测组：WER/SIM（首300）+ 提前截断率/时长比 + 尾部伪影扫描 + EOS 校准曲线（按距真实结束距离分桶）+ 按句长与 T mod 32 分桶。

## 7. Emilia 对照臂附录（Codex 评审采纳）
- bt=15648 = **source-token parity**（独立数据 token 逐步等量；模型位置/算力有意 ~1.91×，非字面等量）。历史 400 样本实测展开 1.911×，15648/1.91=8192.67。
- 发车前在最终清单上实测：展开比、非 padding 装箱填充率、每步样本数/音频帧数、超长丢弃、filler 过滤丢弃（全 rank）；**parity 容差收紧到 ≤1%**。
- **filter_edge_fillers 在 parity 臂关闭**（官方基线无此数据干预；若冒烟显示 EOS 标签质量依赖它，改为声明第四处理项，不得静默保留）。
- gen_emilia_data_config.py 需校验预期 chunk 全集/计数/小时数/引用文件存在，输出清单指纹，缺失即 fail；不得静默接受缺语言/缺 dev。
- 论文两行 en 指标已澄清非矛盾：0.717/1.72/3.88 = Seed-TTS test-en；0.697/1.57/4.23 = LibriSpeech-PC test-clean。
- num_seconds=0.000 仅元数据，非装箱依据，非阻塞项。
