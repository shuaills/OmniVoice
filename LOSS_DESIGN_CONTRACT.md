# split-loss 设计契约（评审第二封定稿，2026-07-11）

L = L_audio + λ_eos·L_eos + λ_void·L_void

统计单位（不可混用分母）：
- L_audio：masked acoustic **cell** 级。L_audio = Σ_c w̄_c · (Σ_s Σ_{i∈A_sc} ℓ)/(Σ_s |A_sc|)，w̄_c = w_c/Σw。保持现行语义，不做每句先平均（长度均衡归 sampler 管）。
- L_eos：**utterance 事件**级。L_eos = (1/N_utt) Σ_s ℓ_eos_s。不继承 cb0 的 8/40 权重。
- L_void：**句内先平均**再对"有 void 的样本"平均；实现用预计算 cell_weight = w̄_c/|V_sc|。消 T mod 32 的 1~31 列方差。

实现铁律：
1. processor 显式输出 loss_kind: uint8[C,L]（IGNORE=0/ACOUSTIC=1/EOS=2/VOID=3）。**禁止按 token 值猜类**（静音帧可以是正常声学内容——句中停顿）。不变量="该位置为何受监督"。
2. model 只算一次 per_token_ce，返回结构化统计（audio_sum/count[8]、eos_sum/event_count、void_event_sum/count）；全局归一化在 trainer 做，不藏在 model 平均里。
3. DDP×GA 全局分母：分母只依赖 labels——backward 前从 accumulation window 的 collated batch 预算 category counts + all_reduce，各 rank loss 写成 W·Σw̄_c·S_rc/N_c^global。加断言防 Accelerate 自动除 GA 造成双重除。
4. λ 初始化 = 经验占比回推：λ_eos^(0)≈α_e/α_a，λ_void^(0)≈α_v/α_a（保持旧平均训练强度，只消 batch 构成漂移）。附几十个 batch 的 ‖∇‖ 三项探针。
5. 零行为埋点版必须过**逐位一致回归**（改前改后 scalar loss bit-identical）。

四臂短 fork（各 2k-8k optimizer steps，ckpt-200000 起，同数据序）：
A 原混合 loss（对照）｜B 拆分 @ 等价 λ｜C λ_eos×2｜D λ_void×2
评测组：WER/SIM（首300）+ 提前截断率/时长比 + 尾部伪影扫描 + EOS 校准曲线（按距真实结束距离分桶）+ 按句长与 T mod 32 分桶。

前置离线件：α_k 统计（纯数据侧：跑 processor 数 A/E/V 占比及按句长/mask ratio/T mod 32 分桶）→ 直接给出 λ^(0)。
