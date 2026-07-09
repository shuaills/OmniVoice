# DESIGN — junction/EOS 战役（2026-07-09/10 夜间设计稿）

主线判决后的下一仗：块因果配方在三条接缝上的终止/起始病理。本稿汇总全部实证、机理模型、AR 世界可移植修法、以及预注册的作战序列。真相以 RESULTS.md 为准；本稿是作战计划。

## 1. 战场：三表型、三来源（全部实证，见 RESULTS 深夜 III/IV）

| 表型 | 发生率 | 来源判决 | 关键证据 |
|---|---|---|---|
| 瞬停截断（系统性） | B2G 全集 ~2%（双语一致） | **转换/init 携带** + **min_gen_frames seed_rem bug 使能** | B2S 素面 0 条；floor≥32 下 frames<32 本应不可能；rescue 重跑进行中 |
| 瞬停（prompt 型） | 单例 | prompt 零尾静音 | +0.15s 治愈双 init（T=1→162/155） |
| 首块哼鸣 | 双 init 85–90%（声学） | **配方几何放大**（数据先验仅 ~28%，续写几何 14%） | fillerometer 84 vs 88；token 法证=cb0 全 token 0 |
| 末块残差异响 | 常见 | 并行去掩码残差提交 | 时长法证 100/100，vocoder 无罪 |

## 2. 机理模型（当前最优解释，全部有实证锚点）

1. **junction 不确定性 → cb0 退化吸引子 token 0**：首块无声学历史可依，模型对 cb0 的后验塌缩到 token 0（数据 rank ~275 的罕见 token，非真实静音——数据静音是 244 为首的分布）；上层码本提交半随机残差；vocoder 渲染成 10–15Hz 哼鸣。平坦尾（ban 后）同一 token —— 两案一犯。
2. **放大而非复读**：数据 onset 声学先验 ~28%（gap-inclusive 14%），生成 85–90% —— 配方放大 3–6×。与词汇口癖 7× 放大同构。CFG 对特殊类外推的放大（E1 单因子结论）是头号嫌疑放大器。
3. **KV 自激级联**：哼鸣块进干净 KV → 下块续哼（strip 条件多块 ememem 实证）。
4. **EOS 终止先验**（v1 已从 32 列收到 4 列，必要不充分）：B2S 100k/200k 生长曲线将判定其残留权重。

## 3. AR 经验映射（web 勘察，一手源验证；完整报告在会话记录）

三条接缝，AR 世界各有成熟修法，无一系统裸奔：

| 接缝 | AR 标准做法 | 我们的对应 |
|---|---|---|
| prompt→生成 | ref trim 后补固定 0.2–0.3s 静音（CosyVoice 0.2s / GPT-SoVITS 0.3s 生产代码）；VoiceCraft 枚举 silence_tokens 逐 token 惩罚 | padding 判决（0.15s 治愈）+ token-0 onset ban（已武装） |
| 生成起点 EOS | EOS 禁窗按文本长度缩放（CosyVoice min_token_text_ratio=2，max 20×）；Whisper begin_suppress | min_gen_frames（bug 已修，建议升级为按剩余文本缩放 + 跨块正确计数） |
| 语音尾→EOS | Tortoise calm_token=83 改写 stop 后全部 token（"否则 BLAH 异响"——与我们残差提交同病）；Bark min_eos_p=0.2 eager-stop | 残差提交修法原型：EOS 判决后末块剩余格子写死真实静音 token（244 族） |

数据侧共识：边缘 trim + **一致的** onset/offset 边距（Emilia 30ms / VCTK trim+0.25s / FireRedTTS 0.3s）——关键是一致性，不是零静音。转写侧：Székely SSW2019 受控实证"音频有呼吸+文本不写 → 模型无提示插入口癖"= 我们哼鸣机制的发表版；现代解法 = 呼吸打标（CosyVoice2 [breath]）。

## 4. 作战序列（预注册）

**第一梯队（今夜，零训练，4090）**
- P1 rescue 重跑（进行中）：63 条截断 utt × 修复采样器 → 若回收，headline zh~3.5/en~3.8 零训练兑现，全集 re-eval 提请批准。
- P2 token-0 onset ban（已武装，ONSET_BAN_BLOCKS/ONSET_BAN_TOKEN_ID @2a3a158）：junction-12 + gen-100 + fillerometer。判据：哼鸣 <20% 且 WER-100 无回归 → 立即可用的推理侧修法；若吸引子转移到其他 token → 记录新 token，改用 VoiceCraft 名单法。
- P3 静音种子（备选）：用真实数据 onset token（244 族）预填首块头几帧，测级联是否断裂 —— 区分"吸引子问题"vs"冷启动问题"。

**第二梯队（输入规范 + 解码规范，随 P1/P2 判决落地）**
- I1 prompt 端点归一化规范：trim 尾部 →补固定 0.2s 数字静音（padding 判决 + CosyVoice/GPT-SoVITS 双背书）。写进 eval 与 serving 的输入预处理契约（输入规范≠输出后处理，不违"禁 workaround"军规）。
- D1 min_gen_frames 升级：主树应用 seed_rem 修复（B2S 安全窗）+ 按剩余文本缩放下限、20× 上限。
- D2 eager-EOS + 末块静音改写（Tortoise 原型）：EOS 判决后，末块未提交格子写 244 族静音而非任其残差提交。

**第三梯队（训练侧根修，需卡/排期，待拍板）**
- **T1+T2 合并升级：EOS/padding 角色解耦（文本 dLLM 文献同构印证，用户勘察 2026-07-10）**。
  文献链：-EOS paper（arXiv 2601.22527，LLaDA 类 EOS-padding → EOS Trap：早期 denoise EOS confidence 异常高）；Rainbow Padding（2510.03680，`<eos> overflow` 归因 EOS 双重角色=终止符+空位 placeholder，修法=单 EOS + 循环 pad 分散概率质量）；VoidPadding（2606.17999，[VOID] 专管 padding，EOS 只管 semantic termination）；SSD-2（EOS pad 到 block 边界=工程 trick 非语义）。**我们的 labels[0,T:]=eos（v1 收到 4 列）正是被批判的 dual-role 设计；三表型与 `<eos> overflow` 同构**（文本=短回答/EOS 串，音频=瞬停/哼鸣/尾部残差）。
  **音频版解耦标签规范（草案，实验臂待拍板）**：
  1. EOS 只做 stop event：单列（或 ≤2 列鲁棒窗）labels[0,T]=eos。
  2. T+1 起一个 block 内：全 8 码本监督为**真实静音帧**（音频天然 [VOID]=数据静音族，今晚已测得各码本静音众数：cb0=244、cb1=354、cb4=433、cb6=926、cb7=419/858；用真实静音而非人造 pad token=有数据支撑）。防新吸引子：只监督 T+1..T+32 一个 block，之外 ignore（Rainbow 用循环 pad 防单 token 独大的同款顾虑）。
  3. 推理契约：首个 EOS 即停、EOS 后格子永不进 vocoder（现 stop_abs 裁切已保证）；EOS 判决从单点 confidence 升级为块内 EOS density/survival（抗噪）。
  4. 判据：junction-12 = 0/12、fillerometer 哼鸣 <20/100、lead_hum_s 中位 →0、全集 frames<32=0（修复采样器下）、WER/SIM 无回归；对照臂=现 v2 配方同步数。
- T2' 生长曲线判据仍跑：B2S 100k/200k junction+哼鸣复测（病随训练涨 → dual-role 残留权重定量）。
- T3 onset 一致性数据卫生：turn 起点 trim 到一致边距（AR 共识）；鉴于数据先验只有 28%，优先级低于 T1/T2。
- T4 CFG 特殊类旁路（E1 移植）：cb0 的 onset 期 CFG 旁路/降幅，压放大器而非压症状。

## 5. 仪器与判据（全部已交付使用）

junction-12（分钟级，确定性复现）；fillerometer v1（耳测验证，阈 0.25；v2 run-length 待做）；截断法证（frames vs 时长）；token 法证（DUMP_TOKENS_DIR）；全集 eval（8 卡 30 分钟）。
统一过线判据：哼鸣 <20/100、junction 0/12、全集 frames<32 = 0、WER-100 无回归、SIM 平。

## 6. 待用户拍板

- P1 若回收 → 全集 re-eval 作业提交（需批准）。
- 第三梯队训练臂的排期与占卡（与 B2S 500k 争卡）。
- tier-2/3 上游修复（transformers cast PR + PyTorch issue）与 PR#212 证据评论（包已备待审）。
