#!/usr/bin/env bash
# B2G-50k verdict eval round (pre-registered protocol):
#   zh gen (8 shards) -> en gen (8 shards) -> WER zh+en -> SIM-o zh+en
#   -> junction instant-stop probe (12 prompts) -> zh first-100 filler probe.
# Locked main-table settings: steps-per-block 16, guidance-scale 2.0, fp16.
# Fail-soft: a failing phase records FAILED in its verdict line; later phases still run.
# Grep-able: VERDICT_* lines + B2G_EVAL_ALL_DONE. All outputs under results_b2g_50k/.
set -uo pipefail

REPO_ROOT="/opt/gpfs/users/shuai/work/block-conversion-b2/OmniVoice"
DONOR_VENV="/opt/gpfs/users/yinfeng/work/OmniVoice/.venv"
DL_ROOT="/opt/gpfs/users/yinfeng/work/OmniVoice"
MODELS="${DL_ROOT}/download/tts_eval_models"
CKPT="exp/block_b2g/checkpoint-50000"
BASE="/opt/gpfs/users/shuai/work/block-conversion/pretrained_models/OmniVoice-block"
RES="results_b2g_50k"

cd "${REPO_ROOT}"
source "${DONOR_VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}"
python -c "import omnivoice, sys; p=omnivoice.__file__; print('omnivoice from:', p); sys.exit(0 if p.startswith('${REPO_ROOT}') else 1)" || exit 1
mkdir -p logs "${RES}"
VF="${RES}/VERDICT.txt"
: > "${VF}"
verdict() { echo "$1" | tee -a "${VF}"; }

# ---------- phase 1+2: Seed-TTS generation, 8 shards x 1 GPU ----------
gen_lang() {
  local lang=$1
  local tsv="${DL_ROOT}/download/tts_eval_datasets/seedtts_testset/${lang}/test.tsv"
  local out="${RES}/seedtts_${lang}"
  mkdir -p "${out}"
  local pids=() rc=0
  NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"; NGPU="${NGPU:-8}"
  for i in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$i python tests/seedtts_blockwise_gen.py \
      --tsv "${tsv}" --ckpt "${CKPT}" --base "${BASE}" --out "${out}" \
      --steps-per-block 16 --guidance-scale 2.0 --dtype fp16 \
      --shard ${i}/${NGPU} >> "logs/b2g50k_gen_${lang}_shard${i}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return ${rc}
}

echo "[phase] zh generation ($(date -u +%H:%M:%S))"
if gen_lang zh; then verdict "GEN_ZH=OK wavs=$(ls ${RES}/seedtts_zh/*.wav 2>/dev/null | wc -l)"; else verdict "GEN_ZH=FAILED wavs=$(ls ${RES}/seedtts_zh/*.wav 2>/dev/null | wc -l)"; fi
echo "[phase] en generation ($(date -u +%H:%M:%S))"
if gen_lang en; then verdict "GEN_EN=OK wavs=$(ls ${RES}/seedtts_en/*.wav 2>/dev/null | wc -l)"; else verdict "GEN_EN=FAILED wavs=$(ls ${RES}/seedtts_en/*.wav 2>/dev/null | wc -l)"; fi

# ---------- scorer test lists (absolute ref_audio paths) ----------
python - <<'PY'
import json, os
DL = "/opt/gpfs/users/yinfeng/work/OmniVoice"
for lang, lname in (("zh", "Chinese"), ("en", "English")):
    tsv = f"{DL}/download/tts_eval_datasets/seedtts_testset/{lang}/test.tsv"
    with open(tsv) as f, open(f"results_b2g_50k/test_{lang}.jsonl", "w") as w:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            ref = p[2] if os.path.isabs(p[2]) else os.path.join(DL, p[2])
            if not os.path.exists(ref): ref = ref.replace("/download/", "/download/tts_eval_datasets/", 1)
            w.write(json.dumps({"id": p[0], "ref_text": p[1], "ref_audio": ref,
                                "text": p[3], "language_id": lang,
                                "language_name": lname}, ensure_ascii=False) + "\n")
print("TESTLISTS_BUILT")
PY

# ---------- phase 3: WER (whisper-large-v3 en / paraformer zh; uses all visible GPUs) ----------
for lang in zh en; do
  echo "[phase] WER ${lang} ($(date -u +%H:%M:%S))"
  if python omnivoice/eval/wer/seedtts.py \
      --wav-path "${RES}/seedtts_${lang}" --test-list "${RES}/test_${lang}.jsonl" \
      --model-dir "${MODELS}" --lang "${lang}" \
      --decode-path "${RES}/wer_${lang}.tsv" > "logs/b2g50k_wer_${lang}.log" 2>&1; then
    AVG=$(grep -oE "Seed-TTS WER \(Avg of WERs\): [0-9.]+%" "logs/b2g50k_wer_${lang}.log" | tail -1 | grep -oE "[0-9.]+%")
    WTD=$(grep -oE "WER \(Weighted\): [0-9.]+%" "logs/b2g50k_wer_${lang}.log" | tail -1 | grep -oE "[0-9.]+%")
    verdict "VERDICT_WER_$(echo ${lang} | tr a-z A-Z)=${AVG:-PARSE_FAILED} weighted=${WTD:-?}"
  else
    verdict "VERDICT_WER_$(echo ${lang} | tr a-z A-Z)=FAILED"
  fi
done

# ---------- phase 4: SIM-o (WavLM ECAPA-TDNN; uses all visible GPUs) ----------
for lang in zh en; do
  echo "[phase] SIM ${lang} ($(date -u +%H:%M:%S))"
  if python omnivoice/eval/speaker_similarity/sim.py \
      --wav-path "${RES}/seedtts_${lang}" --test-list "${RES}/test_${lang}.jsonl" \
      --model-dir "${MODELS}" \
      --decode-path "${RES}/sim_${lang}.tsv" > "logs/b2g50k_sim_${lang}.log" 2>&1; then
    S=$(grep -oE "SIM-o score: [0-9.]+" "logs/b2g50k_sim_${lang}.log" | tail -1 | grep -oE "[0-9.]+")
    verdict "VERDICT_SIM_$(echo ${lang} | tr a-z A-Z)=${S:-PARSE_FAILED}"
  else
    verdict "VERDICT_SIM_$(echo ${lang} | tr a-z A-Z)=FAILED"
  fi
done

# ---------- phase 5: junction instant-stop probe (12 prompts, GPU 0) ----------
echo "[phase] junction probe ($(date -u +%H:%M:%S))"
if CKPT="${CKPT}" NPROMPTS=12 CUDA_VISIBLE_DEVICES=0 \
    python tests/eos_displacement_test12.py > "logs/b2g50k_eos12.log" 2>&1; then
  {
    echo "--- junction raw table (ban=0/8/30) ---"
    grep -E "ban=0:" "logs/b2g50k_eos12.log"
  } >> "${VF}"
  STOPS=$(grep -E "ban=0:" "logs/b2g50k_eos12.log" | grep -oE "ban=0: T=[0-9]+" | grep -oE "[0-9]+$" | awk '$1 < 32 {c++} END {print c+0}')
  TOTAL=$(grep -cE "ban=0:" "logs/b2g50k_eos12.log")
  verdict "VERDICT_INSTANT_STOP=${STOPS}/${TOTAL} (criterion: ban=0 first-attempt T<32, i.e. under one block)"
else
  verdict "VERDICT_INSTANT_STOP=FAILED"
fi

# ---------- phase 6: zh first-100 sentence-initial filler probe ----------
# Method per RESULTS.md 2026-07-07 protocol: first 100 zh testset items, ASR
# hypothesis (paraformer, from the WER decode file), sentence-initial extra
# filler. Counting script was not archived; criterion reconstructed as:
# hyp starts with a filler char AND truth does not start with that char.
echo "[phase] filler probe ($(date -u +%H:%M:%S))"
python - <<'PY' >> "results_b2g_50k/VERDICT.txt" 2>&1
import os
FILLERS = set("呃嗯啊哦唔嘛哎唉诶欸呀")
ids = []
with open("/opt/gpfs/users/yinfeng/work/OmniVoice/download/tts_eval_datasets/seedtts_testset/zh/test.tsv") as f:
    for ln in f:
        p = ln.rstrip("\n").split("\t")
        if len(p) >= 4:
            ids.append(p[0])
        if len(ids) == 100:
            break
hyp = {}
truth = {}
try:
    with open("results_b2g_50k/wer_zh.tsv") as f:
        next(f)
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 4:
                name = os.path.basename(p[0]).removesuffix(".wav")
                truth[name], hyp[name] = p[2], p[3]
except FileNotFoundError:
    print("VERDICT_FILLER_FIRST100=FAILED (wer_zh.tsv missing)")
    raise SystemExit
n = hit = 0
for i in ids:
    h, t = hyp.get(i), truth.get(i)
    if not h:
        continue
    n += 1
    if h[:1] in FILLERS and (not t or t[:1] != h[:1]):
        hit += 1
        print(f"  filler_hit {i}: {h[:12]}...")
print(f"VERDICT_FILLER_FIRST100={hit}/{n} ({100.0*hit/max(n,1):.1f}%) [criterion reconstructed; B2 baseline 18.2%, pass <=8%]")
PY
grep "VERDICT_FILLER_FIRST100" "${VF}" | tail -1

echo "--- reference: B2 baseline zh 4.16%/0.693, en 3.68%/0.641 (WER/SIM); official 0.89%/0.778, 1.65%/0.741 ---" >> "${VF}"
echo "B2G_EVAL_ALL_DONE"
