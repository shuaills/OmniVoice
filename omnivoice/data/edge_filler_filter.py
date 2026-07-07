"""Drop samples whose transcript starts (zh/ja/ko) or starts/ends (en) with a
disfluency filler. Rationale: YODAS segments teach an utterance-edge filler
habit that block-causal decoding amplifies ~7x (data ~5% -> generations 35.6%
zh sentence-initial; en mirrors with trailing "uh" before EOS). Text-level
filter removes the transcribed mass; effect is verified by probe filler rate.

zh trailing particles (好啊/走呀) are grammatical -- zh/ja/ko filter leading
edge only. en filler words are never grammatical at either edge.
"""
import logging

logger = logging.getLogger(__name__)

_ZH_F = set("啊呃哦嗯唉诶哈呀")
_ZH_NEXT = set(",，。、!！?？ 　:：…~—") | _ZH_F
_EN_F = {"uh", "um", "ah", "oh", "hmm", "mm", "hm", "er", "uhm", "mhm", "eh"}
_JA_PREF = ("あの", "えっと", "えー", "あー", "うーん", "まあ", "なんか", "えっ", "あっ")
_JA_NEXT = set("、。,， ")
_KO_F = set("어음아")
_KO_NEXT = set(" ,，.")


def is_edge_filler(text: str, language: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    lang = (language or "").lower()
    if lang.startswith("zh"):
        return t[0] in _ZH_F and (len(t) == 1 or t[1] in _ZH_NEXT)
    if lang.startswith("en"):
        w = t.lower().split()
        return bool(w) and (
            w[0].strip(".,!?") in _EN_F or w[-1].strip(".,!?") in _EN_F
        )
    if lang.startswith("ja"):
        if t.startswith(_JA_PREF):
            return True
        return t[0] in "あえ" and len(t) > 1 and t[1] in _JA_NEXT
    if lang.startswith("ko"):
        return t[0] in _KO_F and len(t) > 1 and t[1] in _KO_NEXT
    return False


class EdgeFillerFilterDataset:
    """Transparent iterable wrapper: drops edge-filler samples, logs counters."""

    def __init__(self, dataset, log_every: int = 20000):
        self.dataset = dataset
        self.log_every = log_every

    def __iter__(self):
        kept = dropped = 0
        for sample in self.dataset:
            label = sample.get("label") if isinstance(sample, dict) else None
            if label is not None and is_edge_filler(
                label.get("text", ""),
                label.get("language_id") or label.get("language", ""),
            ):
                dropped += 1
            else:
                kept += 1
                yield sample
            total = kept + dropped
            if total % self.log_every == 0:
                logger.info(
                    "edge-filler filter: seen=%d dropped=%d (%.2f%%)",
                    total, dropped, 100.0 * dropped / total,
                )
