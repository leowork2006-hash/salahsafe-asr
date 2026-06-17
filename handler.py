"""
SalahSafe Tilāwah — CTC worker with FORCED ALIGNMENT (the Tarteel approach).

The Quran text is FIXED. So instead of letting the model transcribe freely
(which mishears on noisy mic audio), we force-align the audio against the KNOWN
ayah and score each expected word. A word the reciter actually said scores high
(green); a skipped/wrong word scores low (red). The model can only ever match
the real words → accuracy is bounded by the canonical text, not the recogniser.

Model: rabah2026/wav2vec2-large-xlsr-53-arabic-quran (wav2vec2-CTC, Quran-tuned).

Input  : { "input": { "audio_b64": <b64 PCM16 mono 16k>, "sample_rate": 16000,
                       "expected": "<ayah words, space-separated>" } }
Output : { "text": "<greedy decode>",
           "words": [ {"i": <wordIdx>, "conf": <0..1>}, ... ] }   # when expected given
If `expected` is omitted (or alignment fails) → falls back to { "text": <greedy> }.
"""
import base64

import numpy as np
import runpod
import torch
from torchaudio.functional import forced_align, merge_tokens
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_ID = "rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final"

_device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(_device).eval()
_tok = processor.tokenizer
_blank = _tok.pad_token_id if _tok.pad_token_id is not None else 0


def _decode_pcm16(audio_b64: str) -> np.ndarray:
    raw = base64.b64decode(audio_b64)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def handler(event):
    inp = (event or {}).get("input") or {}
    audio_b64 = inp.get("audio_b64")
    if not audio_b64:
        return {"error": "missing audio_b64"}
    try:
        audio = _decode_pcm16(audio_b64)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"bad audio: {exc}"}
    if audio.size < 800:
        return {"text": ""}

    iv = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(_device)
    with torch.no_grad():
        logits = model(iv).logits  # (1, T, C)
    greedy = processor.batch_decode(torch.argmax(logits, dim=-1))[0].strip()

    expected = (inp.get("expected") or "").strip()
    if not expected:
        return {"text": greedy}

    # Build target token ids for forced alignment (chars per word; track word index).
    words = expected.split()
    ids, word_of_tok = [], []
    for wi, w in enumerate(words):
        for ch in w:
            tid = _tok.convert_tokens_to_ids(ch)
            if tid is None or tid == _tok.unk_token_id:
                continue
            ids.append(tid)
            word_of_tok.append(wi)
    if not ids:
        return {"text": greedy}

    try:
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = torch.tensor([ids], device=_device, dtype=torch.int32)
        aligned, scores = forced_align(log_probs, targets, blank=_blank)
        spans = merge_tokens(aligned[0], scores[0])  # one span per target token, in order
        per_word: dict = {}
        for span, wi in zip(spans, word_of_tok):
            per_word.setdefault(wi, []).append(float(span.score))
        out_words = []
        for wi in range(len(words)):
            ss = per_word.get(wi)
            conf = float(np.exp(np.mean(ss))) if ss else 0.0
            out_words.append({"i": wi, "conf": round(conf, 3)})
        return {"text": greedy, "words": out_words}
    except Exception as exc:  # noqa: BLE001
        return {"text": greedy, "alignError": str(exc)}


runpod.serverless.start({"handler": handler})
