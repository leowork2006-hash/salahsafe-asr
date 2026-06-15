"""
SalahSafe Tilāwah — RunPod ASR worker, CTC edition (wav2vec2).

Architecturally correct path (fact-checked): a CTC model outputs frame-level
character probabilities in a SINGLE forward pass — far faster than Whisper's
autoregressive decoding, and the Quran fine-tune keeps it constrained to
classical Arabic. Drop-in: same input/output as before (audio_b64 → text), so
the app needs no change.

Model: rabah2026/wav2vec2-large-xlsr-53-arabic-quran (wav2vec2-CTC, Quran-tuned).

Input  : { "input": { "audio_b64": <base64 PCM16 mono 16k>, "sample_rate": 16000 } }
Output : { "text": "<transcribed arabic>" }
"""
import base64

import numpy as np
import runpod
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_ID = "rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final"

_device = "cuda" if torch.cuda.is_available() else "cpu"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(_device).eval()


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
    if audio.size < 800:  # <0.05 s
        return {"text": ""}

    try:
        iv = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).input_values.to(_device)
        with torch.no_grad():
            logits = model(iv).logits
        ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(ids)[0].strip()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"ctc failed: {exc}"}

    return {"text": text}


runpod.serverless.start({"handler": handler})
