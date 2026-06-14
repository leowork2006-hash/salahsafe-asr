"""
SalahSafe Tilāwah — RunPod serverless ASR worker.

Realtime, Tarteel-style: the phone streams SHORT rolling windows (~the last
2–3 s of audio) and this worker transcribes just that window with Tarteel's
public Quran fine-tune. The app then aligns the text against the KNOWN ayah to
light words green/red (mistake / makharij). We never send back "scripture" —
only what was heard, for matching.

Input  : { "input": { "audio_b64": <base64 PCM16 mono>, "sample_rate": 16000 } }
Output : { "text": "<transcribed arabic>" }
"""
import base64

import numpy as np
import runpod
import torch
from transformers import pipeline

MODEL_ID = "tarteel-ai/whisper-base-ar-quran"

_use_cuda = torch.cuda.is_available()
_device = 0 if _use_cuda else -1
_dtype = torch.float16 if _use_cuda else torch.float32

# Loaded once per worker, then reused for every request (warm = instant).
asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    device=_device,
    torch_dtype=_dtype,
    chunk_length_s=30,
)


def _decode_pcm16(audio_b64: str) -> np.ndarray:
    raw = base64.b64decode(audio_b64)
    # int16 little-endian mono -> float32 in [-1, 1]
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def handler(event):
    inp = (event or {}).get("input") or {}
    audio_b64 = inp.get("audio_b64")
    sample_rate = int(inp.get("sample_rate", 16000))
    if not audio_b64:
        return {"error": "missing audio_b64"}

    try:
        audio = _decode_pcm16(audio_b64)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"bad audio: {exc}"}

    if audio.size == 0:
        return {"text": ""}

    out = asr(
        {"raw": audio, "sampling_rate": sample_rate},
        generate_kwargs={"language": "ar", "task": "transcribe"},
    )
    return {"text": (out.get("text") or "").strip()}


runpod.serverless.start({"handler": handler})
