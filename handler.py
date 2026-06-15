"""
SalahSafe Tilāwah — RunPod serverless ASR worker (transformers pipeline).

Uses the WORKING transformers Whisper path (GPU-proven ~0.7s) with the Tadabur
Quran fine-tune (trained on ~1,400 h — the most accurate public Quran model).

We load the model weights from Tadabur but the STANDARD whisper-small tokenizer
+ feature extractor (Quran fine-tunes keep Whisper's vocab) — this sidesteps any
broken tokenizer.json in community repos.

Input  : { "input": { "audio_b64": <base64 PCM16 mono 16k>, "sample_rate": 16000,
                       "prompt": "<expected ayah words>" } }
Output : { "text": "<transcribed arabic>" }
"""
import base64

import numpy as np
import runpod
import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    pipeline,
)

MODEL_ID = "FaisaI/tadabur-Whisper-Small"
TOKENIZER_ID = "openai/whisper-small"  # same vocab as any whisper-small fine-tune

_cuda = torch.cuda.is_available()
_device = 0 if _cuda else -1
_dtype = torch.float16 if _cuda else torch.float32

_model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=_dtype)
_tokenizer = WhisperTokenizer.from_pretrained(TOKENIZER_ID)
_feature_extractor = WhisperFeatureExtractor.from_pretrained(TOKENIZER_ID)

asr = pipeline(
    "automatic-speech-recognition",
    model=_model,
    tokenizer=_tokenizer,
    feature_extractor=_feature_extractor,
    device=_device,
    torch_dtype=_dtype,
    chunk_length_s=30,
)


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
    if audio.size == 0:
        return {"text": ""}

    # Bias decoding toward the expected ayah (we know what should be recited).
    prompt = (inp.get("prompt") or "").strip()
    gen_kwargs = {}
    if prompt:
        try:
            pid = asr.tokenizer.get_prompt_ids(prompt[:300], return_tensors="pt")
            gen_kwargs = {"prompt_ids": pid.to(asr.model.device)}
        except Exception:  # noqa: BLE001
            gen_kwargs = {}

    sample = {"raw": audio, "sampling_rate": int(inp.get("sample_rate", 16000))}
    try:
        out = asr(sample, generate_kwargs=gen_kwargs) if gen_kwargs else asr(sample)
    except Exception:  # noqa: BLE001
        out = asr(sample)

    text = (out.get("text") or "").strip()
    if prompt and text.startswith(prompt[:300]):
        text = text[len(prompt[:300]):].strip()
    return {"text": text}


runpod.serverless.start({"handler": handler})
