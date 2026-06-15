"""
SalahSafe Tilāwah — RunPod serverless ASR worker (faster-whisper / CTranslate2).

Tarteel's real architecture, scaled to budget:
  • Quran-tuned Whisper (whisper-small-quran) converted to CTranslate2 INT8 → ~4x
    faster inference (the accessible version of their TensorRT).
  • Built-in Silero VAD (vad_filter) → skips silence, like Tarteel's VAD stage.
  • initial_prompt = the expected ayah → biases decoding to the right words.
  • beam_size=1 → lowest latency.

The app aligns the returned text against the KNOWN mushaf — we never render the
model output as scripture.

Input  : { "input": { "audio_b64": <base64 PCM16 mono 16k>, "sample_rate": 16000,
                       "prompt": "<expected ayah words>" } }
Output : { "text": "<transcribed arabic>" }
"""
import base64

import numpy as np
import runpod
from faster_whisper import WhisperModel

# int8_float16 on GPU = fast + tiny VRAM (~0.5 GB for small).
model = WhisperModel("/app/model_ct2", device="cuda", compute_type="int8_float16")


def _decode_pcm16(audio_b64: str) -> np.ndarray:
    raw = base64.b64decode(audio_b64)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def handler(event):
    inp = (event or {}).get("input") or {}
    audio_b64 = inp.get("audio_b64")
    if not audio_b64:
        return {"error": "missing audio_b64"}
    prompt = (inp.get("prompt") or "").strip() or None

    try:
        audio = _decode_pcm16(audio_b64)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"bad audio: {exc}"}
    if audio.size == 0:
        return {"text": ""}

    try:
        segments, _info = model.transcribe(
            audio,
            language="ar",
            initial_prompt=prompt,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text for s in segments).strip()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"transcribe failed: {exc}"}

    return {"text": text}


runpod.serverless.start({"handler": handler})
