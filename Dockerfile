FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    runpod==1.7.7 \
    "ctranslate2==4.4.0" \
    "faster-whisper==1.0.3" \
    "transformers==4.44.2" \
    "nvidia-cublas-cu12" \
    "nvidia-cudnn-cu12==9.1.0.70"

# ctranslate2 must find cuDNN/cuBLAS at runtime or it crawls on CPU (~40s/clip).
# Put the pip-installed NVIDIA libs on the loader path.
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}

# Convert the Quran-tuned Whisper to CTranslate2 INT8 — ~4x faster inference
# (the accessible equivalent of Tarteel's TensorRT optimization).
RUN ct2-transformers-converter \
    --model basharalrfooh/whisper-small-quran \
    --output_dir /app/model_ct2 \
    --quantization int8_float16 --force

# Quran fine-tunes keep Whisper's original vocab, so use the standard
# whisper-small tokenizer + feature extractor (reliable, complete files).
RUN python -c "from transformers import WhisperTokenizerFast, WhisperFeatureExtractor; \
    WhisperTokenizerFast.from_pretrained('openai/whisper-small').save_pretrained('/app/model_ct2'); \
    WhisperFeatureExtractor.from_pretrained('openai/whisper-small').save_pretrained('/app/model_ct2')"

# Pre-download the Silero VAD so cold starts don't fetch it.
RUN python -c "from faster_whisper.vad import get_vad_model; get_vad_model()" || true

COPY handler.py .

CMD ["python", "-u", "handler.py"]
