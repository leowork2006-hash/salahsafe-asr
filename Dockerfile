FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    runpod==1.7.7 \
    "transformers==4.44.2" \
    accelerate==0.34.2

# Bake the Tadabur Quran model + standard whisper-small tokenizer into the image
# so cold starts don't download them.
RUN python -c "from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor; \
    WhisperForConditionalGeneration.from_pretrained('FaisaI/tadabur-Whisper-Small'); \
    WhisperTokenizer.from_pretrained('openai/whisper-small'); \
    WhisperFeatureExtractor.from_pretrained('openai/whisper-small')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
