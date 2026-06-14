FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# Worker deps. Torch/CUDA already in the base image.
RUN pip install --no-cache-dir \
    runpod==1.7.7 \
    "transformers==4.44.2" \
    accelerate==0.34.2

COPY handler.py .

# Bake the Tarteel model into the image so cold starts don't download it.
RUN python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='tarteel-ai/whisper-base-ar-quran')"

CMD ["python", "-u", "handler.py"]
