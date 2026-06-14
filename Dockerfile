FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# Worker deps. Torch/CUDA already in the base image.
# Newer transformers/tokenizers so modern model tokenizer.json files parse.
RUN pip install --no-cache-dir \
    runpod==1.7.7 \
    "transformers==4.48.0" \
    "accelerate==1.2.1" \
    "tokenizers>=0.21,<0.22"

COPY handler.py .

# Bake the Tarteel model into the image so cold starts don't download it.
RUN python -c "from transformers import pipeline; pipeline('automatic-speech-recognition', model='dmoayad/whisper-medium-tarteel-quraan')"

CMD ["python", "-u", "handler.py"]
