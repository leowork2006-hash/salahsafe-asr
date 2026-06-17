# SalahSafe Tilāwah — CTC (wav2vec2) ASR worker.
# cuDNN 9 in the system path so torch uses the GPU properly.
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# torch (CUDA 12.1 wheels work on the 12.3 runtime) + transformers + runpod
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir runpod==1.7.7 "transformers==4.44.2"

# Bake the Quran-tuned wav2vec2-CTC model so cold starts don't download it.
RUN python3 -c "from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor; \
    Wav2Vec2Processor.from_pretrained('rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final'); \
    Wav2Vec2ForCTC.from_pretrained('rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final')"

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
