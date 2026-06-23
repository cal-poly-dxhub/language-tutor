FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends espeak-ng libsndfile1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    transformers==4.44.0 \
    phonemizer==3.3.0 \
    websockets==13.1 \
    numpy==1.26.4 \
    soundfile==0.12.1 \
    librosa==0.10.2

# Download model at build time for faster cold starts
RUN python -c "from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor; \
    Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-xlsr-53-espeak-cv-ft'); \
    Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-xlsr-53-espeak-cv-ft')"

COPY serve.py /opt/ml/code/serve.py

EXPOSE 8080

ENTRYPOINT ["python", "/opt/ml/code/serve.py"]
