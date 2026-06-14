# Build from repo root: docker build -f examples/asr/serving/Dockerfile .
FROM nvcr.io/nvidia/nemo:26.06

WORKDIR /app/serving

# Install serving dependencies
COPY examples/asr/serving/requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# Copy serving code
COPY examples/asr/serving/ .

# Copy configs (serving + streaming reference config)
COPY examples/asr/serving/conf/ conf/
COPY examples/asr/conf/asr_streaming_inference/ ../conf/asr_streaming_inference/

EXPOSE 8000

# Single worker — GPU models are not fork-safe
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000"]
