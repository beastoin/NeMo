FROM nvcr.io/nvidia/nemo:26.02

WORKDIR /app

COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

COPY . .

EXPOSE 8000

# Single worker — GPU models are not fork-safe
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8000"]
