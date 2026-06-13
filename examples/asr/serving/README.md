# NeMo ASR Serving

High-performance ASR inference server for NVIDIA NeMo Parakeet TDT and Nemotron Streaming models. Single GPU, dynamic batching, zero-failure production deployments.

## Performance

| GPU | Sustained RPS | torch.compile | Failures | Cost (spot) | Daily Capacity |
|-----|:------------:|:-------------:|:--------:|:-----------:|:--------------:|
| **L4** (24GB) | **49.6** | Yes | 0 | $0.50/hr | 4.3M req |
| **T4** (16GB) | **6.0** | No | 0 | $0.35/hr | 518K req |

L4 is 5.8x more cost-efficient per request. T4 is viable for low-traffic or budget-constrained deployments.

## Architecture

```
Clients ──► FastAPI Server ──► Dynamic Batcher ──► GPU Worker Thread ──► NeMo Model
  │              │                    │                    │
  │         REST /v1/transcribe   Collects requests    Single thread
  │         WS   /v1/stream       Flushes by size      torch.inference_mode
  │                                or timer             gc.collect per dispatch
  │
  └── 503 backpressure when queue full
```

All GPU inference runs on a **single dedicated thread** to avoid CUDA context contention. The server is async (FastAPI + uvicorn) but GPU work is serialized through a work queue.

Key design decisions:
- **Dynamic batching**: Incoming requests are collected and flushed as a batch when `max_batch_size` is reached or `max_wait_seconds` elapses. This is the primary throughput lever.
- **Priority queues**: Streaming requests get priority over batch to minimize real-time latency.
- **Result serialization on GPU thread**: NeMo result objects hold CUDA tensors internally. They are converted to plain Python dicts on the GPU thread to prevent cross-thread CUDA segfaults.
- **Forced GC on GPU thread**: `gc.collect()` after each dispatch ensures NeMo's internal generators (which hold CUDA pinned-memory tensors) are collected on the correct thread.

## Quick Start

### Docker Compose (simplest)

```bash
# From repo root:
docker compose -f examples/asr/serving/docker-compose.yaml up -d

# Check health:
curl http://localhost:8000/health

# Transcribe:
curl -F file=@audio.wav http://localhost:8000/v1/transcribe
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### Docker (manual)

```bash
# Build (from repo root):
docker build -f examples/asr/serving/Dockerfile.batch -t parakeet-batch .

# Run:
docker run --gpus all -p 8000:8000 parakeet-batch
```

`Dockerfile.batch` uses the PyTorch NGC base image (~5GB) instead of the full NeMo image (~20GB).

### Bare Metal

```bash
pip install nemo_toolkit[asr] fastapi uvicorn[standard] python-multipart pyyaml aiohttp numpy

cd examples/asr/serving
python server.py --config conf/serving-batch.yaml --host 0.0.0.0 --port 8000
```

### Kubernetes

```bash
# L4 GPU (production):
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml

# T4 GPU (budget):
kubectl apply -f k8s/deployment-t4.yaml -f k8s/service.yaml
```

## API Reference

### POST /v1/transcribe

Transcribe a single audio file. The request is dynamically batched with other concurrent requests for GPU efficiency.

```bash
# Basic transcription:
curl -F file=@audio.wav http://localhost:8000/v1/transcribe

# With word-level timestamps:
curl -F file=@audio.wav "http://localhost:8000/v1/transcribe?timestamps=true"
```

**Response:**
```json
{"text": "the quick brown fox jumps over the lazy dog", "audio_path": "/tmp/tmpXXXX.wav"}
```

With `timestamps=true`:
```json
{
  "text": "the quick brown fox",
  "audio_path": "/tmp/tmpXXXX.wav",
  "word": [{"word": "the", "start": 0.0, "end": 0.12}, ...]
}
```

### POST /v1/transcribe/batch

Transcribe multiple files in a single request. All files are batched together.

```bash
curl -F files=@audio1.wav -F files=@audio2.wav -F files=@audio3.wav \
  http://localhost:8000/v1/transcribe/batch
```

**Response:**
```json
{"results": [{"text": "...", "audio_path": "..."}, {"text": "...", "audio_path": "..."}, ...]}
```

Max 64 files per request. Max file size: 100MB (configurable via `max_upload_bytes`).

### WebSocket /v1/stream

Real-time streaming ASR via WebSocket. Requires Nemotron Streaming model (configured in `serving.yaml`).

```python
import asyncio, websockets, json

async def stream():
    async with websockets.connect("ws://localhost:8000/v1/stream") as ws:
        init = json.loads(await ws.recv())  # {"stream_id": "...", "status": "opened"}

        # Send 16kHz mono PCM16 audio chunks:
        await ws.send(audio_chunk_bytes)
        result = json.loads(await ws.recv())
        # {"partial_transcript": "the quick", "final_transcript": "", "is_final": false}

        # Close:
        await ws.send(json.dumps({"action": "close"}))
        final = json.loads(await ws.recv())
        # {"final_text": "the quick brown fox", "status": "closed"}
```

### GET /health

```json
{"status": "ok", "ready": true, "uptime_seconds": 1234.5}
```

### GET /metrics

```json
{
  "uptime_seconds": 1234.5,
  "batch": {"total_requests": 5000, "total_batches": 200, "total_files": 5000, "rejected_requests": 0, "pending_requests": 0},
  "stream": {"total_streams_opened": 10, "total_streams_closed": 10, "total_chunks_processed": 5000, "active_streams": 0}
}
```

### GET/POST /admin/config

Live-tune server parameters without restart.

```bash
# View current config:
curl http://localhost:8000/admin/config

# Tune (changes take effect immediately):
curl -X POST "http://localhost:8000/admin/config?max_batch_size=64&max_wait_seconds=0.01"
```

Tunable parameters:
- `max_batch_size` (1-256): GPU batch size
- `max_wait_seconds` (0.001-5.0): Max time to wait before flushing a partial batch
- `max_queue_depth` (16-8192): Max pending requests before 503
- `gpu_poll_timeout` (0.001-1.0): GPU thread poll interval

## Configuration

Three config presets are included:

| Config | File | Use Case |
|--------|------|----------|
| Full (batch + streaming) | `conf/serving.yaml` | Production with both models |
| Batch-only | `conf/serving-batch.yaml` | Batch transcription only (saves ~2GB VRAM) |
| T4 optimized | `conf/serving-batch-t4.yaml` | Cost-optimized for T4 GPU |

### Config Reference

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1                    # Must be 1 — GPU models are not fork-safe

batch_model:
  name: "nvidia/parakeet-tdt-0.6b-v3"
  device: "cuda:0"
  compile: true                 # torch.compile: +20-30% throughput, ~60s warmup
  amp: true                     # Automatic mixed precision

stream_model:                   # Omit this section for batch-only mode
  name: "nvidia/nemotron-3.5-asr-streaming-0.6b"
  device: "cuda:0"
  compile: false
  amp: true
  latency_mode: "480ms"         # "80ms", "160ms", "480ms", "1040ms"
  source_language: "English"

batcher:
  max_batch_size: 32            # Optimal with torch.compile on L4
  max_wait_seconds: 0.002       # Near-instant flush — 0.1s wastes 100ms GPU idle per cycle
  max_queue_depth: 4096         # Burst protection (256 caused cascading 503s in testing)
  max_upload_bytes: 104857600   # 100MB max per file

stream:
  max_concurrent_streams: 128
  chunk_duration_ms: 160
  sample_rate: 16000
  max_stream_duration: 1800     # Auto-close streams after 30 minutes
  max_chunk_bytes: 524288       # 512KB max per WebSocket frame
```

### Tuning Guide

**Batch size**: With torch.compile, `batch_size=32` outperforms `batch_size=96` because smaller batches cycle through compiled kernels faster. Without torch.compile, larger batches (64-96) can be more efficient.

**max_wait_seconds**: Set to `0.002` (2ms). The default `0.1` (100ms) wastes GPU time waiting for more requests. At high load the batch fills before the timer fires anyway.

**max_queue_depth**: Set to `4096`. In testing, `256` caused cascading 503 failures under burst traffic. High queue depth absorbs bursts while the GPU catches up.

**torch.compile**: Enable on L4 and newer GPUs (Ada Lovelace+). Disable on T4 — concurrent compiled inference triggers CUDA segfaults on Turing architecture.

## Scaling

### Horizontal (multi-GPU)

Each server instance uses one GPU. Scale horizontally by adding replicas.

The included `k8s/hpa.yaml` provides HPA (Horizontal Pod Autoscaler) configuration:
- Scale 1-4 replicas based on CPU utilization
- 60s stabilization window for scale-up
- 300s stabilization window for scale-down

With 4x L4 GPUs: **~200 RPS** sustained.

### GPU Selection Guide

| GPU | VRAM | RPS | torch.compile | Cost/hr (spot) | Best For |
|-----|:----:|:---:|:-------------:|:--------------:|----------|
| L4 | 24GB | 49.6 | Yes | $0.50 | Production workloads |
| T4 | 16GB | 6.0 | No | $0.35 | Dev/staging, low traffic |
| A100 | 40/80GB | ~100+ | Yes | $1.50+ | High throughput |

### Capacity Planning

```
daily_requests = rps * 86400
```

| GPU | 1 replica | 2 replicas | 4 replicas |
|-----|:---------:|:----------:|:----------:|
| L4 | 4.3M/day | 8.6M/day | 17.1M/day |
| T4 | 518K/day | 1.0M/day | 2.1M/day |

## Benchmarking

### Quick Stress Test

```bash
pip install aiohttp numpy

# Batch throughput (default: 32 concurrent, 200 requests):
python stress_test.py --server http://localhost:8000 --mode batch

# With real speech audio (requires espeak-ng):
python stress_test.py --server http://localhost:8000 --mode batch --real-audio

# Streaming:
python stress_test.py --server http://localhost:8000 --mode stream --streams 16

# Mixed batch + streaming:
python stress_test.py --server http://localhost:8000 --mode mixed

# Transcription quality (WER measurement):
python stress_test.py --server http://localhost:8000 --mode quality
```

### Industrial Benchmark

Full performance characterization: warmup, concurrency sweep (1-128), audio duration sweep (1s-60s), sustained load.

```bash
python benchmark.py --server http://localhost:8000 --sustained-minutes 5 --output report.json
```

### In-Cluster Benchmark (K8s)

```bash
kubectl apply -f k8s/benchmark-configs.yaml
kubectl logs -f job/benchmark-configs
```

## Files

```
examples/asr/serving/
  server.py              # FastAPI application, REST + WebSocket endpoints
  gpu_worker.py          # Dedicated GPU inference thread
  batch_engine.py        # Dynamic batching engine
  stream_engine.py       # Streaming session manager
  stress_test.py         # Stress test + quality test client
  benchmark.py           # Industrial performance benchmark
  requirements-serving.txt

  conf/
    serving.yaml         # Full config (batch + streaming)
    serving-batch.yaml   # Batch-only (no streaming model)
    serving-batch-t4.yaml # T4 cost-optimized

  Dockerfile             # Full NeMo image (batch + streaming)
  Dockerfile.batch       # Lightweight batch-only (~5GB vs ~20GB)
  docker-compose.yaml    # Single-command local deploy

  k8s/
    deployment.yaml      # L4 production deployment + PVC
    deployment-t4.yaml   # T4 budget deployment
    service.yaml         # ClusterIP service
    hpa.yaml             # Horizontal Pod Autoscaler
    benchmark-configs.yaml # In-cluster benchmark job
```

## Known Limitations

- **Single GPU per instance**: The server binds to one GPU. Scale horizontally for more throughput.
- **T4 concurrency cap**: Concurrent load above 16 can trigger CUDA host allocator segfaults on T4 (Turing architecture). This is a NeMo RNNT decoder thread-safety issue distinct from #15771.
- **torch.compile warmup**: First inference after startup takes ~60 seconds for kernel compilation. Subsequent requests are fast.
- **workers=1 required**: GPU models are not fork-safe. The `workers` config must be 1.
