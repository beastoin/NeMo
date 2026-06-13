# NeMo ASR Batch Serving

High-throughput batch transcription server for NVIDIA NeMo Parakeet TDT 0.6B. Dynamic batching on a single GPU, zero-failure production deployments, deploy anywhere from Docker to Kubernetes.

## Performance

| GPU | Sustained RPS | torch.compile | Failures | Cost (spot) | Daily Capacity |
|-----|:------------:|:-------------:|:--------:|:-----------:|:--------------:|
| **L4** (24GB) | **49.6** | Yes | 0 | $0.50/hr | 4.3M req |
| **T4** (16GB) | **6.0** | No | 0 | $0.35/hr | 518K req |

L4 is **5.8x more cost-efficient** per request ($0.0028 vs $0.0162 per 1K requests). T4 is viable for low-traffic or budget-constrained deployments.

All benchmarks: real speech audio (espeak-ng TTS), 0% WER, 0 failures.

## Architecture

```
                         ┌─────────────────────────────────────────────┐
                         │              FastAPI Server                 │
                         │         (async, single process)             │
                         └──────┬──────────────────┬──────────────────┘
                                │                  │
                     POST /v1/transcribe    POST /v1/transcribe/batch
                         (single file)        (up to 64 files)
                                │                  │
                                ▼                  ▼
                         ┌─────────────────────────────────────────────┐
                         │           Dynamic Batch Engine              │
                         │                                             │
                         │  Collects incoming requests into batches.   │
                         │  Flushes when:                              │
                         │    • max_batch_size reached (default: 32)   │
                         │    • max_wait_seconds elapsed (default: 2ms)│
                         │  Backpressure: 503 when queue > 4096        │
                         └──────────────────┬──────────────────────────┘
                                            │
                                      Work Queue
                                            │
                                            ▼
                         ┌─────────────────────────────────────────────┐
                         │          GPU Worker Thread                  │
                         │                                             │
                         │  Single dedicated thread for all inference. │
                         │  • torch.inference_mode()                   │
                         │  • Serializes results to plain Python dicts │
                         │  • gc.collect() after each dispatch         │
                         │  • cudnn.benchmark + float32 high precision │
                         └──────────────────┬──────────────────────────┘
                                            │
                                            ▼
                         ┌─────────────────────────────────────────────┐
                         │     Parakeet TDT 0.6B (torch.compile)      │
                         │           ~4GB VRAM, batch=32               │
                         └─────────────────────────────────────────────┘
```

**Why a single GPU thread?** NeMo models hold CUDA state that is not thread-safe. A dedicated thread avoids CUDA context contention, prevents cross-thread tensor GC segfaults, and gives predictable latency. The async server handles I/O concurrency while the GPU thread handles compute.

**Why dynamic batching?** Individual requests arrive at random times. Without batching, each request runs alone on the GPU — wasting parallel compute capacity. The batch engine collects requests and flushes them as one GPU batch, which is the primary throughput lever (1 RPS serial vs 49.6 RPS batched on L4).

## Quick Start

### Docker Compose (simplest)

```bash
# From repo root:
docker compose -f examples/asr/serving/docker-compose.yaml up -d

# Check health (model loading takes ~60-90s):
curl http://localhost:8000/health

# Transcribe a file:
curl -F file=@audio.wav http://localhost:8000/v1/transcribe

# Transcribe multiple files:
curl -F files=@a.wav -F files=@b.wav http://localhost:8000/v1/transcribe/batch
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### Docker (manual build)

```bash
# Build from repo root (~5GB image, PyTorch NGC base):
docker build -f examples/asr/serving/Dockerfile.batch -t parakeet-batch .

# Run:
docker run --gpus all -p 8000:8000 parakeet-batch
```

### Bare Metal

```bash
pip install nemo_toolkit[asr] fastapi uvicorn[standard] python-multipart pyyaml numpy

cd examples/asr/serving
python server.py --config conf/serving-batch.yaml
```

### Kubernetes (GKE)

**L4 GPU — production** (49.6 RPS, $0.50/hr spot):
```bash
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml
```

**T4 GPU — budget** (6.0 RPS, $0.35/hr spot):
```bash
kubectl apply -f k8s/deployment-t4.yaml -f k8s/service.yaml
```

Both include health probes (startup/readiness/liveness), model cache PVC, and GPU tolerations. The L4 deployment includes HPA for auto-scaling 1-4 replicas.

## API Reference

### POST /v1/transcribe

Transcribe a single audio file. Dynamically batched with concurrent requests for GPU efficiency.

```bash
curl -F file=@audio.wav http://localhost:8000/v1/transcribe
```

Response:
```json
{"text": "the quick brown fox jumps over the lazy dog", "audio_path": "/tmp/tmpXXXX.wav"}
```

With word-level timestamps:
```bash
curl -F file=@audio.wav "http://localhost:8000/v1/transcribe?timestamps=true"
```

```json
{
  "text": "the quick brown fox",
  "audio_path": "/tmp/tmpXXXX.wav",
  "word": [{"word": "the", "start": 0.0, "end": 0.12}, ...]
}
```

### POST /v1/transcribe/batch

Transcribe up to 64 files in a single request. All files are batched together for maximum GPU utilization.

```bash
curl -F files=@audio1.wav -F files=@audio2.wav -F files=@audio3.wav \
  http://localhost:8000/v1/transcribe/batch
```

```json
{"results": [{"text": "...", "audio_path": "..."}, {"text": "...", "audio_path": "..."}, ...]}
```

Max file size: 100MB (configurable via `max_upload_bytes`).

### GET /health

```json
{"status": "ok", "ready": true, "uptime_seconds": 1234.5}
```

Returns `"status": "loading"` while the model is loading (~60-90s on first start).

### GET /metrics

```json
{
  "uptime_seconds": 1234.5,
  "batch": {
    "total_requests": 5000,
    "total_batches": 200,
    "total_files": 5000,
    "rejected_requests": 0,
    "pending_requests": 3
  }
}
```

### POST /admin/config

Live-tune batch parameters without restart. Changes take effect immediately.

```bash
# View current config:
curl http://localhost:8000/admin/config

# Tune:
curl -X POST "http://localhost:8000/admin/config?max_batch_size=64&max_wait_seconds=0.01"
```

| Parameter | Range | Default | Description |
|-----------|:-----:|:-------:|-------------|
| `max_batch_size` | 1-256 | 32 | Files per GPU batch |
| `max_wait_seconds` | 0.001-5.0 | 0.002 | Max wait before flushing partial batch |
| `max_queue_depth` | 16-8192 | 4096 | Max pending requests (503 above this) |
| `gpu_poll_timeout` | 0.001-1.0 | 0.05 | GPU thread poll interval |

## Configuration

Two batch configs are included:

| Config | File | GPU | torch.compile |
|--------|------|-----|:-------------:|
| **L4 production** | `conf/serving-batch.yaml` | L4 (24GB) | Yes |
| **T4 budget** | `conf/serving-batch-t4.yaml` | T4 (16GB) | Yes* |

*T4 config has `compile: true` in the file but should be set to `false` for concurrent workloads (see Tuning Guide).

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1                    # Must be 1 — GPU models are not fork-safe

batch_model:
  name: "nvidia/parakeet-tdt-0.6b-v3"
  device: "cuda:0"
  compile: true                 # +20-30% throughput, ~60s warmup on first inference
  amp: true                     # Automatic mixed precision

batcher:
  max_batch_size: 32            # Optimal with torch.compile (see tuning guide)
  max_wait_seconds: 0.002       # Near-instant flush
  max_queue_depth: 4096         # Burst absorption buffer
  max_upload_bytes: 104857600   # 100MB max per file
```

### Tuning Guide

**`max_batch_size`**: With torch.compile, `32` outperforms `96` because smaller batches cycle through compiled CUDA kernels faster. Without torch.compile (T4), try `64-96` for better GPU utilization.

**`max_wait_seconds`**: Keep at `0.002` (2ms). The default `0.1` (100ms) wastes GPU time idle-waiting. Under load, batches fill before the timer fires anyway. Under low load, 2ms latency is imperceptible.

**`max_queue_depth`**: Keep at `4096`. In load testing, `256` caused cascading 503 failures during traffic bursts. The deep queue absorbs spikes while the GPU catches up. Memory cost is negligible (only metadata is queued, not audio data).

**`torch.compile`**: Fuses GPU kernels for 20-30% throughput gain. Enable on L4 and newer (Ada Lovelace+). **Disable on T4** for concurrent workloads — compiled inference triggers CUDA segfaults on Turing architecture under concurrent load.

**GPU optimizations** (applied automatically): `cudnn.benchmark=True` (optimal convolution algorithms), `set_float32_matmul_precision('high')` (tensor core utilization).

## Scaling

### Horizontal Scaling

Each server instance uses one GPU. Scale by adding replicas — requests are stateless.

The included `k8s/hpa.yaml` auto-scales 1-4 replicas based on CPU utilization:

| Replicas | L4 RPS | T4 RPS | L4 Daily Capacity |
|:--------:|:------:|:------:|:-----------------:|
| 1 | 49.6 | 6.0 | 4.3M |
| 2 | ~99 | ~12 | 8.6M |
| 4 | ~198 | ~24 | 17.1M |

### GPU Selection

| GPU | VRAM | RPS | torch.compile | Spot $/hr | $/1K req | Best For |
|-----|:----:|:---:|:-------------:|:---------:|:--------:|----------|
| **L4** | 24GB | 49.6 | Yes | $0.50 | $0.0028 | Production |
| **T4** | 16GB | 6.0 | No | $0.35 | $0.0162 | Dev/staging, low traffic |
| A100 | 80GB | ~100+ | Yes | $1.50+ | ~$0.004 | Maximum throughput |

### Deployment Patterns

**Single-region production**: 2-4x L4 behind a load balancer. HPA scales based on queue depth or CPU. Model cache PVC avoids re-download on pod restart.

**Multi-region**: Deploy identical stacks per region. The server is stateless — no cross-replica coordination needed.

**Cost-optimized**: Use T4 spot instances for dev/staging. Switch to L4 for production. The same image and config work on both — only `compile: true/false` differs.

**Burst handling**: The 4096-deep queue absorbs traffic spikes. For sustained high load, scale replicas rather than increasing queue depth.

## Benchmarking

### Quick Validation

```bash
pip install aiohttp numpy

# Throughput test (32 concurrent, 200 requests, synthetic audio):
python stress_test.py --server http://localhost:8000 --mode batch

# With real speech audio (requires: apt install espeak-ng):
python stress_test.py --server http://localhost:8000 --mode batch --real-audio --concurrency 16

# Transcription quality (WER measurement against known text):
python stress_test.py --server http://localhost:8000 --mode quality
```

### Full Performance Characterization

Warmup, concurrency sweep (1-128), audio duration sweep (1s-60s), and sustained load test. Outputs a structured JSON report for capacity planning.

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
  server.py                # FastAPI server, REST endpoints, health/metrics
  gpu_worker.py            # Dedicated GPU inference thread
  batch_engine.py          # Dynamic batching engine
  stress_test.py           # Stress test + quality test client
  benchmark.py             # Industrial performance benchmark
  requirements-serving.txt # Python dependencies (NeMo assumed pre-installed)

  conf/
    serving-batch.yaml     # L4 production config
    serving-batch-t4.yaml  # T4 budget config

  Dockerfile.batch         # Lightweight batch image (~5GB, PyTorch NGC base)
  docker-compose.yaml      # Single-command local deploy

  k8s/
    deployment.yaml        # L4 deployment + model cache PVC
    deployment-t4.yaml     # T4 budget deployment
    service.yaml           # ClusterIP service
    hpa.yaml               # Horizontal Pod Autoscaler (1-4 replicas)
```

## Thread-Safety Fixes

This server addresses three thread-safety bugs in NeMo's transcription path:

1. **freeze/unfreeze race** (#15771): Concurrent `transcribe()` calls crash on `_frozen_grad_map`. Fixed by removing redundant freeze/unfreeze (covered by `@torch.inference_mode()`).

2. **Cross-thread CUDA tensor GC**: NeMo's RNNT decoder holds CUDA pinned-memory tensors. When GC'd on the async thread instead of the GPU thread, `CachingHostAllocator` segfaults (signal 139). Fixed by serializing results to plain Python dicts on the GPU thread.

3. **Residual generator leak**: NeMo's internal generators can outlive the function return. Fixed by `gc.collect()` on the GPU thread after each dispatch.

## Known Limitations

- **Single GPU per instance**: Scale horizontally for more throughput.
- **T4 concurrency cap at 16**: Higher concurrency triggers CUDA host allocator segfaults on Turing architecture. This is a NeMo-internal issue.
- **torch.compile warmup**: First inference takes ~60s for kernel compilation. Health probe `startupProbe` in K8s handles this gracefully.
- **workers=1 required**: GPU models are not fork-safe. Do not set `workers > 1`.
