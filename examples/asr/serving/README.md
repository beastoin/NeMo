# NeMo ASR Serving

High-performance ASR inference server for NVIDIA NeMo models. Two modes on a single GPU:

- **Batch** (REST API) — Parakeet TDT 0.6B for offline transcription, dynamically batched, 343x realtime on L4
- **Streaming** (WebSocket) — Nemotron 3.5 ASR Streaming 0.6B for real-time transcription, 56 concurrent streams on L4

Deploy either mode independently or both together. Production-tested with [Omi](https://github.com/BasedHardware/omi), an open-source AI wearable processing thousands of concurrent audio streams.

## Performance

### Batch Mode (Parakeet TDT 0.6B)

| GPU | Sustained RPS | Realtime | RTF | torch.compile | Failures | Daily Capacity |
|-----|:------------:|:--------:|:---:|:-------------:|:--------:|:--------------:|
| **L4** (24GB) | **43** | **343x** | **0.003** | Yes | 0 | 3.7M req |
| **T4** (16GB) | **6.9** | **55x** | **0.018** | Yes | 0 | 596K req |

### Streaming Mode (Nemotron 3.5 ASR Streaming 0.6B)

| GPU | Max Concurrent | WER (LibriSpeech) | p99 Latency | Throughput | Failures |
|-----|:--------------:|:-----------------:|:-----------:|:----------:|:--------:|
| **L4** (24GB) | **c=56** | **6.9%** | **2.8s** | **129 sess/min** | 0 |

**What the numbers mean:**
- **343x realtime** (RTF 0.003) means one L4 GPU transcribes 343 seconds of audio per wall-clock second
- **c=56** means 56 users can stream audio simultaneously to a single GPU with no quality degradation
- **WER 6.9%** on LibriSpeech test-clean — production-grade accuracy under full concurrent load

All benchmarks: real speech audio, L4 GPU, NGC 26.02 container (PyTorch 2.6 + CUDA 12.8), zero failures.

### Feature Matrix

| Feature | Batch | Streaming | Notes |
|---------|:-----:|:---------:|-------|
| Transcription | **Yes** | **Yes** | |
| Word timestamps | **Yes** | No | `?timestamps=true` on REST API |
| Partial transcripts | N/A | **Yes** | Per-chunk partial + final |
| Punctuation & capitalization | **Yes** | **Yes** | Built into model output |
| Multilingual | No | **Yes** | Prompt-conditioned language selection |
| ITN (inverse text normalization) | No | No | Supported by models, not yet exposed |
| Diarization | No | No | Requires separate model (e.g. pyannote) |
| Confidence scores | No | No | |
| VAD | No | No | |

### Performance at a Glance (L4 GPU)

| Metric | Batch | Streaming |
|--------|:-----:|:---------:|
| WER (LibriSpeech) | ~3-5% | 6.9% |
| RTFx | 343x | N/A (realtime-paced) |
| RTF | 0.003 | N/A |
| Max throughput | 43 RPS | c=56 (129 sess/min) |
| p99 latency | ~2.7s | 2.8s |
| VRAM | ~4 GB | ~5.7 GB |
| torch.compile | Yes (+20-30%) | No benefit |
| Zero failures | Yes | Yes |

## Architecture

```
                          ┌────────────────────────────────────────────────┐
                          │               FastAPI Server                  │
                          │          (async, single process)              │
                          └──────┬────────────────────────┬──────────────┘
                                 │                        │
                     ┌───────────┘                        └───────────┐
                     │                                                │
          POST /v1/transcribe                              WS /v1/stream
          POST /v1/transcribe/batch                     (per-client WebSocket)
                     │                                                │
                     ▼                                                ▼
          ┌─────────────────────┐                      ┌─────────────────────────┐
          │  Batch Engine       │                      │  Stream Engine          │
          │                     │                      │                         │
          │  Collects requests  │                      │  Manages per-client     │
          │  into GPU batches.  │                      │  streaming sessions.    │
          │  Flushes at:        │                      │  Routes PCM16 chunks    │
          │  • batch_size=32    │                      │  to GPU worker.         │
          │  • wait=2ms         │                      │  Backpressure: 503      │
          │  Backpressure: 503  │                      │  at 128 streams.        │
          └────────┬────────────┘                      └────────────┬────────────┘
                   │                                                │
                   └──────────────┐          ┌──────────────────────┘
                                  ▼          ▼
                          ┌────────────────────────────────────────────────┐
                          │              GPU Worker Thread                │
                          │                                               │
                          │  Single dedicated thread for all inference.   │
                          │  • torch.inference_mode()                     │
                          │  • gc.collect(0) per batch, full gc/50        │
                          │  • All CUDA state on one thread               │
                          └──────┬────────────────────────┬──────────────┘
                                 │                        │
                                 ▼                        ▼
                   ┌──────────────────────┐  ┌───────────────────────────┐
                   │   Parakeet TDT 0.6B  │  │  Nemotron Streaming 0.6B │
                   │   (torch.compile)    │  │  (cache-aware RNNT)       │
                   │   ~4GB VRAM          │  │  ~5.7GB VRAM              │
                   └──────────────────────┘  └───────────────────────────┘
```

**Why a single GPU thread?** NeMo models hold CUDA state that is not thread-safe. A dedicated thread avoids CUDA context contention, prevents cross-thread tensor GC segfaults, and gives predictable latency. The async server handles I/O concurrency while the GPU thread handles compute.

**Why dynamic batching?** Individual requests arrive at random times. Without batching, each request runs alone on the GPU. The batch engine collects requests and flushes them as one GPU batch (1 RPS serial vs 43 RPS batched on L4).

**Why chunk accumulation for streaming?** Clients send small audio chunks (e.g. 40ms) over WebSocket. The streaming model expects 320ms chunks matching its training chunk size. The GPU worker accumulates small chunks per stream and processes them when 320ms of audio is buffered, which is required for multilingual models to produce non-blank output.

## Quick Start

### Streaming Mode (recommended for real-time products)

**Docker:**
```bash
# Build from repo root:
docker build -f examples/asr/serving/Dockerfile.stream -t nemotron-stream .

# Run (model downloads on first start, ~60-90s):
docker run --gpus all -p 8000:8000 nemotron-stream

# Test with a WebSocket client (see "Streaming Client Example" below)
```

**Bare metal:**
```bash
pip install nemo_toolkit[asr] fastapi uvicorn[standard] python-multipart pyyaml numpy websockets

cd examples/asr/serving
python server.py --config conf/serving-stream.yaml
```

### Batch Mode (for offline transcription)

**Docker Compose (simplest):**
```bash
# From repo root:
docker compose -f examples/asr/serving/docker-compose.yaml up -d

# Check health (model loading takes ~60-90s):
curl http://localhost:8000/health

# Transcribe a file:
curl -F file=@audio.wav http://localhost:8000/v1/transcribe
```

**Docker (manual build):**
```bash
# Build from repo root (~5GB image, PyTorch NGC base):
docker build -f examples/asr/serving/Dockerfile.batch -t parakeet-batch .

# Run:
docker run --gpus all -p 8000:8000 parakeet-batch
```

**Bare metal:**
```bash
pip install nemo_toolkit[asr] fastapi uvicorn[standard] python-multipart pyyaml numpy

cd examples/asr/serving
python server.py --config conf/serving-batch.yaml
```

### Both Models (batch + streaming on one GPU)

```bash
# Uses ~10GB VRAM total (4GB batch + 5.7GB streaming)
cd examples/asr/serving
python server.py --config conf/serving.yaml
```

### Kubernetes (GKE)

**Streaming (L4 GPU):**
```bash
kubectl apply -f k8s/deployment-stream.yaml -f k8s/service.yaml
```

**Batch (L4 GPU, with HPA):**
```bash
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml
```

**Batch (T4 GPU, budget):**
```bash
kubectl apply -f k8s/deployment-t4.yaml -f k8s/service.yaml
```

All deployments include health probes (startup/readiness/liveness), model cache PVC, and GPU tolerations. The L4 batch deployment includes HPA for auto-scaling 1-4 replicas.

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for Docker deployments.

## API Reference

### WebSocket /v1/stream

Real-time streaming transcription. Each WebSocket connection is one audio stream.

**Protocol:**

1. Client connects to `ws://host:8000/v1/stream`
2. Server sends: `{"stream_id": "...", "status": "opened"}`
3. Client sends raw **PCM16 audio chunks** (binary WebSocket frames, 16kHz mono)
4. Server sends partial transcripts after each chunk:
   ```json
   {
     "stream_id": "abc-123",
     "partial_transcript": "the quick brown",
     "final_transcript": "the quick",
     "is_final": true
   }
   ```
5. Client sends text frame `{"action": "close"}` to end the stream
6. Server sends final transcript and closes:
   ```json
   {
     "stream_id": "abc-123",
     "final_text": "the quick brown fox jumps over the lazy dog",
     "status": "closed"
   }
   ```

**Audio format:** Raw PCM16 (signed 16-bit little-endian), 16kHz sample rate, mono. No WAV headers. Clients can send any chunk size — the server accumulates to the model's native chunk size (320ms / 5120 samples) internally.

**Recommended chunk size:** 40-160ms (640-2560 bytes). Smaller chunks give lower perceived latency. The server buffers internally regardless of client chunk size.

**Streaming client example (Python):**

```python
import asyncio
import numpy as np
import websockets

async def stream_audio(audio_path, server="ws://localhost:8000"):
    """Stream a WAV file and print live transcription."""
    import soundfile as sf
    audio, sr = sf.read(audio_path, dtype='int16')
    if sr != 16000:
        import librosa
        audio = (librosa.resample(audio.astype(np.float32) / 32768.0,
                                  orig_sr=sr, target_sr=16000) * 32768).astype(np.int16)

    chunk_samples = 640  # 40ms chunks
    async with websockets.connect(f"{server}/v1/stream") as ws:
        # Receive session confirmation
        session = await ws.recv()
        print(f"Session: {session}")

        # Stream audio in realtime-paced chunks
        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i:i + chunk_samples].tobytes()
            await ws.send(chunk)
            response = await ws.recv()
            import json
            result = json.loads(response)
            if result.get("partial_transcript"):
                print(f"  partial: {result['partial_transcript']}")
            if result.get("is_final"):
                print(f"  >>> {result['final_transcript']}")
            await asyncio.sleep(chunk_samples / 16000)  # realtime pacing

        # Close stream and get final transcript
        await ws.send('{"action": "close"}')
        final = await ws.recv()
        print(f"Final: {json.loads(final).get('final_text', '')}")

asyncio.run(stream_audio("audio.wav"))
```

**Streaming client example (JavaScript / Browser):**

```javascript
const ws = new WebSocket('ws://localhost:8000/v1/stream');

ws.onopen = () => console.log('Connected');
ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  if (result.status === 'opened') {
    // Start sending audio from getUserMedia
    startAudioCapture(ws);
  } else if (result.partial_transcript) {
    document.getElementById('transcript').textContent = result.partial_transcript;
  }
};

async function startAudioCapture(ws) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx = new AudioContext({ sampleRate: 16000 });
  const source = ctx.createMediaStreamSource(stream);
  const processor = ctx.createScriptProcessor(1024, 1, 1);

  processor.onaudioprocess = (e) => {
    const float32 = e.inputBuffer.getChannelData(0);
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
    }
    ws.send(int16.buffer);
  };

  source.connect(processor);
  processor.connect(ctx.destination);
}
```

### POST /v1/transcribe

Transcribe a single audio file (batch mode). Dynamically batched with concurrent requests.

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

Returns `"status": "loading"` while models are loading (~60-90s on first start).

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
  },
  "stream": {
    "total_streams_opened": 150,
    "total_streams_closed": 148,
    "total_chunks_processed": 45000,
    "active_streams": 2
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

Four configs are included for different deployment scenarios:

| Config | File | Mode | Model | GPU |
|--------|------|------|-------|-----|
| **Streaming production** | `conf/serving-stream.yaml` | Stream only | Nemotron 3.5 ASR Streaming 0.6B | L4 |
| **Batch production** | `conf/serving-batch.yaml` | Batch only | Parakeet TDT 0.6B | L4 |
| **Batch budget** | `conf/serving-batch-t4.yaml` | Batch only | Parakeet TDT 0.6B | T4 |
| **Combined** | `conf/serving.yaml` | Both | Both models | L4 |

### Streaming Configuration

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1                    # Must be 1 — GPU models are not fork-safe

stream_model:
  name: "nvidia/nemotron-3.5-asr-streaming-0.6b"
  device: "cuda:0"
  compile: false                # RNNT streaming pipeline doesn't benefit from torch.compile
  amp: true                     # FP16 inference
  latency_mode: "480ms"         # Quality/latency tradeoff (see below)
  source_language: "English"    # Prompt language for multilingual model

stream:
  max_concurrent_streams: 128   # Max simultaneous WebSocket sessions
  chunk_duration_ms: 160        # Client-side chunk size hint
  sample_rate: 16000            # Expected audio sample rate
  max_stream_duration: 1800     # Auto-close after 30 minutes
  max_chunk_bytes: 524288       # 512KB max per WebSocket frame
```

### Batch Configuration

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 1                    # Must be 1 — GPU models are not fork-safe

batch_model:
  name: "nvidia/parakeet-tdt-0.6b-v3"
  device: "cuda:0"
  compile: true                 # +20-30% throughput, ~60s warmup
  amp: true                     # Automatic mixed precision
  cuda_graphs: true             # RNNT decoder CUDA graphs (disable on T4)

batcher:
  max_batch_size: 32            # Optimal with torch.compile
  max_wait_seconds: 0.002       # Near-instant flush
  max_queue_depth: 4096         # Burst absorption buffer
  max_upload_bytes: 104857600   # 100MB max per file
```

### Tuning Guide

**`latency_mode`** (streaming): Controls the quality-latency tradeoff via attention context size. Lower modes give faster partial results but worse accuracy.

| Mode | Right Context | Use Case |
|------|:------------:|----------|
| `"80ms"` | 0 | Ultra-low latency, lower accuracy |
| `"160ms"` | 1 | Low latency |
| `"480ms"` | 6 | **Recommended** — best quality/latency balance |
| `"1040ms"` | 13 | Highest accuracy, higher latency |

**`source_language`** (streaming): The Nemotron 3.5 ASR Streaming model supports multiple languages. Set this to match the expected input language. The model uses a 128-dim one-hot prompt vector to condition on language. The prompt is set once on the first audio frame of each stream.

**`max_batch_size`** (batch): With torch.compile, `32` outperforms `96` because smaller batches cycle through compiled CUDA kernels faster. Without torch.compile (T4), try `64-96` for better GPU utilization.

**`max_wait_seconds`** (batch): Keep at `0.002` (2ms). The default `0.1` wastes GPU time idle-waiting. Under load, batches fill before the timer fires anyway.

**`max_queue_depth`** (batch): Keep at `4096`. In load testing, `256` caused cascading 503 failures during traffic bursts. The deep queue absorbs spikes while the GPU catches up.

**`torch.compile`** (batch): Fuses GPU kernels for 20-30% throughput gain. Works on L4 and T4. The streaming model does not benefit from torch.compile due to cache-aware state management.

**`cuda_graphs`** (batch): NeMo's RNNT decoder uses CUDA Graph conditional nodes. These work on Ada Lovelace+ (L4, A100) but cause `cudaErrorIllegalAddress` on Turing (T4, compute capability 7.5). **Set `cuda_graphs: false` for T4 deployments.**

**`prefetch`** (batch): Pre-reads audio files on a background thread and passes tensors directly to `model.transcribe()`, bypassing NeMo's DataLoader overhead. Gives **+7% throughput and -24% latency** on T4.

**GPU optimizations** (applied automatically): `cudnn.benchmark=True` (optimal convolution algorithms), `set_float32_matmul_precision('high')` (tensor core utilization).

## Scaling

### Horizontal Scaling

Each server instance uses one GPU. Scale by adding replicas — requests are stateless (streaming sessions are per-connection, no cross-replica state).

The included `k8s/hpa.yaml` auto-scales 1-4 batch replicas based on CPU utilization:

| Replicas | Batch RPS (L4) | Concurrent Streams (L4) |
|:--------:|:--------------:|:-----------------------:|
| 1 | 43 | 56 |
| 2 | ~86 | ~112 |
| 4 | ~172 | ~224 |

### GPU Selection

| GPU | VRAM | Batch RPS | Stream Max c | torch.compile | CUDA Graphs | Best For |
|-----|:----:|:---------:|:------------:|:-------------:|:-----------:|----------|
| **L4** | 24GB | 43 | 56 | Yes | Yes | Production |
| **T4** | 16GB | 6.9 | — | Yes | No | Dev/staging |
| A100 | 80GB | ~100+ | ~100+ | Yes | Yes | Maximum throughput |

### Deployment Patterns

**Real-time product (Omi-like):** Deploy streaming-only (`Dockerfile.stream` + `serving-stream.yaml`) on L4 GPUs. One L4 handles 56 concurrent users. Scale horizontally with a WebSocket-aware load balancer (e.g. Nginx with `proxy_pass` and `Upgrade` headers, or a GKE ingress with WebSocket support).

**Offline transcription:** Deploy batch-only (`Dockerfile.batch` + `serving-batch.yaml`) on L4 GPUs. HPA scales based on queue depth or CPU. Model cache PVC avoids re-download on pod restart.

**Hybrid:** Deploy both models on a single L4 (`serving.yaml`). Uses ~10GB of 24GB VRAM. Useful when you need both real-time and offline transcription and want to minimize infrastructure.

**Cost-optimized:** Use T4 for dev/staging (batch only). Switch to L4 for production. Same image and code — T4 needs `cuda_graphs: false`.

## Benchmarking

### Streaming Benchmark

```bash
pip install websockets aiohttp numpy soundfile

# Quick concurrency test (synthetic audio):
python stream_benchmark.py --server ws://localhost:8000 --concurrency 1,4,16,32

# Sustained load test (5 minutes):
python stream_benchmark.py --server ws://localhost:8000 --sustained-minutes 5

# Full concurrency sweep:
python stream_benchmark.py --server ws://localhost:8000 --concurrency 1,2,4,8,16,32,48,56,64
```

### Batch Benchmark

```bash
pip install aiohttp numpy

# Throughput test (32 concurrent, 200 requests):
python stress_test.py --server http://localhost:8000 --mode batch

# With real speech audio (requires: apt install espeak-ng):
python stress_test.py --server http://localhost:8000 --mode batch --real-audio --concurrency 16

# Transcription quality (WER):
python stress_test.py --server http://localhost:8000 --mode quality

# Mixed batch + streaming:
python stress_test.py --server http://localhost:8000 --mode mixed
```

### Full Performance Characterization

Warmup, concurrency sweep (1-128), audio duration sweep (1s-60s), and sustained load test:

```bash
python benchmark.py --server http://localhost:8000 --sustained-minutes 5 --output report.json
```

### In-Cluster Benchmark (K8s)

```bash
kubectl apply -f k8s/benchmark-configs.yaml
kubectl logs -f job/benchmark-configs
```

## Dockerfiles

Three Dockerfiles for different deployment scenarios:

| File | Mode | Base Image | Size | VRAM |
|------|------|------------|------|------|
| `Dockerfile.stream` | Streaming only | `nvcr.io/nvidia/nemo:26.02` | ~20GB | ~5.7GB |
| `Dockerfile.batch` | Batch only | `nvcr.io/nvidia/pytorch:24.12-py3` | ~9GB | ~4GB |
| `Dockerfile` | Both | `nvcr.io/nvidia/nemo:26.06` | ~20GB | ~10GB |

The streaming Dockerfile **must** use the full NeMo NGC image because the cache-aware RNNT pipeline depends on NeMo's inference wrappers. The batch Dockerfile uses a lighter PyTorch base with `nemo_toolkit[asr]` installed via pip.

**Build from repo root** (all Dockerfiles expect repo root as build context):

```bash
docker build -f examples/asr/serving/Dockerfile.stream -t nemotron-stream .
docker build -f examples/asr/serving/Dockerfile.batch  -t parakeet-batch .
docker build -f examples/asr/serving/Dockerfile        -t nemo-asr-serving .
```

## Files

```
examples/asr/serving/
  server.py                # FastAPI server — REST + WebSocket endpoints, health/metrics
  gpu_worker.py            # Dedicated GPU inference thread — batch + streaming dispatch
  batch_engine.py          # Dynamic batching engine for offline transcription
  stream_engine.py         # Streaming session manager — lifecycle + chunk routing
  stress_test.py           # Batch/stream/mixed/quality test client
  stream_benchmark.py      # Streaming-specific benchmark (concurrency sweep, sustained load)
  benchmark.py             # Industrial batch benchmark (sweep, duration, cost analysis)
  benchmark-report.html    # L4/T4 batch benchmark report
  requirements-serving.txt # Python dependencies (NeMo assumed pre-installed)

  conf/
    serving.yaml            # Combined batch + streaming config
    serving-stream.yaml     # Streaming-only config (L4)
    serving-batch.yaml      # Batch-only config (L4)
    serving-batch-t4.yaml   # Batch-only config (T4)

  Dockerfile               # Combined batch + streaming image
  Dockerfile.stream        # Streaming-only image (NeMo NGC base)
  Dockerfile.batch         # Batch-only image (PyTorch NGC base)
  docker-compose.yaml      # Single-command batch deploy

  k8s/
    deployment.yaml         # L4 batch deployment + model cache PVC
    deployment-stream.yaml  # L4 streaming deployment + model cache PVC
    deployment-t4.yaml      # T4 batch deployment
    deployment-t4-bench.yaml # T4 benchmark deployment
    service.yaml            # ClusterIP service
    hpa.yaml                # Horizontal Pod Autoscaler (1-4 replicas)
    benchmark-job.yaml      # In-cluster benchmark job
    benchmark-configs.yaml  # Benchmark configuration sweep job
```

## Thread-Safety Fixes

This server addresses five issues in NeMo's inference path that cause crashes under concurrent load:

1. **Streaming prompt_vectors dropped** — NeMo's `execute_step()` in the cache-aware RNNT wrapper silently drops `prompt_vectors` when processing streaming frames. Multilingual models (like Nemotron 3.5 ASR Streaming) require these vectors for language conditioning. Without them, the model produces blank or garbled output. Fixed by projecting `prompt_vectors` through the model's `prompt_kernel` MLP in the streaming path.

2. **Mel spectrogram fragmentation** — Small WebSocket chunks (e.g. 40ms) produce fragmented mel spectrograms when processed individually. The cache-aware RNNT pipeline expects chunks matching its training chunk size (320ms / 5120 samples at 16kHz). Fixed by accumulating client audio chunks per stream and only creating frames when the native chunk size is reached. The chunk size is read from `pipeline.chunk_size_in_secs * pipeline.sample_rate`.

3. **CUDA Graph crash on Turing GPUs** — NeMo's RNNT decoder uses CUDA Graph conditional nodes compiled via NVRTC. These cause `cudaErrorIllegalAddress` on T4 (Turing, compute capability 7.5), poisoning the CUDA context. Fixed by adding a `cuda_graphs: false` config option. L4/A100+ are unaffected.

4. **freeze/unfreeze race** (#15771) — Concurrent `transcribe()` calls crash on `_frozen_grad_map`. Fixed by removing redundant freeze/unfreeze (covered by `@torch.inference_mode()`).

5. **Cross-thread CUDA tensor GC** — NeMo's RNNT decoder holds CUDA pinned-memory tensors. When garbage-collected on the async event-loop thread instead of the GPU thread, `CachingHostAllocator` segfaults (signal 139). Fixed by: (a) serializing results to plain Python dicts on the GPU thread before returning, (b) disabling automatic GC (`gc.disable()`) and running `gc.collect(0)` on the GPU thread after every batch.

## Case Study: Omi (AI Wearable)

[Omi](https://github.com/BasedHardware/omi) is an open-source AI wearable that captures conversations and transcribes them in real time. Their backend processes thousands of audio segments from concurrent users.

**Problem:** NeMo's `model.transcribe()` crashed under concurrent requests — CUDA segfaults from cross-thread tensor GC, freeze/unfreeze races, and CUDA Graph incompatibilities. The service was unstable above 2-3 concurrent users.

**Solution:** Deployed this serving stack on GKE with the NGC 26.02 container and NeMo thread-safety patches.

**Batch results** (L4 GPU, torch.compile + CUDA Graphs):

| Metric | Value |
|--------|-------|
| Sustained RPS | **35.92** |
| Realtime | **286x** (RTF 0.003) |
| Total requests tested | 5,134 |
| Failures | **0** |

**Streaming results** (L4 GPU, Nemotron 3.5 ASR Streaming 0.6B):

| Metric | Value |
|--------|-------|
| Max concurrent streams | **c=56** |
| WER (LibriSpeech) | **6.9%** |
| p99 latency | **2.8s** |
| Throughput | **129 sessions/min** |
| Failures | **0** |

**Key deployment choices:**
- NGC 26.02 base image (PyTorch 2.6 + CUDA 12.8) — PyTorch 2.12+ has a CachingHostAllocator bug
- NeMo fork installed with `pip install --no-deps --force-reinstall` to preserve NGC's CUDA-aligned PyTorch
- All models loaded on the GPU worker thread (not the main thread) to prevent cross-thread CUDA tensor ownership crashes
- `gc.disable()` at module level + `gc.collect(0)` per batch on GPU thread
- Client audio chunks (40ms) accumulated to model's native 320ms before inference

## Known Limitations

- **Single GPU per instance.** Scale horizontally for more throughput.
- **T4 streaming not benchmarked.** T4's 16GB VRAM is tight for the streaming model (~5.7GB) plus batch model (~4GB). Streaming-only on T4 should work but is not production-tested.
- **T4 requires `cuda_graphs: false`** for batch mode. CUDA Graph conditional nodes in the RNNT decoder crash on Turing architecture. Throughput impact is minor (~5%).
- **torch.compile warmup.** First batch inference takes ~60s for kernel compilation. K8s `startupProbe` handles this gracefully.
- **workers=1 required.** GPU models are not fork-safe. Do not set `workers > 1`.
- **Stream duration limit.** Streams auto-close after 30 minutes (`max_stream_duration`). Clients should reconnect for longer sessions.
- **No TLS/auth built in.** Deploy behind a reverse proxy (Nginx, Envoy, GKE Ingress) for TLS termination and authentication.
