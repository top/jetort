# JetOrt

<p align="center">
  <img src="https://raw.githubusercontent.com/top/jetort/main/assets/banner.png" alt="JetOrt Banner" width="700" onerror="this.style.display='none'"/>
</p>

<p align="center">
  <strong>Ultra-lightweight, zero-overhead one-shot tensor inference engine and server, powered by Rust.</strong>
</p>

<p align="center">
  <a href="https://github.com/top/jetort/actions"><img src="https://img.shields.io/github/actions/workflow/status/top/jetort/ci.yml?branch=main&label=CI&logo=github" alt="CI Status"/></a>
  <a href="https://crates.io/crates/jetort"><img src="https://img.shields.io/badge/rust-1.80%2B-orange.svg?logo=rust" alt="Rust Version"/></a>
  <a href="https://github.com/top/jetort/releases"><img src="https://img.shields.io/github/v/release/top/jetort?color=blue&logo=github" alt="GitHub Release"/></a>
  <a href="https://github.com/top/jetort/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"/></a>
  <a href="https://platform.openai.com/docs/api-reference"><img src="https://img.shields.io/badge/OpenAI%20API-Compatible-412991.svg" alt="OpenAI Compatible"/></a>
  <a href="https://kserve.github.io/website/latest/modelserving/data_plane/v2_protocol/"><img src="https://img.shields.io/badge/KServe%20v2-CNCF%20Standard-blue.svg" alt="KServe v2"/></a>
</p>

---

## Overview

Modern AI deployments typically pair a generative Large Language Model (LLM) with auxiliary perception models for embeddings, cross-encoder reranking, speech recognition (ASR), and computer vision.

JetOrt runs single-pass ONNX models: embeddings, rerank, CTC ASR, classifiers, vision encoders, and raw tensors. Autoregressive graphs (paired KV cache, or a genai config that describes a decoder loop) are detected and refused at load. A `genai_config.json` file by itself does not make a model autoregressive. Chat stays on a decoder runtime outside this process. Auxiliary models are otherwise stuck in Python stacks (PyTorch, Transformers, FastAPI), which introduces severe overhead:
* **Hundreds of megabytes to gigabytes of VRAM/RAM overhead** just for Python runtime and PyTorch dependencies.
* **Global Interpreter Lock (GIL) contention** bottlenecking concurrent I/O.
* **Complex runtime environments** (CUDA wheels, Python virtualenvs, C ABI incompatibilities).

**JetOrt** solves this problem. It is an ultra-fast, universal non-autoregressive inference server compiled into a single self-contained binary. Designed for edge hardware (NVIDIA Jetson Orin / Nano) and desktop/server systems, JetOrt delivers sub-millisecond execution times with resident memory usage measured in tens of megabytes.

```
+-----------------------------------------------------------------------------------+
|                                  HTTP Clients                                     |
|           (OpenAI SDK / LangChain / LlamaIndex / cURL / Web UIs)                  |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        JetOrt High-Performance Server                             |
|                                                                                   |
|  +---------------------------------------+  +----------------------------------+  |
|  |  OpenAI Protocol Layer (/v1)          |  | CNCF KServe v2 Protocol Layer    |  |
|  |  * /v1/models                         |  |  * /v2/health/live               |  |
|  |  * /v1/embeddings                     |  |  * /v2/health/ready              |  |
|  |  * /v1/rerank                         |  |  * /v2/models/:name              |  |
|  |  * /v1/systemone                      |  |  * /v2/models/:name/infer        |  |
|  |  * /v1/audio/transcriptions           |  |  * /v2/repository/models/...     |  |
|  +---------------------------------------+  +----------------------------------+  |
|                                         |                                         |
|                 +-----------------------v-----------------------+                 |
|                 | Tokenizer & Vocab Auto-Discovery              |                 |
|                 | (tokenizer.json, vocab.json, vocab.txt)       |                 |
|                 +-----------------------+-----------------------+                 |
|                                         |                                         |
|                 +-----------------------v-----------------------+                 |
|                 | Asynchronous Offload & Dynamic Tensor Prep    |                 |
|                 | (Auto-padding, Matryoshka, Pooling, Resample) |                 |
|                 +-----------------------+-----------------------+                 |
+-----------------------------------------|-----------------------------------------+
                                          v
+-----------------------------------------------------------------------------------+
|                             Hardware Execution Tier                               |
|  Single-pass ONNX Runtime. Autoregressive graphs are refused at load.           |
|  Provider chain depends on the build target (see scripts/build.sh)               |
+-----------------------------------------------------------------------------------+
```

---

## Key Features

* **High-Fidelity OpenAI API Parity**:
  - `/v1/models`: OpenAI-compatible enumeration plus llama.cpp-style runtime `meta`; calibrated decision models are also listed in TypeSafe's `models` shape.
  - `/v1/embeddings`: Full usage token metrics (`prompt_tokens`, `total_tokens`), Matryoshka dimensionality reduction (`dimensions`), and L2 normalization.
  - `/v1/audio/transcriptions`: Supports standard `multipart/form-data` and `application/json`, output formats (`json`, `verbose_json`, `text`), and token usage reporting.
* **llama.cpp-Style Performance Telemetry**:
  - Detailed model architecture and weight loading logs (`llama_load_model: ...`).
  - Precise per-request latency & throughput metrics:
    - `prompt eval time = ... ms / ... tokens (... tokens per second)`
    - `eval time = ... ms / ... tokens (... tokens per second)`
    - `total time = ... ms / ... tokens (... tokens per second)`
* **Drop-in Cohere / BGE Reranker Support**:
  - `/v1/rerank` and `/rerank`: High-throughput cross-encoder scoring compatible with Cohere SDK and LiteLLM gateways.
* **TypeSafe System One API Compatibility**:
  - `/v1/systemone`: Choice, Score, and Noul questions backed by calibrated GLiClass/OpenJev ONNX bundles.
  - Automatically detects `config.json`, `tokenizer.json`, and `calibrator.json`; no JetOrt-specific model type setting is required.
* **KServe v2 health and metadata**:
  - `/v2/health/live`, `/v2/health/ready`, and `/v2/models/{name}` metadata. `/v2/models/{name}/infer` is a placeholder that returns HTTP 501 `not_implemented` and is not listed on model cards.
* **Universal Model-Agnostic Tokenizer & Vocab Auto-Discovery**:
  - Automatically locates and parses `tokenizer.json`, `vocab.json`, `vocab.txt`, `tokens.txt`, and `dict.txt` from local Hugging Face cache snapshots.
* **Per-target execution providers**:
  - `scripts/build.sh --target` selects the chain compiled into the binary: Jetson and Linux NVIDIA `TensorRT → CUDA → CPU`, macOS `CoreML → CPU`, Windows NVIDIA `TensorRT → CUDA → CPU`, Windows (including AMD) `DirectML → CPU`, Linux AMD and CPU `CPU`. The provider name in logs and `/v1/models` is the backend that actually loaded.
* **Persistent TensorRT Engine Cache**:
  - At startup, missing TensorRT engines are compiled one model at a time in isolated processes and serialized to `.cache/<model>`. Successful builds are reused across restarts.
  - If a build fails, that model immediately uses the next configured provider for the current process; requests never trigger another TensorRT build.
* **Sequence padding**:
  - `pad_seq_len` defaults to dynamic (unset) on CUDA and CPU. It is not the context length. When the loaded backend is TensorRT, sequences use the smallest of 128/256/512/1024 that fits the input and `n_ctx`.
* **Smart Model Alias & Fallback Routing**:
  - Built-in fallback routing handles clients that hardcode `whisper-1` or `text-embedding-ada-002` by automatically delegating to registered audio or embedding models.
* **Zero-Downtime Hot Reload**:
  - Atomically load, update, and unload models at runtime via HTTP API using lock-free `Arc<RwLock>` registry management.

---

## Quickstart

### 1. Installation

#### Option A: Pre-built Binaries
Download the latest release binary for your platform from [GitHub Releases](https://github.com/top/jetort/releases).

#### Option B: Build from this checkout
```bash
./scripts/build.sh --target cpu
./target/release/jetort --config config.yaml
```
On a Jetson:
```bash
./scripts/jetson.sh
```
`jetson.sh` only fills Jetson parameters and calls `build.sh`, then installs the user service and starts it. Other targets: `./scripts/build.sh --target linux-nvidia|macos|windows-nvidia|windows-other|linux-amd|cpu`. With no `--target` on a terminal, the script asks; without a terminal it exits. Default Cargo features are CPU-only.
The compiled binary will be located at `target/release/jetort`.

---

### 2. Quick Configuration

Create a `config.yaml` file. JetOrt can automatically discover models downloaded via `huggingface-cli`:

```yaml
server:
  host: "0.0.0.0"
  port: 11433
  log_level: info

models:
  # Embedding Model
  - name: "qwen3-embedding"
    alias: "text-embedding-3-small"
    hf_repo: "onnx-community/Qwen3-Embedding-0.6B-ONNX"
    hf_file: "onnx/model_int8.onnx"

  # Reranker Model
  - name: "bge-reranker-v2-m3"
    hf_repo: "onnx-community/bge-reranker-v2-m3-ONNX"
    hf_file: "onnx/model_int8.onnx"

  # Automatic Speech Recognition (SenseVoice)
  - name: "sensevoice"
    alias: "whisper-1"
    hf_repo: "OpenVoiceOS/sensevoice-small-onnx"
    hf_file: "model_int8.onnx"

  # System One Decisions (OpenJev/Verdict)
  - name: "openjev"
    alias: "jev-latest"
    hf_repo: "heman10x/rlcd-modernbert-151m"
    hf_file: "model_fp16.onnx"
```

Download the required model weights using `hf`:
```bash
# Embeddings & Reranker
hf download onnx-community/Qwen3-Embedding-0.6B-ONNX
hf download onnx-community/bge-reranker-v2-m3-ONNX

# Speech Recognition (include weights and vocabulary)
hf download OpenVoiceOS/sensevoice-small-onnx --include model_int8.onnx --include vocab.txt

# System One model, tokenizer, and calibrator
hf download heman10x/rlcd-modernbert-151m
```

---

### 3. Launch Server

```bash
./target/release/jetort --config config.yaml
```

Output:
```text
INFO jetort::engine::session: Loading ONNX model session... model="qwen3-embedding"
INFO jetort::engine::session: Tokenizer loaded successfully from json model="qwen3-embedding"
INFO jetort::engine::session: Model session loaded successfully model="qwen3-embedding" active_provider="TensorRT"
INFO jetort: Starting HTTP inference service listening on http://0.0.0.0:11433
```

---

## API Reference & Usage Examples

### Chat

`/v1/chat/completions` is not served. JetOrt loads single-pass ONNX models only. A graph with a decoder loop is refused at load and is not executed.

---

---

### 2. Text Embeddings (`/v1/embeddings` & `/embeddings`)

Compatible with the official OpenAI Python/Node.js SDKs, LangChain, LlamaIndex, and cURL.

#### cURL Example:
```bash
curl -s http://localhost:11433/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding",
    "input": "High-performance edge tensor inference with Rust."
  }' | jq
```

Response:
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0142, -0.0381, 0.0094, 0.0215, ...]
    }
  ],
  "model": "qwen3-embedding",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

#### Python OpenAI SDK Example:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11433/v1", api_key="not-needed")

response = client.embeddings.create(
    model="qwen3-embedding",
    input=["Document search query", "Second sentence for embedding"]
)
print(f"Computed {len(response.data)} embeddings. Tokens: {response.usage.total_tokens}")
```

#### Matryoshka Representation Learning (Dimension Slicing):
```bash
curl -s http://localhost:11433/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-embedding",
    "input": "Dimension truncation test",
    "dimensions": 256
  }' | jq '.data[0].embedding | length'
# Returns: 256 (automatically truncated and L2 re-normalized)
```

---

### 2. Cross-Encoder Reranker (`/v1/rerank` & `/rerank`)

Compatible with Cohere SDK, LiteLLM, Dify, and FastGPT.

#### cURL Example:
```bash
curl -s http://localhost:11433/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-reranker-v2-m3",
    "query": "What is JetOrt?",
    "documents": [
      "JetOrt is a lightweight Rust tensor inference server.",
      "The weather in Tokyo is sunny today.",
      "NVIDIA Jetson is an embedded AI platform."
    ],
    "top_n": 2
  }' | jq
```

Response:
```json
{
  "id": "rerank-0",
  "model": "bge-reranker-v2-m3",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.9842,
      "document": "JetOrt is a lightweight Rust tensor inference server."
    },
    {
      "index": 2,
      "relevance_score": 0.5120,
      "document": "NVIDIA Jetson is an embedded AI platform."
    }
  ],
  "usage": {
    "prompt_tokens": 36,
    "total_tokens": 36
  },
  "meta": {
    "api_version": { "version": "1" },
    "billed_units": { "search_units": 1 },
    "tokens": { "input_tokens": 36, "output_tokens": 0 }
  }
}
```

---

### 3. Speech Recognition (`/v1/audio/transcriptions`)

Drop-in replacement for OpenAI Whisper API. Accepts standard `multipart/form-data` audio file uploads as well as raw JSON PCM arrays.

#### cURL File Upload (Multipart Form):
```bash
curl -s http://localhost:11433/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=sensevoice" \
  -F "language=zh" | jq
```

Response (`response_format="json"`):
```json
{
  "text": "Hello, this is a real-time speech transcription test."
}
```

#### Verbose Response with Token Metrics (`response_format="verbose_json"`):
```bash
curl -s http://localhost:11433/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=sensevoice" \
  -F "response_format=verbose_json" | jq
```

Response:
```json
{
  "task": "transcribe",
  "language": "zh",
  "duration": 3.42,
  "text": "Hello, this is a real-time speech transcription test.",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 3.42,
      "text": "Hello, this is a real-time speech transcription test.",
      "tokens": [1024, 439, 1289, 88],
      "temperature": 0.0,
      "avg_logprob": 0.0,
      "compression_ratio": 1.0,
      "no_speech_prob": 0.0
    }
  ],
  "usage": {
    "type": "duration",
    "seconds": 3.42,
    "prompt_tokens": 342,
    "completion_tokens": 4,
    "total_tokens": 346
  }
}
```

#### Python SDK Example:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11433/v1", api_key="not-needed")

with open("speech.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",  # Automatically resolved to sensevoice
        file=f
    )
print(transcript.text)
```

---

### 4. System One Decisions (`/v1/systemone`)

Wire-compatible with TypeSafe's Choice, Score, and Noul request/response schema:

```bash
curl -s http://localhost:11433/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openjev",
    "state": "The customer was charged twice and will cancel tomorrow.",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments and refunds",
          "technical": "Bugs and outages"
        }
      },
      "severity": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["routine", "handle today", "urgent", "critical"]
      },
      "escalate": {
        "type": "noul",
        "instructions": "Should this be escalated immediately?"
      }
    }
  }' | jq
```

JetOrt batches all questions through the existing ONNX Runtime session, applies the bundle's `calibrator.json`, and returns the official `answers` and `usage` shape. Responses include `x-typesafe-request-id`.

The current public OpenJev model supports at most 24 Choice options and 512 tokens per question. Requests beyond those model limits return `422` instead of silently truncating. This endpoint is API-compatible with TypeSafe System One, but local OpenJev weights are not identical to TypeSafe's hosted Jev model.

---

### 5. Model Discovery (`/v1/models`)

The response preserves OpenAI's `object/data` schema and adds optional runtime metadata. The additional `models` array follows TypeSafe's model-list schema and contains only models that can serve `/v1/systemone`.

```json
{
  "object": "list",
  "data": [{
    "id": "openjev",
    "object": "model",
    "created": 1790280000,
    "owned_by": "jetort",
    "meta": {
      "engine": "onnxruntime",
      "provider": "TensorRT",
      "context_length": 8192,
      "size": 302145536,
      "inputs": ["input_ids", "attention_mask"],
      "outputs": ["logits"],
      "endpoints": ["/v2/models/openjev/infer", "/v1/systemone"]
    }
  }],
  "models": [{
    "name": "openjev",
    "description": "Local System One compatible decision model",
    "release_date": "2026-09-17"
  }]
}
```

`created` and `release_date` use the model file's filesystem creation time, falling back to its modification time. If neither timestamp is available, `created` is `0` and `release_date` is `1970-01-01`.

---

### 6. CNCF KServe v2 Data Plane (`/v2`)

Direct, high-performance tensor buffer exchange for low-level integration and Triton migration.

#### Health Probes:
```bash
curl -i http://localhost:11433/v2/health/live
curl -i http://localhost:11433/v2/health/ready
```

#### Inspect Model Input/Output Signatures:
```bash
curl -s http://localhost:11433/v2/models/qwen3-embedding | jq
```

#### Raw Tensor Forward Execution:

`POST /v2/models/{name}/infer` is reserved for the KServe v2 data-plane
implementation and currently returns HTTP 501 `not_implemented`. Use the
OpenAI-compatible endpoints above for supported inference operations.

#### Zero-Downtime Model Reload & Unload:
```bash
# Dynamically register or update a model
curl -s -X POST http://localhost:11433/v2/repository/models/new-model/load \
  -H "Content-Type: application/json" \
  -d '{
    "name": "new-model",
    "hf_repo": "BAAI/bge-small-en-v1.5",
    "hf_file": "onnx/model.onnx"
  }'

# Safely unload model and free GPU/CPU memory
curl -s -X POST http://localhost:11433/v2/repository/models/new-model/unload
```

---

## Configuration Reference (`config.yaml`)

```yaml
server:
  host: "0.0.0.0"
  port: 11433
  # Used when RUST_LOG is not set: trace, debug, info, warn, or error.
  log_level: info
  inference_threads: 4            # Worker thread pool size for blocking forward passes
  ort_dylib_path: null            # Optional manual path to libonnxruntime.so

models:
  - name: "verdict"
    alias: "jev-latest"            # llama.cpp-style single public alias
    hf_repo: "heman10x/rlcd-modernbert-151m"
    hf_file: "model_fp16.onnx"
    task: "decision"               # Optional explicit public capability
    decision_contract: "openjev"   # Optional protocol: d1, token_logits, candidate_logits, openjev, gliclass
    pad_seq_len: null             # Dynamic sequence length (or integer for fixed padding)
    providers: ["auto"]           # "auto", "TensorRT", "CUDA", "CoreML", "CPU"
    load_on_startup: true         # Load after startup TensorRT cache attempts complete
    # trt_cache_dir: "./custom-cache" # Optional; defaults to .cache/<name>
    intra_threads: 2
    inter_threads: 1

  - name: "sensevoice"
    alias: "whisper-1"
    hf_repo: "OpenVoiceOS/sensevoice-small-onnx"
    hf_file: "model_int8.onnx"
    tokenizer_path: null          # Auto-discovered (or specify explicit path)

  - name: "qwen3-embedding"
    alias: "text-embedding-3-small"
    hf_repo: "onnx-community/Qwen3-Embedding-0.6B-ONNX"
    hf_file: "onnx/model_int8.onnx"

  - name: "bge-reranker-v2-m3"
    hf_repo: "onnx-community/bge-reranker-v2-m3-ONNX"
    hf_file: "onnx/model_int8.onnx"

```

JetOrt does not require a model `type`. It detects a decoder loop from the graph and refuses that model. A `genai_config.json` file by itself does not. `engine_kind` is only for the ambiguous case.
System One bundles can use the explicit `task: decision` and optional
`decision_contract` fields above. The contract is configuration, not a required
sidecar manifest: JetOrt does not scan model directories for quantization
pipeline manifests. When omitted, existing companion-file and ONNX-signature
detection remains available for backward compatibility. Use the
llama.cpp-style `alias`, `ctx_size`, and `batch_size` names only when an override is needed.
JetOrt's lifecycle control is `load_on_startup`.

#### Decision model configuration

Decision models have two decision-specific configuration properties:

| Property | Required | Canonical values | Meaning |
|---|---:|---|---|
| `task` | No | `auto`, `decision`, `rerank`, `embedding`, `classification`, `asr`, `custom` | Public model capability. Use `decision` for `/v1/systemone` and `/v1/decision`. |
| `decision_contract` | No | `auto`, `gliclass`, `openjev`, `candidate_logits`, `token_logits`, `d1` | Input/output protocol used by the decision graph and tokenizer. |

For a d1 bundle, configure `task: decision` and `decision_contract: d1`.
`decision_contract` is only meaningful for decision models. `task` remains the
capability field; the contract must not be encoded as values such as
`decision_d1`. Hyphens and underscores are accepted interchangeably
(`candidate-logits`, `token_logits`, etc.). The existing auto-detection path
remains available when either field is omitted.

### Operational Logging

JetOrt emits structured, human-readable logs for server startup, per-model loading progress,
HTTP requests, inference queue time, latency, throughput, token rates, and process/system memory.
Every HTTP response includes `x-request-id`; callers may supply that header to preserve an
existing correlation ID.

```bash
# Override config.yaml for temporary diagnostics.
RUST_LOG=jetort=debug,ort=warn cargo run --release -- --config config.yaml
```

Logs always use the compact operational format. Embedding, reranking, raw ONNX, System One, and transcription logs include batch size, token counts, and tokens per second for the forward pass.

### Model Lifecycle and Memory Pressure

Startup is deliberately sequential:

1. Discover every configured model and inspect its TensorRT cache.
2. Compile each missing OneShot TensorRT cache in a separate child process, one at a time.
3. If a compile fails, remove TensorRT from that model's provider list for this process and continue with CUDA/CPU.
4. After compilation is complete, load `load_on_startup: true` models in configuration order.

The compiler process exits after each model so CUDA, TensorRT, and Jetson UMA build allocations
are fully reclaimed before the next model. JetOrt automatically reserves 12.5% of system memory
(clamped to 1-4 GB) and evicts the least-recently-used idle model under pressure. Active request
leases protect in-flight inference, including streaming responses, until completion.

Configured models remain visible in `/v1/models` with `meta.state` such as `loaded`, `unloaded`,
`loading`, or `failed`. `load_on_startup: false` skips startup load; the first request loads
that model using the already selected provider chain. Requests do not initiate a cold TensorRT
build. If loading fails, `meta.state` is `failed` and the model is not served as loaded.

`POST /v2/repository/models/:name/load` returns
`202 Accepted`; unloading an active model is deferred until its current requests finish.

---

## Hardware Deployment Guide

### NVIDIA Jetson (JetPack 5.x / 6.x / L4T)

TensorRT cold compilation temporarily consumes several copies of FP16 weights plus tactic
workspace in unified memory. JetOrt attempts it during startup, one model at a time. If the
build fails, the process continues on CUDA/CPU; a cached TensorRT engine is still loaded when
available.

For a 0.6B FP16 embedding or reranker on a 16 GB Jetson, build the cache with that model loaded
alone, then restore the full configuration:

```yaml
models:
  - name: "qwen3-reranker"
    path: "/models/Qwen3-Reranker-0.6B"
    hf_file: "model_fp16.onnx"
    providers: ["TensorRT", "CUDA"]
    max_batch_size: 1
    trt_max_workspace_mb: 256
    trt_profile_min_shapes: "input_ids:1x1,attention_mask:1x1"
    trt_profile_opt_shapes: "input_ids:1x256,attention_mask:1x256"
    trt_profile_max_shapes: "input_ids:1x512,attention_mask:1x512"
```

If no TensorRT cache is required, use `providers: ["CUDA", "CPU"]`; this avoids the expensive
cold build and is the recommended multi-model setting on a 16 GB device.

Each script is independent and safe to run repeatedly. Existing usable libraries are detected
with `ldconfig`/`ldd` and skipped without downloading or overwriting anything.

1. **Install or verify ONNX Runtime**:
   ```bash
   ./scripts/setup_onnxruntime.sh
   ./scripts/setup_onnxruntime.sh --check
   ```

2. **Build and start on Jetson**:
   ```bash
   ./scripts/jetson.sh
   ```
   `jetson.sh` calls `build.sh --target jetson` (CUDA arch 87 from `Cargo.toml`) and then
   `install-user-service.sh`. `--no-service` builds only. `--no-start` installs the unit
   without restarting it. `deploy.sh` does not compile.

   On NVIDIA systems, deployment enables TensorRT only when all three required components are
   found: `libnvinfer.so`, `NvInfer.h`, and the ONNX Runtime TensorRT provider library. Detection
   checks standard Jetson/x86 paths as well as `ldconfig`; the selected Cargo features are printed
   before compilation. At startup JetOrt logs its compiled providers and each model's automatic
   priority order (`TensorRT -> CUDA -> CPU`).

All setup scripts accept `--force` and `--prefix`. Use `--wheel` for an explicit ORT wheel.
For a complete fresh installation:

```bash
./scripts/setup_onnxruntime.sh &&
./scripts/jetson.sh
```

4. **Systemd Service Management**:
   ```bash
   systemctl --user status jetort.service
   journalctl --user -u jetort.service -f
   ```

### Generic Linux / Docker / macOS / Windows
JetOrt compiles natively across all major platforms:
* **macOS**: Automatically enables the **Apple CoreML** Execution Provider for Neural Engine (ANE) and GPU acceleration.
* **Windows**: Supports **DirectML** and **CUDA/TensorRT**.
* **Linux (x86_64 / aarch64)**: Supports **TensorRT**, **CUDA**, and **CPU**.

---

## Contributing

Contributions, issues, and feature requests are welcome!
Feel free to check [issues page](https://github.com/top/jetort/issues) for open tasks.

To run the test suite:
```bash
cargo test
```

Please ensure all Rust code adheres to the project's formatting and style guidelines:
```bash
cargo fmt --all -- --check
cargo clippy -- -D warnings
```

---

## License

This project is open-sourced under the [MIT License](LICENSE).
