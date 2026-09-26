# JetOrt

<p align="center">
  <img src="https://raw.githubusercontent.com/raytang/jetort/main/assets/banner.png" alt="JetOrt Banner" width="700" onerror="this.style.display='none'"/>
</p>

<p align="center">
  <strong>Ultra-lightweight, zero-overhead one-shot tensor inference engine and server, powered by Rust.</strong>
</p>

<p align="center">
  <a href="https://github.com/raytang/jetort/actions"><img src="https://img.shields.io/github/actions/workflow/status/raytang/jetort/ci.yml?branch=main&label=CI&logo=github" alt="CI Status"/></a>
  <a href="https://crates.io/crates/jetort"><img src="https://img.shields.io/badge/rust-1.80%2B-orange.svg?logo=rust" alt="Rust Version"/></a>
  <a href="https://github.com/raytang/jetort/releases"><img src="https://img.shields.io/github/v/release/raytang/jetort?color=blue&logo=github" alt="GitHub Release"/></a>
  <a href="https://github.com/raytang/jetort/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"/></a>
  <a href="https://platform.openai.com/docs/api-reference"><img src="https://img.shields.io/badge/OpenAI%20API-Compatible-412991.svg" alt="OpenAI Compatible"/></a>
  <a href="https://kserve.github.io/website/latest/modelserving/data_plane/v2_protocol/"><img src="https://img.shields.io/badge/KServe%20v2-CNCF%20Standard-blue.svg" alt="KServe v2"/></a>
</p>

---

## Overview

Modern AI deployments typically pair a generative Large Language Model (LLM) with auxiliary perception models for embeddings, cross-encoder reranking, speech recognition (ASR), and computer vision.

While tools like `llama.cpp` excel at autoregressive LLM generation, auxiliary models are traditionally run through Python stacks (PyTorch, Transformers, FastAPI). This introduces severe overhead:
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
|  |  * /v1/chat/completions (SSE Stream)  |  |  * /v2/health/live               |  |
|  |  * /v1/models                         |  |  * /v2/health/ready              |  |
|  |  * /v1/embeddings                     |  |  * /v2/models/:name              |  |
|  |  * /v1/rerank                         |  |  * /v2/models/:name/infer        |  |
|  |  * /v1/systemone                      |  |  * /v2/repository/models/...     |  |
|  |  * /v1/audio/transcriptions           |  |  * /v2/repository/models/...     |  |
|  +---------------------------------------+  +----------------------------------+  |
|                                         |                                         |
|                 +-----------------------v-----------------------+                 |
|                 | Tokenizer & Vocab Auto-Discovery              |                 |
|                 | (tokenizer.json, vocab.json, Jinja2 template) |                 |
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
|        TensorRT (Cached Engine) -> CUDA -> CoreML -> DirectML -> CPU              |
+-----------------------------------------------------------------------------------+
```

---

## Key Features

* **High-Fidelity OpenAI API Parity**:
  - `/v1/chat/completions`: Full streaming SSE (`stream: true` with standard `data: {...}` and `data: [DONE]`, `stream_options.include_usage`) and non-streaming responses. Accurate `finish_reason` (`stop` vs `length`), system fingerprints, and token metrics.
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
* **CNCF KServe v2 Data Plane (Open Inference Protocol)**:
  - Standard multi-dimensional raw tensor forward passes (`/v2/models/{name}/infer`), model metadata reflection, and health probes (`/v2/health/live`, `/v2/health/ready`).
* **Universal Model-Agnostic Tokenizer & Vocab Auto-Discovery**:
  - Automatically locates and parses `tokenizer.json`, `vocab.json`, `vocab.txt`, `tokens.txt`, and `dict.txt` from local Hugging Face cache snapshots.
* **Multiplatform Acceleration with Auto-Fallback**:
  - Full execution provider cascade: `TensorRT -> CUDA -> CoreML -> DirectML -> CPU`.
* **Persistent TensorRT Engine Cache**:
  - Automatically compiles and serializes TensorRT engines to `.cache/<model>` to eliminate subsequent warmup delays across restarts.
* **Dynamic Static-Shape Auto-Padding**:
  - Configurable `pad_seq_len` automatically pads smaller inputs with zeros and truncates larger inputs in-memory, allowing models with static dimensions to accept variable-length inputs without caller-side manipulation.
* **Smart Model Alias & Fallback Routing**:
  - Built-in fallback routing handles clients that hardcode `whisper-1` or `text-embedding-ada-002` by automatically delegating to registered audio or embedding models.
* **Zero-Downtime Hot Reload**:
  - Atomically load, update, and unload models at runtime via HTTP API using lock-free `Arc<RwLock>` registry management.

---

## Quickstart

### 1. Installation

#### Option A: Pre-built Binaries
Download the latest release binary for your platform from [GitHub Releases](https://github.com/raytang/jetort/releases).

#### Option B: Build from Source
Ensure Rust 1.80+ is installed:
```bash
git clone https://github.com/raytang/jetort.git
cd jetort
cargo build --release
```
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

### 1. Chat Completions (`/v1/chat/completions` & `/chat/completions`)

Fully compatible with official OpenAI SDKs, LangChain, OpenWebUI, LlamaIndex, and cURL. Supports both token streaming via Server-Sent Events (SSE) and synchronous responses.

#### cURL Streaming SSE Example (`stream: true`):
```bash
curl -N -s http://localhost:11433/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniCPM5-2B",
    "messages": [
      {"role": "user", "content": "Hello! Introduce yourself briefly."}
    ],
    "max_tokens": 64,
    "temperature": 0.7,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

Stream Output:
```text
data: {"id":"chatcmpl-1790261413473","object":"chat.completion.chunk","created":1790261413,"model":"MiniCPM5-2B","system_fingerprint":"fp_jetort","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-1790261413473","object":"chat.completion.chunk","created":1790261413,"model":"MiniCPM5-2B","system_fingerprint":"fp_jetort","choices":[{"index":0,"delta":{"content":"Hello"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-1790261413473","object":"chat.completion.chunk","created":1790261413,"model":"MiniCPM5-2B","system_fingerprint":"fp_jetort","choices":[{"index":0,"delta":{"content":"!"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chatcmpl-1790261413473","object":"chat.completion.chunk","created":1790261413,"model":"MiniCPM5-2B","system_fingerprint":"fp_jetort","choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}

data: {"id":"chatcmpl-1790261413473","object":"chat.completion.chunk","created":1790261413,"model":"MiniCPM5-2B","system_fingerprint":"fp_jetort","choices":[],"usage":{"prompt_tokens":14,"completion_tokens":20,"total_tokens":34}}

data: [DONE]
```

#### Python OpenAI SDK Example:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11433/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="MiniCPM5-2B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a fun fact."}
    ],
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content if chunk.choices else ""
    if content:
        print(content, end="", flush=True)
print()
```

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
```bash
curl -s -X POST http://localhost:11433/v2/models/qwen3-embedding/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "input_ids",
        "shape": [1, 4],
        "datatype": "INT64",
        "data": [101, 2054, 2003, 102]
      },
      {
        "name": "attention_mask",
        "shape": [1, 4],
        "datatype": "INT64",
        "data": [1, 1, 1, 1]
      }
    ]
  }' | jq
```

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
  genai_dylib_path: null          # Optional manual path to libonnxruntime-genai.so (OGA_DYLIB_PATH)

models:
  - name: "verdict"
    alias: "jev-latest"            # llama.cpp-style single public alias
    hf_repo: "heman10x/rlcd-modernbert-151m"
    hf_file: "model_fp16.onnx"
    pad_seq_len: null             # Dynamic sequence length (or integer for fixed padding)
    providers: ["auto"]           # "auto", "TensorRT", "CUDA", "CoreML", "CPU"
    load_on_startup: true         # Load after all missing TensorRT caches are compiled
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

  - name: "MiniCPM5-2B"
    path: "/home/ray/quant_pipeline/work/openbmb_MiniCPM5-2B/genai_int8_mixed_cuda"
    file: "model.onnx"
    ctx_size: 32768
    # llama.cpp `models.ini`-style per-model sampling defaults. Applied only when a
    # /v1/chat/completions request doesn't set its own value; a per-request value always
    # wins. Useful because some fine-tunes (reasoning/tool-calling models in particular)
    # only produce well-formed output at a specific temperature/top_p/top_k.
    temperature: 1.0               # alias: temp
    top_p: 0.95                    # alias: top-p
    top_k: 40                      # alias: top-k
    repetition_penalty: 1.1        # aliases: repeat_penalty, repeat-penalty, repetition-penalty
```

JetOrt does not require a model `type`. It detects GenAI, Hybrid, ASR, Embedding, Reranker,
and supported System One bundles from companion files and ONNX input/output metadata. Use the
llama.cpp-style `alias`, `ctx_size`, and `batch_size` names only when an override is needed.
JetOrt's lifecycle control is `load_on_startup`.

### Per-Model Default Sampling Parameters

Mirroring llama.cpp's `models.ini` (`temp`, `top-p`, `load-on-startup`, ...), each entry under
`models:` may set its own default `temperature`, `top_p`, `top_k`, and `repetition_penalty`.
This matters because some models (especially reasoning/tool-calling fine-tunes) only emit
well-formed output — closed tags, complete tool calls, correctly terminated Markdown — at a
specific sampling configuration; leaving generation at the server's hardcoded defaults can
silently produce malformed completions that look like an inference bug but are actually a
mismatched sampling config.

Resolution order for every `/v1/chat/completions` request, per parameter:

1. The value set explicitly in the request body (OpenAI-compatible `temperature`/`top_p`, plus
   the non-standard `top_k` extension field also honored by llama.cpp/vLLM-style servers).
2. This model's `config.yaml` default (`temperature`/`top_p`/`top_k`/`repetition_penalty`, or
   their llama.cpp-style aliases `temp`/`top-p`/`top-k`/`repeat-penalty`).
3. JetOrt's server-wide fallback (`temperature: 0.7`, `top_p: 0.95`, no `top_k`/`repetition_penalty`).

`repetition_penalty` additionally accepts the request's OpenAI `presence_penalty` or
`frequency_penalty` fields ahead of the model default, for compatibility with existing
OpenAI-style clients that don't know about `repetition_penalty` directly.

`top_k` currently only applies to Autoregressive GenAI (ORT-GenAI) models; it is a no-op for
other paradigms.

### Operational Logging

JetOrt emits structured, human-readable logs for server startup, per-model loading progress,
HTTP requests, inference queue time, latency, throughput, token rates, and process/system memory.
Every HTTP response includes `x-request-id`; callers may supply that header to preserve an
existing correlation ID.

```bash
# Override config.yaml for temporary diagnostics.
RUST_LOG=jetort=debug,ort=warn cargo run --release -- --config config.yaml
```

Logs always use the compact operational format. Generation logs include prompt evaluation and decoding tokens/second; embedding, reranking,
raw ONNX, System One, and transcription logs include their relevant batch or realtime metrics.

### Model Lifecycle and Memory Pressure

Startup is deliberately sequential:

1. Discover every configured model and inspect its TensorRT cache.
2. Compile each missing OneShot TensorRT cache in a separate child process, one at a time.
3. After compilation is complete, load `load_on_startup: true` models in configuration order.

The compiler process exits after each model so CUDA, TensorRT, and Jetson UMA build allocations
are fully reclaimed before the next model. JetOrt automatically reserves 12.5% of system memory
(clamped to 1-4 GB) and evicts the least-recently-used idle model under pressure. Active request
leases protect in-flight inference, including streaming responses, until completion.

Configured models remain visible in `/v1/models` with `meta.state` such as `loaded`, `unloaded`,
`loading`, or `failed`. Requesting an unloaded model starts one serialized background load and
returns:

```http
HTTP/1.1 503 Service Unavailable
Retry-After: 2
```

```json
{
  "error": "Model 'example' is not loaded",
  "code": "model_not_loaded",
  "state": "loading",
  "retryable": true
}
```

Retry after the model reaches `loaded`. `POST /v2/repository/models/:name/load` returns
`202 Accepted`; unloading an active model is deferred until its current requests finish.

---

## Hardware Deployment Guide

### NVIDIA Jetson (JetPack 5.x / 6.x / L4T)

TensorRT cold compilation temporarily consumes several copies of FP16 weights plus tactic
workspace in unified memory. JetOrt now skips cold compilation and falls back to CUDA when the
estimated peak would exhaust safe NvMap headroom. A cached TensorRT engine is still loaded.

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

2. **Install or build ONNX Runtime GenAI when generative models are needed**:
   ```bash
   ./scripts/setup_genai.sh
   ./scripts/setup_genai.sh --check
   ```

3. **Build JetOrt and install/update its user service**:
   ```bash
   ./scripts/deploy.sh
   ```
   `deploy.sh` never installs Rust or system libraries. Use `--no-service` for a build only, or
   `--no-start` to update the unit without restarting it.

   On NVIDIA systems, deployment enables TensorRT only when all three required components are
   found: `libnvinfer.so`, `NvInfer.h`, and the ONNX Runtime TensorRT provider library. Detection
   checks standard Jetson/x86 paths as well as `ldconfig`; the selected Cargo features are printed
   before compilation. At startup JetOrt logs its compiled providers and each model's automatic
   priority order (`TensorRT -> CUDA -> CPU`).

All setup scripts accept `--force` and `--prefix`. Use `--wheel` for an explicit ORT wheel and
`--package` for a prebuilt GenAI archive. For a complete fresh installation:

```bash
./scripts/setup_onnxruntime.sh &&
./scripts/setup_genai.sh &&
./scripts/deploy.sh
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
Feel free to check [issues page](https://github.com/raytang/jetort/issues) for open tasks.

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
