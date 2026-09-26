# Master Technical Specification & Development Plan
# Project: JetOrt (Universal Adaptive ONNX Inference Engine)

> **Vision**: An ultra-lightweight, unified, zero-overhead ONNX inference engine and server written in Rust. Covers the entire spectrum of ONNX models—from one-shot perception tasks (Embeddings, Cross-Encoder Reranking, Non-autoregressive ASR, Vision) to autoregressive Generative LLMs and Hybrid Encoder-Decoder architectures—with hardware-adaptive execution (TensorRT → CUDA → CoreML → DirectML → CPU) and strict INT8-QDQ standardization.

---

## 1. Engineering Meta-Guidelines & Refactoring Principles
*(Mandatory reading for all engineers and AI agents prior to modifying code)*

To ensure long-term maintainability, zero-crash reliability, and clean architectural evolution, any implementation or refactoring work on this repository must strictly adhere to the following five meta-principles:

### 1.1 "Context-First" Refactoring Discipline (先自省、再设计、后落子)
1. **Source & Context Ingestion**: Never write code blindly. Before proposing or implementing any change, thoroughly read the relevant modules, tracing their call graphs, implicit concurrency invariants, and memory reclamation hooks (such as `malloc_trim` or `ort::IoBinding`).
2. **Document Alignment**: Verify that the proposed changes strictly advance the target architecture outlined in this document rather than introducing temporary patches.
3. **Abstraction Precedes Implementation**: Before adding concrete logic, establish clean Trait boundaries (e.g. `EngineBackend`, `TokenStream`). Ensure that swapping the underlying C/C++ engine runtime in the future will require zero modifications to the upper HTTP/dispatch layers.
4. **Minimal Incremental Verification**: Avoid "big bang" refactorings. Every refactoring step must compile, maintain backward compatibility with existing tests, and be independently verifiable.

### 1.2 Southbound Micro-Core & Soft Dynamic Loading (南北向解耦与微内核化)
1. **Northbound Stability**: All client-facing interfaces (OpenAI `/v1/*`, KServe `/v2/*`) must maintain strict protocol contracts, predictable error responses, and clean JSON serialization. Protocol layers must know nothing about whether an ONNX model is running on TensorRT or CPU.
2. **Southbound Micro-Core & Soft Dynamic Loading**:
   - Both `libonnxruntime` and `libonnxruntime-genai` must be loaded dynamically at runtime via symbol lookup (`libloading` / `dlsym`).
   - **Forward Compatibility & ABI Isolation**: If the underlying shared library upgrades or modifies minor internal behaviors, as long as C ABI signatures remain stable, the compiled JetOrt binary requires zero recompilation.
   - **Graceful Partial Availability**: If `libonnxruntime-genai` is not installed on a host system, JetOrt must still start cleanly and serve all One-Shot models (Embeddings, Reranker, ASR); it must only return a clean runtime error when a GenAI chat endpoint is requested.
   - All raw C pointers (`*mut OgaModel`, `*mut OgaGenerator`) must be immediately wrapped in safe Rust structs implementing RAII `Drop`. **Zero pointer leakage** outside `src/engine/`.

### 1.3 Memory & Concurrency First Principles (内存与并发第一性原则)
1. **Memory Respect on Edge Hardware**: Many deployment targets (e.g. NVIDIA Jetson) use Unified Memory Architecture (UMA) where CPU and GPU share physical RAM.
   - Do not allocate unbounded buffers.
   - In dynamic shape scenarios, enforce sequence length padding ceilings (`pad_seq_len`) and micro-batch budgets.
   - Proactively reclaim heap arenas using system hooks (`malloc_trim(0)`) after heavy model loading or failed execution provider initialization.
2. **Minimal Lock Contention**:
   - Long-running forward passes (Token generation, large batch embeddings) must never block asynchronous Tokio reactor threads; always offload to dedicated blocking worker threads or channel-based background workers.
   - Model registry read/write locks (`RwLock`) must be held for microseconds only (retrieving `Arc<ModelSession>`); never hold a registry lock across an inference execution.

### 1.4 Defensive Simplicity & Anti-Over-Engineering (防御性简洁与反过度设计)
1. **Avoid Overly Complex Type Gymnastics**: Rust makes it tempting to introduce deeply nested lifetimes and generic parameter cascades. Prefer straightforward composition, explicit `enum` state machines, and clear trait objects (`Box<dyn Trait>` / `Arc<dyn Trait>`) where dynamic dispatch simplifies reasoning.
2. **Explicit Fallbacks over Panics**: Hardware environments vary widely. A missing CUDA driver or failed TensorRT compilation must never crash the service; it must gracefully degrade along the fallback chain (`TensorRT -> CUDA -> DirectML/CoreML -> CPU`) with structured telemetry logging.
3. **Strict File Independence (No Private Formats)**: Never invent bespoke binary formats or mutate upstream files. Upstream config files (`tokenizer.json`, `genai_config.json`, `config.json`) remain strictly untouched in their pristine industry-standard formats.

### 1.5 Zero-Regression & Self-Sustaining Delivery (零回归与自闭环契约)
This specification document is intended to be completely self-contained. Any engineer or AI agent operating in an isolated workspace should be able to audit the codebase against this document, identify gaps, and incrementally deliver complete, production-grade implementations without external context.

---

## 2. Input Asset Specifications & Pipeline Decomposition

JetOrt natively ingests self-contained **Standard Model Bundles** produced from raw upstream checkpoints (Safetensors / PyTorch) by the automated `cooker.py` toolchain:

### 2.1 The Three Canonical Model Bundle Types

```text
Type A: Pure One-Shot Model       Type B: Autoregressive LLM        Type C: Hybrid / Multimodal (e.g. MiniCPM-V)
┌───────────────────────────┐    ┌───────────────────────────┐    ┌─────────────────────────────────────────┐
│ model-bundle/             │    │ model-bundle/             │    │ model-bundle/                           │
│ ├── model.onnx (INT8-QDQ) │    │ ├── model.onnx (+.data)   │    │ ├── vision_encoder.onnx (INT8-QDQ TRT)  │
│ ├── model.onnx.data (opt) │    │ ├── genai_config.json     │    │ ├── model.onnx (+.data, LLM Backbone)   │
│ └── tokenizer.json        │    │ ├── tokenizer.json        │    │ ├── genai_config.json                   │
│                           │    │ └── tokenizer_config.json │    │ ├── tokenizer.json                      │
└───────────────────────────┘    └───────────────────────────┘    │ └── preprocessor_config.json            │
                                                                  └─────────────────────────────────────────┘
```

1. **Type A: Pure One-Shot Model (Embedding / Reranker / Non-AR ASR / YOLO)**
   - Files: `model.onnx` (and optional `model.onnx.data` if weights > 2GB), `tokenizer.json`.
   - Dispatch: Loaded exclusively by **One-Shot Core** (`engine::session`).
   - Execution: Static/semi-dynamic graph execution with output tensor zero-copy pooling.

2. **Type B: Autoregressive LLM (Qwen, Llama, Phi, MiniCPM)**
   - Files: `model.onnx` (+ `.data`), `genai_config.json`, `tokenizer.json`, `tokenizer_config.json`.
   - Dispatch: Identified by presence of `genai_config.json` or ONNX metadata `jetort.framework_type = "autoregressive"`.
   - Execution: Handled by **GenAI Core** (`engine::genai`) leveraging PagedAttention and FlashAttention.

3. **Type C: Hybrid / Multimodal Pipelines (MiniCPM-V, Whisper, Florence-2)**
   - **Vision / Audio Encoder (`vision_encoder.onnx` / `audio_encoder.onnx`)**:
     - Dense, one-shot compute topology.
     - Compiled to TensorRT engine cache with explicit INT8-QDQ nodes. Executed on **One-Shot Core**.
     - Generates image/audio embedding features once per request in milliseconds.
   - **Language Backbone (`model.onnx`)**:
     - Autoregressive decoder topology with `genai_config.json`.
     - Ingests pre-computed vision/audio feature vectors via device pointer zero-copy transfer. Executed on **GenAI Core**.
   - **Lifecycle Advantage**: Vision encoder device memory can be aggressively trimmed/reclaimed immediately after feature extraction, leaving maximum memory for long-context LLM token generation.

---

## 3. Dual-Core Engine & Shared Physical Substrate

### 3.1 Why Native ORT Remains the Uncontested Choice for One-Shot Tasks
For non-autoregressive tasks (Embeddings, Cross-Encoders, ViT, Non-AR ASR), Native ONNX Runtime is objectively unmatched:
* **True Multi-Platform Hardware Cascading**: It is the only production runtime that seamlessly falls back through `TensorRT -> CUDA -> CoreML -> DirectML -> CPU (VNNI/AVX-512/NEON)`. Pure TensorRT cannot run on CPU; OpenVINO cannot run on NVIDIA GPU; LibTorch is bloated.
* **Minimal Footprint**: Operates with tens of megabytes of resident memory.

### 3.2 Dual-Core Architectural Unification
While exposed as two clean logical modules in Rust, the two cores physically share the same underlying library:
```text
┌────────────────────────────────────────────────────────┐
│                     JetOrt Server                      │
├────────────────────────────┬───────────────────────────┤
│       One-Shot Core        │         GenAI Core        │
│   (Embedding / ViT / ASR)  │         (LLM / VLM)       │
└─────────────┬──────────────┴─────────────┬─────────────┘
              │ (Direct C-API / ort)        │ (Oga C-API)
              │                            ▼
              │                 libonnxruntime-genai.so
              │                 (PagedAttention & KV Loop)
              │                            │
              ▼                            ▼
┌────────────────────────────────────────────────────────┐
│             libonnxruntime.so (Unified Substrate)      │
│  • Single execution provider cascade (TRT/CUDA/CPU)    │
│  • Unified BFC allocator & hardware device streams     │
│  • Zero-copy GPU memory pointer passing across cores   │
└────────────────────────────────────────────────────────┘
```
This architecture ensures:
1. **Zero Library Redundancy**: Both cores share the single `libonnxruntime.so` on the system.
2. **Zero-Copy Hybrid Handoff**: Image feature embeddings produced on GPU by One-Shot Core are passed directly via CUDA memory pointer to GenAI Core without round-tripping through host RAM.

---

## 4. Hardware Arbiter & Multi-Platform Execution Blueprint

### 4.1 The Hardware Arbiter (Super-Engine Capability)
Hardware detection and execution provider arbitration is an autonomous capability situated **above the execution cores**. Before loading any model session, the `HardwareArbiter`:
1. Probes driver existence (`cudaGetDeviceCount`, presence of `libnvinfer.so` for TensorRT).
2. Measures host/device memory headroom against the model footprint.
3. Automatically synthesizes the optimal provider chain and profile shapes:

```text
                                  ┌───────────────────────────┐
                                  │   Incoming HTTP Request   │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Unified HTTP / Protocol Layer (Axum + Tokio)                                             │
│    • OpenAI v1 Compatible: /v1/chat/completions (SSE Stream), /v1/embeddings, /v1/rerank,   │
│      /v1/audio/transcriptions, /v1/models                                                   │
│    • KServe v2 Data Plane: /v2/models/{name}/infer, /v2/models/{name}, /v2/health/*         │
└───────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. Dynamic Model Registry & Dispatcher                                                      │
│    • Zero-copy Model Header Prober (Protobuf inspection of metadata_props & IO signatures)   │
│    • Bundle Signature Inspection (presence of genai_config.json / vision_encoder.onnx)      │
└───────────────────────┬─────────────────────────────────────────────┬───────────────────────┘
                        │                                             │
     [OneShotForward Paradigm]                         [AutoregressiveGenAI Paradigm]
                        │                                             │
                        ▼                                             ▼
┌───────────────────────────────────────────────┐ ┌───────────────────────────────────────────┐
│ 3A. One-Shot Execution Core (Native ORT)      │ │ 3B. GenAI Execution Core (ORT-GenAI Oga) │
│     • Tasks: Embedding, Reranker, Vision Enc  │ │     • Tasks: LLM, VLM Autoregressive Gen  │
│     • Native Rust bindings (`ort` crate)      │ │     • Soft dynamic C FFI (`oga-sys`)      │
│     • Custom Tokenizer & Vocab fast-path      │ │     • Automated KV-Cache & PagedAttention │
│     • Direct `IoBinding` / zero-copy buffers  │ │     • Hardware FlashAttention kernels     │
│     • Concurrency via Session Pools           │ │     • Token-by-Token SSE Stream generator │
└───────────────────────┬───────────────────────┘ └───────────────────────────┬───────────────┘
                        │                                                     │
                        └───────────────────────┬─────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. Universal Hardware Arbiter & Fallback Hierarchy                                          │
│    [Tier 1] TensorRT EP (with compiled engine cache & INT8-QDQ Tensor Core acceleration)    │
│       └──► [Tier 2] CUDA EP (Unified Memory / Fast Kernel Dispatch)                         │
│             └──► [Tier 3] CoreML (macOS ANE) / DirectML (Windows NPU/GPU)                   │
│                   └──► [Tier 4] CPU EP (Intel VNNI / AVX-512 / ARM NEON Hardware QDQ)       │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Header-Driven Tiered Arbitration & Cache Invariance Protocol
To guarantee zero-regression when loading community models without toolchain synthesis:
1. **Tier 1 (Cooked Bundles with Toolchain Metadata)**:
   - Uses `metadata_props` (`jetort.quant_type = "int8_qdq"`, `jetort.framework_type`) directly.
2. **Tier 2 (Bare ONNX Community Models - Zero-Copy Header Inspection)**:
   - **Explicit QDQ Verification**: Scans ONNX header nodes for `QuantizeLinear` + `DequantizeLinear`. Only confirmed QDQ models target TensorRT `INT8` (`with_int8(true)`).
   - **Dynamic INT8 Protection**: Community models with `MatMulInteger` / `DynamicQuantizeLinear` (e.g. `model_int8.onnx`) do NOT have TensorRT calibration graphs. They are safely routed to TensorRT **`FP16` mixed precision** (`with_int8(false).with_fp16(true)`), preventing the 12+ GB memory explosion in the TRT builder.
3. **Cache-First Invariance**:
   - If an existing `.engine` cache exists in `cache_dir`, the arbiter preserves the baseline profile signature and FP16 precision to ensure 100% cache-key match and instant load.


### 4.2 Hardware Optimization Matrix

| Deployment Target | One-Shot Core Path (Encoder/Vision) | GenAI Core Path (LLM) |
| :--- | :--- | :--- |
| **NVIDIA Jetson (Orin/Nano)** | TensorRT EP + INT8-QDQ Cached Engine (Fast-boot) | CUDA / TensorRT EP + PagedAttention (UMA Shared Memory) |
| **NVIDIA Desktop/Server** | TensorRT EP / CUDA EP (FP16 / INT8-QDQ) | CUDA EP + FlashAttention-2 Fused Kernels |
| **Apple Silicon (Mac)** | CoreML EP (ANE Acceleration) -> CPU fallback | CPU EP (NEON SIMD / Accelerate Framework) |
| **Intel Modern PC / Industrial** | OpenVINO / CPU EP (VNNI / AMX native acceleration) | CPU EP (AVX-512 + VNNI INT8 compute) |
| **Generic Embedded CPU** | CPU EP (Multithreaded, aggressive heap trimming) | CPU EP (INT8-QDQ sequential step loop) |

---

## 5. Technical Stack & Dependencies

| Subsystem | Component | Technology / Crate | Rationale |
| :--- | :--- | :--- | :--- |
| **Server Framework** | HTTP / REST / SSE | `axum` (0.7+), `tower`, `tokio` | Maximum async I/O performance, memory safety, native SSE streaming support. |
| **One-Shot Runtime** | Core ONNX Engine | `ort` (2.x) with `load-dynamic` | Official ONNX Runtime bindings, zero system C compilation dependencies. |
| **Generative Runtime**| GenAI Engine | `onnxruntime-genai` (C API) | Microsoft's official LLM engine with PagedAttention and automatic KV-Cache. Loaded dynamically via `libloading`. |
| **Tokenization** | Subword / BPE | `tokenizers` (HF official Rust) | High-speed, HuggingFace `tokenizer.json` drop-in compatibility. |
| **Template Engine** | Chat Formatting | `minijinja` | Renders `chat_template` (Jinja2) from `tokenizer_config.json` into prompt strings. |
| **Standard Quantization**| INT8-QDQ | `onnxruntime.quantization` | Produces explicit `QuantizeLinear` / `DequantizeLinear` nodes compatible with TensorRT & CPU VNNI. |

---

## 6. Detailed Component Specifications

### 6.1 The GenAI Execution Core (`src/engine/genai.rs` & `src/engine/oga_sys.rs`)
*   **Soft Dynamic C API Loader (`oga_sys`)**:
    *   Dynamically probes and loads `libonnxruntime-genai.so` (`.dylib` / `.dll`) via `libloading`.
    *   Functions resolved:
        *   `OgaCreateModel(path, &model)`
        *   `OgaCreateTokenizer(model, &tokenizer)`
        *   `OgaCreateGeneratorParams(model, &params)`
        *   `OgaGeneratorParamsSetSearchNumber(params, "max_length", val)`
        *   `OgaGeneratorParamsSetSearchNumber(params, "temperature", val)`
        *   `OgaGeneratorParamsSetSearchNumber(params, "top_p", val)`
        *   `OgaCreateGenerator(model, params, &generator)`
        *   `OgaGenerator_IsDone(generator)`
        *   `OgaGenerator_GenerateNextToken(generator)`
        *   `OgaGenerator_GetNextTokens(generator, &tokens, &count)`
        *   `OgaTokenizerDecode(tokenizer, tokens, count, &string_out)`
*   **Safe Rust Encapsulation (`GenAiSession`)**:
    *   Implements `Send + Sync`.
    *   Implements RAII `Drop` for clean memory deallocation of models, generators, and state buffers.
    *   Provides an async generator stream yielding token deltas via `tokio::sync::mpsc::channel`.

### 6.2 Chat Template Processing & Thinking Token Handling
*   Loads untouched `tokenizer_config.json` from the model snapshot directory.
*   Parses the Jinja2 template string under key `chat_template` via `minijinja`.
*   Renders OpenAI messages array (`[{role: "system", content: "..."}, {role: "user", content: "..."}]`) into the final tokenizable prompt.
*   Preserves raw special tokens (including `<think>`, `</think>`, `<|im_start|>`, `<|im_end|>`) so reasoning streams render natively in OpenAI-compatible clients.

### 6.3 Automated Quantization & Config Synthesis Pipeline (`tools/cooker.py`)
A standalone Python CLI tool that ingests standard upstream models (Safetensors / Hugging Face) and converts them into production-ready JetOrt bundles without mutating upstream formats:
*   **Command**: `python tools/cooker.py --hf-model <model-id> --output ./bundle --quant int8_qdq`
*   **Execution Steps**:
    1.  **Topology Split & Export**: If multimodal/hybrid (e.g. MiniCPM-V, Whisper), export `vision_encoder.onnx` and `model.onnx` separately.
    2.  **Config Synthesis**: Read upstream `config.json` to auto-synthesize standard `genai_config.json` (mapping attention heads, dimensions, EOS/BOS tokens).
    3.  **INT8-QDQ Calibration**: Run static calibration (using C4/Wikitext or MS MARCO subsets) and insert explicit `QuantizeLinear`/`DequantizeLinear` nodes.
    4.  **Metadata Injection**: Populate `ModelProto.metadata_props` with `jetort.framework_type`, `jetort.quant_type`, and version stamps.
*   **Output**: Ready-to-serve standard bundle directory.

---

## 7. Master Engineering Implementation Plan

### Phase 0: System Decoupling & Modernization
- [x] Baseline one-shot server with `axum` and `ort` (Current State).
- [x] Zero-copy ONNX header introspection (`src/engine/onnx_introspect.rs`).
- [x] Hardware telemetry and glibc memory compaction (`src/engine/memory.rs`).
- [x] Create paradigm classifier `src/engine/header_prober.rs`.
- [x] Refactor `ModelSession` into dispatch enum:
  ```rust
  pub enum ModelEngine {
      OneShot(OneShotSession),
      GenAi(GenAiSession),
  }
  ```
- [x] Maintain 100% test pass rate for existing embeddings, reranker, and ASR pipelines.

### Phase 1: Generative LLM Integration (ORT-GenAI)
- [x] Create `src/engine/oga_sys.rs`:
  - Implement soft dynamic symbol loader using `libloading` for `libonnxruntime-genai`.
  - Provide fallback behavior when library is absent.
- [x] Implement `src/engine/genai.rs`:
  - Safe Rust RAII wrapper around `OgaModel`, `OgaTokenizer`, and `OgaGenerator`.
  - Stream-based asynchronous token generation interface.
- [x] Integrate Jinja2 Chat Templating:
  - Add `minijinja` crate dependency.
  - Implement conversation rendering using `tokenizer_config.json`.
- [x] Implement OpenAI Chat Completions API (`src/server/handlers.rs`):
  - Endpoint: `POST /v1/chat/completions`.
  - Support `stream: false` (full buffered JSON response).
  - Support `stream: true` (Server-Sent Events `text/event-stream` with chunk deltas).

### Phase 2: INT8-QDQ Quantization & Auto-Synthesis Toolchain
- [x] Implement `tools/cooker.py`:
  - Support Hugging Face Safetensors and local ONNX inputs.
  - Auto-synthesize `genai_config.json` from `config.json`.
  - Enforce `QuantFormat.QDQ` static calibration.
  - Insert `metadata_props` into ModelProto.
- [x] Implement `tools/validate.py`:
  - CLI diagnostic tool that verifies whether a given `.onnx` has valid QDQ nodes and proper metadata headers for TensorRT compatibility.
- [x] Publish documentation (`docs/QUANTIZATION.md`):
  - Complete user guide for community model conversion.

### Phase 3: Performance Hardening & Zero-Copy Pipelines
- [x] Zero-copy tensor output:
  - Upgraded raw tensor inference and `OneShotSession` to support `infer_with_io_binding` using `ort::session::IoBinding` with automatic CPU/GPU fallback.
- [x] Dynamic batching & session pools:
  - Replaced `Mutex<Session>` in `OneShotSession` with an RAII-managed `SessionPool` (`src/engine/session_pool.rs`) with configurable `session_pool_size` (default 1 for Jetson UMA memory safety, $N$ for concurrent CPU/GPU workers).
- [x] Hybrid model orchestration:
  - Implemented `HybridSession` (`src/engine/hybrid.rs`) for Type C multimodal architectures (e.g. `vision_encoder.onnx` + `model.onnx` LLM backbone) with header-based automatic paradigm detection.

### Phase 4: Production Packaging & Release
- [x] Universal build scripts:
  - Updated `scripts/deploy.sh` to package both `libonnxruntime.so` and `libonnxruntime-genai.so` libraries, and auto-export `OGA_DYLIB_PATH`.
- [x] Containerization:
  - Multi-arch Dockerfiles (`Dockerfile.cuda`, `Dockerfile.cpu`, `Dockerfile.jetson`).
- [ ] Automated end-to-end integration tests:
  - Embeddings parity test with `qwen3-embedding-0.6b` INT8-QDQ.
  - Reranker scoring test against BGE reference scores.
  - Chat completions token streaming test with `Qwen2.5-0.5B-Instruct` / `Qwen3-0.6B` INT8-QDQ.

---

## 8. Verification & Acceptance Criteria

1. **Embedding & Reranker Parity**:
   - `/v1/embeddings` output matches standard Hugging Face ONNX outputs within cosine similarity $> 0.999$.
2. **Generative LLM Streaming**:
   - `/v1/chat/completions` responds via SSE stream with latency to first token (TTFT) $< 50\text{ms}$ on GPU / $< 150\text{ms}$ on CPU for small models (0.5B - 1.5B).
3. **Hardware Fallback Correctness**:
   - On an NVIDIA system with TensorRT installed: Engine compiles or loads cache, logs active provider `TensorRT`.
   - On a machine without NVIDIA drivers: Seamlessly boots with `CPU`, correctly activates VNNI/NEON INT8-QDQ instructions, zero crashes.
4. **Memory Footprint**:
   - Server baseline RSS $< 60\text{MB}$ without models loaded.
   - Dynamic model unload frees $> 95\%$ of allocated device/host memory back to the OS.
