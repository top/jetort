# JetOrt INT8-QDQ Quantization & Model Cooking Guide

This document details the standardization and toolchain for converting, calibrating, and bundling ONNX models into high-performance, TensorRT- and CPU-compatible **JetOrt Model Bundles**.

---

## 1. Why INT8-QDQ Is the Universal Standard

Traditional static quantization compresses weights and activations into integer operators (such as `MatMulInteger` or `QLinearConv`). While functional on pure CPU backends:
* **TensorRT Rejection**: TensorRT's optimization graph compiler cannot fuse or map arbitrary integer operators, leading to compilation aborts or severe memory fragmentation.
* **The Solution (Explicit QDQ)**: INT8-QDQ represents quantization explicitly as paired `QuantizeLinear` and `DequantizeLinear` nodes.
  - **NVIDIA TensorRT**: Maps QDQ nodes directly to `IQuantizeLayer` and `IDequantizeLayer`, activating INT8 Tensor Cores.
  - **Intel / AMD CPU**: Automatically accelerates QDQ using hardware VNNI and AVX-512 integer vector pipelines.
  - **ARM / Jetson CPU**: Leverages ARM NEON dot-product SIMD instructions.

---

## 2. Using `tools/cooker.py`

The `cooker.py` script automates the transformation of upstream models into self-contained bundles:

```bash
# 1. Cook an autoregressive LLM (e.g., Qwen 2.5 / Llama 3)
python tools/cooker.py \
  --model ./raw_models/qwen2.5-0.5b \
  --output ./bundles/qwen2.5-0.5b-int8-qdq \
  --quant int8_qdq \
  --type llm

# 2. Cook a one-shot Embedding model
python tools/cooker.py \
  --model ./raw_models/bge-small-en-v1.5/model.onnx \
  --output ./bundles/bge-small-int8-qdq \
  --quant int8_qdq \
  --type embedding
```

### What `cooker.py` does today
The script is not the engine's packaging toolchain. A `genai_config.json` it writes is not by itself how JetOrt classifies a model. JetOrt refuses autoregressive models detected from paired KV tensors or a trustworthy decoder marker. A thin `genai_config.json` does not trigger that refusal.

What the current script actually attempts:
1. **Config synthesis**: If `genai_config.json` is missing and the `--type` / filename heuristic says the model is generative, it writes a config from `config.json`. That file is not a completed cooker contract.
2. **QDQ attempt**: Calls `onnxruntime.quantization.quantize_static` when that package imports. On failure it copies the ONNX file unchanged.
3. **Metadata stamping**: When the `onnx` package is installed, it writes `jetort.framework_type` and `jetort.quant_type` onto `model.onnx`.

---

## 3. Validating Model Bundles with `tools/validate.py`

Before deploying a bundle to production, run the diagnostic validator:

```bash
python tools/validate.py ./bundles/qwen2.5-0.5b-int8-qdq
```

### Diagnostic Output Example:
```text
==================================================================
📁 Validating JetOrt Model Bundle: ./bundles/qwen2.5-0.5b-int8-qdq
==================================================================
📄 Bundle Structure Audit:
   • model.onnx:              ✅ Present
   • genai_config.json:       file present is not an engine kind
   • tokenizer.json:          ✅ Present
   • tokenizer_config.json:   ✅ Present
   • vision_encoder.onnx:     ⚪ Absent (One-Shot Mode)
   🔍 GenAI Type: qwen2
   💬 Jinja2 Chat Template:   ✅ Configured

==================================================================
🔍 Validating ONNX Model: model.onnx
==================================================================
📦 File Size: 512.40 MB
📋 ONNX IR Version: 8
📋 Producer: onnxruntime.quantization (1.19.0)
🏷️ Metadata Properties (3 entries):
   • jetort.engine_version: 2.0
   • jetort.quant_type: int8_qdq
   • jetort.framework_type: autoregressive

📊 Quantization Node Census:
   • QuantizeLinear:    142
   • DequantizeLinear:  142
   • MatMulInteger:     0
   • DynamicQuantize:   0

🎯 Compatibility Verdict:
   ✅ PASS: Pure INT8-QDQ graph. Fully compatible with TensorRT acceleration and CPU VNNI fallback.
```

---

## 4. Serving in `config.yaml`

Point your JetOrt `config.yaml` to the cooked model bundle:

```yaml
server:
  host: "0.0.0.0"
  port: 8080

models:
  - name: "qwen2.5-0.5b"
    path: "./bundles/qwen2.5-0.5b-int8-qdq/model.onnx"
    provider: "auto"
    concurrency_limit: 4

  - name: "qwen3-embedding"
    path: "./bundles/qwen3-embedding-0.6b/model.onnx"
    provider: "auto"

  # Decision models use the same ONNX loading path. Contract selection is
  # explicit configuration; no quantization manifest is required at runtime.
  - name: "d1"
    path: "./bundles/d1/model.onnx"
    task: "decision"
    decision_contract: "d1"
    tokenizer_path: "./bundles/d1/tokenizer.json"
    providers: ["auto"]
```

For decision models, the runtime configuration has two decision-specific
properties:

| Property | Values |
|---|---|
| `task` | `auto`, `decision`, `rerank`, `embedding`, `classification`, `asr`, `custom` |
| `decision_contract` | `auto`, `gliclass`, `openjev`, `candidate_logits`, `token_logits`, `d1` |

Use `task: decision` to select the public decision capability and
`decision_contract: d1` to select the d1 prompt/readout protocol. These fields
are optional for backward compatibility; when omitted, JetOrt keeps its
existing graph and companion-file detection. The quantization pipeline does
not need to generate a manifest that JetOrt must discover or load.

Start the engine:
```bash
cargo run --release -- --config config.yaml
```
