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

### What `cooker.py` Automates:
1. **Config Synthesis**: If `genai_config.json` is missing, it inspects `config.json` and synthesizes Microsoft ORT-GenAI compatible topology configurations.
2. **Static Calibration**: Injects explicit QDQ nodes (`QuantFormat.QDQ`) with per-channel weight scaling.
3. **Metadata Stamping**: Embeds `jetort.framework_type` and `jetort.quant_type = "int8_qdq"` directly into the ONNX Protobuf header.

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
   • genai_config.json:       ✅ Present (GenAI Mode)
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
```

Start the engine:
```bash
cargo run --release -- --config config.yaml
```
