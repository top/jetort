#!/usr/bin/env python3
"""
JetOrt Model Bundle & QDQ Validator CLI
Checks an ONNX model or bundle directory for:
- Explicit INT8-QDQ nodes (QuantizeLinear & DequantizeLinear)
- Illegal CPU-dynamic quantization nodes (MatMulInteger, DynamicQuantizeLinear)
- Uppercase Einsum equation compatibility
- Metadata properties (jetort.framework_type, jetort.quant_type)
- Companion configuration integrity (genai_config.json, tokenizer.json)
"""

import sys
import os
import json
import argparse
from pathlib import Path

try:
    import onnx
except ImportError:
    onnx = None


def check_onnx_model(model_path: Path):
    print(f"\n==================================================================")
    print(f"🔍 Validating ONNX Model: {model_path}")
    print(f"==================================================================")

    if not model_path.exists():
        print(f"❌ Error: File '{model_path}' does not exist.")
        return False

    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"📦 File Size: {file_size_mb:.2f} MB")

    if onnx is None:
        print("⚠️ Warning: 'onnx' Python package is not installed. Performing fast byte inspection.")
        with open(model_path, "rb") as f:
            header_sample = f.read(16 * 1024 * 1024)
        
        has_q = b"QuantizeLinear" in header_sample
        has_dq = b"DequantizeLinear" in header_sample
        has_dyn = b"DynamicQuantizeLinear" in header_sample or b"MatMulInteger" in header_sample

        print(f"   • QuantizeLinear detected: {'✅ Yes' if has_q else '❌ No'}")
        print(f"   • DequantizeLinear detected: {'✅ Yes' if has_dq else '❌ No'}")
        if has_dyn:
            print("   ⚠️ WARNING: CPU dynamic quantization (MatMulInteger) detected! TensorRT will reject this model.")
        if has_q and has_dq and not has_dyn:
            print("   ✨ INT8-QDQ signature detected: Compatible with TensorRT and CPU VNNI.")
        return True

    # Full protobuf analysis
    try:
        model = onnx.load(str(model_path), load_external_data=False)
    except Exception as e:
        print(f"❌ Failed to parse ONNX protobuf: {e}")
        return False

    print(f"📋 ONNX IR Version: {model.ir_version}")
    print(f"📋 Producer: {model.producer_name} ({model.producer_version})")

    # Metadata Props
    metadata = {prop.key: prop.value for prop in model.metadata_props}
    print(f"🏷️ Metadata Properties ({len(metadata)} entries):")
    for k, v in metadata.items():
        print(f"   • {k}: {v}")

    # Inspect Node Operators
    op_counts = {}
    has_uppercase_einsum = False
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        if node.op_type == "Einsum":
            for attr in node.attribute:
                if attr.name == "equation":
                    eq = attr.s.decode("utf-8", errors="ignore")
                    if any(c.isupper() for c in eq):
                        has_uppercase_einsum = True

    q_count = op_counts.get("QuantizeLinear", 0)
    dq_count = op_counts.get("DequantizeLinear", 0)
    matmul_int_count = op_counts.get("MatMulInteger", 0)
    dyn_q_count = op_counts.get("DynamicQuantizeLinear", 0)

    print("\n📊 Quantization Node Census:")
    print(f"   • QuantizeLinear:    {q_count}")
    print(f"   • DequantizeLinear:  {dq_count}")
    print(f"   • MatMulInteger:     {matmul_int_count}")
    print(f"   • DynamicQuantize:   {dyn_q_count}")

    # Verdict
    is_qdq = q_count > 0 and dq_count > 0 and matmul_int_count == 0
    is_cpu_dyn = matmul_int_count > 0 or dyn_q_count > 0

    print("\n🎯 Compatibility Verdict:")
    if is_qdq:
        print("   ✅ PASS: Pure INT8-QDQ graph. Fully compatible with TensorRT acceleration and CPU VNNI fallback.")
    elif is_cpu_dyn:
        print("   ⚠️ CPU-Only: Model uses CPU dynamic quantization. TensorRT will reject compilation.")
    else:
        print("   ℹ️ Standard Precision: FP32 or FP16 unquantized model.")

    if has_uppercase_einsum:
        print("   ⚠️ Notice: Uppercase characters detected in Einsum equations. JetOrt will automatically sanitize this on cold load.")

    return True


def check_bundle_directory(bundle_dir: Path):
    print(f"\n==================================================================")
    print(f"📁 Validating JetOrt Model Bundle: {bundle_dir}")
    print(f"==================================================================")

    model_file = bundle_dir / "model.onnx"
    genai_cfg = bundle_dir / "genai_config.json"
    tok_json = bundle_dir / "tokenizer.json"
    tok_cfg = bundle_dir / "tokenizer_config.json"
    vision_enc = bundle_dir / "vision_encoder.onnx"

    print("📄 Bundle Structure Audit:")
    print(f"   • model.onnx:              {'✅ Present' if model_file.exists() else '❌ Missing'}")
    print(f"   • genai_config.json:       {'✅ Present (GenAI Mode)' if genai_cfg.exists() else '⚪ Absent (One-Shot Mode)'}")
    print(f"   • tokenizer.json:          {'✅ Present' if tok_json.exists() else '⚪ Absent'}")
    print(f"   • tokenizer_config.json:   {'✅ Present' if tok_cfg.exists() else '⚪ Absent'}")
    print(f"   • vision_encoder.onnx:     {'✅ Present (Hybrid Multimodal)' if vision_enc.exists() else '⚪ Absent'}")

    if genai_cfg.exists():
        try:
            with open(genai_cfg, "r", encoding="utf-8") as f:
                data = json.load(f)
            model_info = data.get("model", {})
            print(f"   🔍 GenAI Type: {model_info.get('type', 'generic')}")
            print(f"   🔍 Vocab Size: {model_info.get('vocab_size', 'unknown')}")
        except Exception as e:
            print(f"   ❌ Corrupted genai_config.json: {e}")

    if tok_cfg.exists():
        try:
            with open(tok_cfg, "r", encoding="utf-8") as f:
                data = json.load(f)
            has_template = "chat_template" in data
            print(f"   💬 Jinja2 Chat Template: {'✅ Configured' if has_template else '⚪ Absent'}")
        except Exception as e:
            print(f"   ⚠️ Could not read tokenizer_config.json: {e}")

    if model_file.exists():
        check_onnx_model(model_file)
    if vision_enc.exists():
        check_onnx_model(vision_enc)


def main():
    parser = argparse.ArgumentParser(description="JetOrt Model Bundle & QDQ Validator")
    parser.add_argument("path", type=str, help="Path to an .onnx file or model bundle directory")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"Error: Path '{target}' does not exist.")
        sys.exit(1)

    if target.is_dir():
        check_bundle_directory(target)
    else:
        check_onnx_model(target)


if __name__ == "__main__":
    main()
