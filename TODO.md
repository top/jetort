# JetOrt remaining work

JetOrt is a single-pass ONNX engine. Autoregressive models are refused at load. onnxruntime-genai is not part of the process.

- **Cooker contract.** `tools/cooker.py` is a partial helper. It does not define the bundle contract for single-pass models (calibration data, metadata that detection can trust).
- **ROCm.** Linux AMD builds are CPU-only. ROCm is a named future target and is not wired up.
- **KServe v2 infer.** `POST /v2/models/{name}/infer` returns HTTP 501 `not_implemented`. Metadata and health routes stay. Do not advertise infer on model cards until a model can serve it.
