use jetort::protocol::codec::normalize_datatype;
use jetort::protocol::kserve_v2::{InferenceRequest, InferenceResponse};
use jetort::protocol::openai_v1::{EmbeddingRequest, RerankRequest};
use serde_json::json;

#[test]
fn test_normalize_datatype() {
    assert_eq!(normalize_datatype("int64"), "INT64");
    assert_eq!(normalize_datatype("I64"), "INT64");
    assert_eq!(normalize_datatype("float32"), "FP32");
    assert_eq!(normalize_datatype("float"), "FP32");
    assert_eq!(normalize_datatype("fp32"), "FP32");
    assert_eq!(normalize_datatype("bool"), "BOOL");
    assert_eq!(normalize_datatype("boolean"), "BOOL");
    assert_eq!(normalize_datatype("uint8"), "UINT8");
    assert_eq!(normalize_datatype("u8"), "UINT8");
}

#[test]
fn test_inference_request_deserialization() {
    let raw_json = json!({
        "id": "req-12345",
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
    });

    let req: Result<InferenceRequest, _> = serde_json::from_value(raw_json);
    assert!(req.is_ok());
    let req = req.unwrap();
    assert_eq!(req.id.as_deref(), Some("req-12345"));
    assert_eq!(req.inputs.len(), 2);
    assert_eq!(req.inputs[0].name, "input_ids");
    assert_eq!(req.inputs[0].shape, vec![1, 4]);
    assert_eq!(req.inputs[0].datatype, "INT64");
}

#[test]
fn test_inference_response_serialization() {
    let resp = InferenceResponse {
        model_name: "bge-small".to_string(),
        model_version: Some("1".to_string()),
        id: Some("req-001".to_string()),
        outputs: vec![jetort::protocol::kserve_v2::KServeOutputTensor {
            name: "sentence_embedding".to_string(),
            shape: vec![1, 3],
            datatype: "FP32".to_string(),
            data: json!([0.1, 0.2, 0.3]),
        }],
    };

    let serialized = serde_json::to_value(&resp).expect("Failed to serialize response");
    assert_eq!(serialized["model_name"], "bge-small");
    assert_eq!(serialized["outputs"][0]["name"], "sentence_embedding");
    assert_eq!(serialized["outputs"][0]["shape"], json!([1, 3]));
}

#[test]
fn test_embedding_request_single_and_batch() {
    let single_json = json!({
        "model": "bge-small",
        "input": "This is a single query"
    });
    let req1: EmbeddingRequest = serde_json::from_value(single_json).unwrap();
    assert_eq!(req1.input.into_vec(), vec!["This is a single query"]);

    let batch_json = json!({
        "model": "bge-small",
        "input": ["First sentence", "Second sentence"]
    });
    let req2: EmbeddingRequest = serde_json::from_value(batch_json).unwrap();
    assert_eq!(
        req2.input.into_vec(),
        vec!["First sentence", "Second sentence"]
    );
}

#[test]
fn test_rerank_request_parsing() {
    let rerank_json = json!({
        "model": "bge-reranker",
        "query": "How to deploy ONNX on Jetson?",
        "documents": [
            "Doc 1: NVIDIA JetPack includes TensorRT.",
            "Doc 2: Cooking recipe for dinner."
        ],
        "top_n": 1
    });

    let req: RerankRequest = serde_json::from_value(rerank_json).unwrap();
    assert_eq!(req.model, "bge-reranker");
    assert_eq!(req.documents.len(), 2);
    assert_eq!(req.top_n, Some(1));
}

#[test]
fn test_transcription_request_and_response() {
    let req_json = json!({
        "model": "sensevoice-small",
        "audio": [0.01, -0.02, 0.05, 0.03],
        "language": "zh"
    });

    let req: jetort::protocol::openai_v1::TranscriptionJsonRequest =
        serde_json::from_value(req_json).unwrap();
    assert_eq!(req.model, "sensevoice-small");
    assert_eq!(req.audio.unwrap().len(), 4);
    assert_eq!(req.language.as_deref(), Some("zh"));

    let resp = jetort::protocol::openai_v1::TranscriptionResponse {
        text: "Hello, welcome to the universal inference service.".to_string(),
    };
    let val = serde_json::to_value(&resp).unwrap();
    assert_eq!(
        val["text"],
        "Hello, welcome to the universal inference service."
    );
}

#[test]
fn test_minimal_hf_model_config() {
    let yaml_str = r#"
models:
  - name: "bge-small"
    hf_repo: "BAAI/bge-small-en-v1.5"
    hf_file: "onnx/model.onnx"
"#;
    let config: jetort::config::AppConfig = serde_yaml::from_str(yaml_str).unwrap();
    assert_eq!(config.models.len(), 1);
    let m = &config.models[0];
    assert_eq!(m.name, "bge-small");
    assert_eq!(m.hf_repo.as_deref(), Some("BAAI/bge-small-en-v1.5"));
    assert_eq!(m.hf_file.as_deref(), Some("onnx/model.onnx"));
    // Default runtime is "auto"
    assert_eq!(m.providers, vec!["auto".to_string()]);
}

#[test]
fn test_maybe_pad_tensor() {
    use jetort::protocol::codec::maybe_pad_tensor;
    use jetort::protocol::kserve_v2::KServeInputTensor;

    let input = KServeInputTensor {
        name: "input_ids".to_string(),
        shape: vec![1, 4],
        datatype: "INT64".to_string(),
        data: serde_json::json!([101, 2054, 2003, 102]),
    };

    // Pad to 8
    let padded = maybe_pad_tensor(&input, Some(8));
    assert_eq!(padded.shape, vec![1, 8]);
    let arr = padded.data.as_array().unwrap();
    assert_eq!(arr.len(), 8);
    assert_eq!(arr[0], 101);
    assert_eq!(arr[3], 102);
    assert_eq!(arr[4], 0);
    assert_eq!(arr[7], 0);

    // Truncate to 2
    let truncated = maybe_pad_tensor(&input, Some(2));
    assert_eq!(truncated.shape, vec![1, 2]);
    let arr = truncated.data.as_array().unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0], 101);
    assert_eq!(arr[1], 2054);
}

#[test]
fn test_full_config_yaml_parsing() {
    let config = jetort::config::AppConfig::load_from_file("config.yaml")
        .expect("Failed to parse config.yaml");
    assert_eq!(config.server.port, 11433);
    assert!(!config.models.is_empty());

    let sample_yaml = r#"
server:
  port: 11433

models:
  - name: Verdict
    hf_file: model_fp16.onnx
  - name: SenseVoiceSmall
    hf_file: model_int8.onnx
  - name: Qwen3-Embedding-0.6B
    hf_file: onnx/model_int8.onnx
  - name: bge-reranker-v2-m3
    hf_file: onnx/model_fp16.onnx
"#;
    let parsed: jetort::config::AppConfig =
        serde_yaml::from_str(sample_yaml).expect("Failed to parse sample yaml");
    assert_eq!(parsed.models.len(), 4);

    let verdict = &parsed.models[0];
    assert_eq!(verdict.name, "Verdict");
    assert_eq!(verdict.hf_file.as_deref(), Some("model_fp16.onnx"));

    let sensevoice = &parsed.models[1];
    assert_eq!(sensevoice.name, "SenseVoiceSmall");
    assert_eq!(sensevoice.hf_file.as_deref(), Some("model_int8.onnx"));

    let qwen = &parsed.models[2];
    assert_eq!(qwen.name, "Qwen3-Embedding-0.6B");
    assert_eq!(qwen.hf_file.as_deref(), Some("onnx/model_int8.onnx"));

    let reranker = &parsed.models[3];
    assert_eq!(reranker.name, "bge-reranker-v2-m3");
    assert_eq!(reranker.hf_file.as_deref(), Some("onnx/model_fp16.onnx"));
}

#[test]
fn test_sanitize_einsum_equations() {
    use jetort::engine::provider::sanitize_einsum_equations;

    // Real ONNX protobuf wire format:
    //   AttributeProto.name (field 1): tag=0x0a, len=0x08, "equation"
    //   AttributeProto.s    (field 4): tag=0x22, len=0x0a, "BD,BCD->BC"
    let mut data = b"prefix\x0a\x08equation\x22\x0aBD,BCD->BC suffix".to_vec();
    let count = sanitize_einsum_equations(&mut data);
    assert_eq!(count, 7); // B, D, B, C, D, B, C
    assert!(data.windows(10).any(|w| w == b"bd,bcd->bc"));
}

#[test]
fn test_memory_and_concurrency_config_defaults() {
    let yaml_str = r#"
models:
  - name: "test-model"
    path: "/tmp/model.onnx"
    cuda_memory_limit_mb: 2048
    trt_max_workspace_mb: 512
    enable_mem_pattern: false
    context_memory_sharing: true
    concurrency_limit: 2
"#;
    let config: jetort::config::AppConfig = serde_yaml::from_str(yaml_str).unwrap();
    assert_eq!(config.models.len(), 1);
    let m = &config.models[0];
    assert_eq!(m.cuda_memory_limit_mb, Some(2048));
    assert_eq!(m.trt_max_workspace_mb, Some(512));
    assert!(!m.enable_mem_pattern);
    assert!(m.context_memory_sharing);
    assert_eq!(m.concurrency_limit, Some(2));

    // Test default model config
    let default_yaml = r#"
models:
  - name: "default-model"
    path: "/tmp/model.onnx"
"#;
    let def_config: jetort::config::AppConfig = serde_yaml::from_str(default_yaml).unwrap();
    let def_m = &def_config.models[0];
    assert_eq!(def_m.cuda_memory_limit_mb, None);
    assert_eq!(def_m.trt_max_workspace_mb, Some(512));
    assert!(!def_m.enable_mem_pattern);
    assert!(def_m.context_memory_sharing);
    assert_eq!(def_m.concurrency_limit, None);
}

#[test]
fn test_trt_engine_cache_and_feasibility_check() {
    let temp_dir = std::env::temp_dir().join(format!("jetort_trt_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&temp_dir);

    // 1. Initially no cached engine
    assert!(!jetort::engine::provider::has_cached_trt_engine(
        temp_dir.to_str().unwrap()
    ));

    // 2. Create a mock engine file (> 1024 bytes)
    let engine_file = temp_dir.join("model_fp16.engine");
    let dummy_data = vec![0u8; 4096];
    std::fs::write(&engine_file, &dummy_data).unwrap();

    assert!(jetort::engine::provider::has_cached_trt_engine(
        temp_dir.to_str().unwrap()
    ));

    // 3. Test feasibility logic
    let yaml_str = format!("name: test-model\ntrt_cache_dir: {}\n", temp_dir.display());
    let mut config: jetort::config::ModelConfig = serde_yaml::from_str(&yaml_str).unwrap();

    // When cache exists, feasibility always passes immediately
    let (feasible, reason) =
        jetort::engine::provider::is_trt_compilation_feasible(&config, None, 1000 * 1024 * 1024);
    assert!(feasible);
    assert!(reason.contains("Found pre-compiled"));

    // Cleanup
    let _ = std::fs::remove_dir_all(&temp_dir);

    // Now test with non-existent cache directory (cold build check)
    config.trt_cache_dir = Some("/tmp/non_existent_trt_cache_dir_xyz".to_string());
    // Small model should pass
    let (small_feasible, _) =
        jetort::engine::provider::is_trt_compilation_feasible(&config, None, 50 * 1024 * 1024);
    assert!(small_feasible);

    #[cfg(target_os = "linux")]
    {
        // Test rejection when memory requirements exceed available system memory
        let (huge_feasible, huge_reason) = jetort::engine::provider::is_trt_compilation_feasible(
            &config,
            None,
            200_000 * 1024 * 1024,
        );
        assert!(!huge_feasible);
        assert!(huge_reason.contains("Insufficient system memory"));
    }
}

#[test]
fn test_detect_qdq_nodes() {
    use jetort::engine::provider::detect_qdq_nodes;

    // Positive case: contains both QuantizeLinear and DequantizeLinear
    let qdq_bytes = b"mock_onnx_node\x0a\x0eQuantizeLinear\x12\x10DequantizeLinear_op";
    assert!(detect_qdq_nodes(qdq_bytes));

    // Negative cases
    let only_q = b"mock_onnx_node\x0a\x0eQuantizeLinear";
    assert!(!detect_qdq_nodes(only_q));

    let only_dq = b"mock_onnx_node\x12\x10DequantizeLinear_op";
    assert!(!detect_qdq_nodes(only_dq));

    let clean_fp16 = b"mock_fp16_model_nodes_without_quantization";
    assert!(!detect_qdq_nodes(clean_fp16));
}

#[test]
fn test_detect_model_precision_and_streaming_qdq() {
    use jetort::engine::provider::{detect_model_precision, detect_qdq_in_file, ModelPrecision};
    use std::path::Path;

    // 1. Filename heuristics
    let p_int8_name = detect_model_precision(Path::new("/models/qwen/model_int8.onnx"));
    assert_eq!(p_int8_name, ModelPrecision::Int8);

    let p_fp16_name = detect_model_precision(Path::new("/models/bge/model_fp16.onnx"));
    assert_eq!(p_fp16_name, ModelPrecision::Fp16);

    let p_fp32_name = detect_model_precision(Path::new("/models/sensevoice/model_fp32.onnx"));
    assert_eq!(p_fp32_name, ModelPrecision::Fp32);

    // 2. Streaming QDQ file detection
    let temp_file =
        std::env::temp_dir().join(format!("test_qdq_stream_{}.bin", std::process::id()));
    let mock_data =
        b"random_prefix_bytes_...QuantizeLinear...intermediate_data...DequantizeLinear...trailing";
    std::fs::write(&temp_file, mock_data).unwrap();

    assert!(detect_qdq_in_file(&temp_file, 1024 * 1024));
    let detected = detect_model_precision(&temp_file);
    assert_eq!(detected, ModelPrecision::Int8);

    let _ = std::fs::remove_file(&temp_file);

    // 3. Fallback chains
    assert_eq!(
        ModelPrecision::Int8.trt_precision_fallback_chain(),
        &[
            ModelPrecision::Int8,
            ModelPrecision::Fp16,
            ModelPrecision::Fp32
        ]
    );
    assert_eq!(
        ModelPrecision::Fp16.trt_precision_fallback_chain(),
        &[ModelPrecision::Fp16, ModelPrecision::Fp32]
    );
    assert_eq!(
        ModelPrecision::Fp32.trt_precision_fallback_chain(),
        &[ModelPrecision::Fp16, ModelPrecision::Fp32]
    );
}

#[test]
fn test_semantic_trt_dynamic_profile_synthesizer() {
    use jetort::engine::onnx_introspect::{synthesize_trt_profiles, OnnxDim, OnnxInputTensorInfo};

    // 1. Audio Speech Model (SenseVoice / Whisper / Paraformer)
    let audio_inputs = vec![
        OnnxInputTensorInfo {
            name: "features".to_string(),
            elem_type: 1, // Float
            dims: vec![
                OnnxDim::Dynamic("batch".to_string()),
                OnnxDim::Dynamic("frames".to_string()),
                OnnxDim::Fixed(80),
            ],
        },
        OnnxInputTensorInfo {
            name: "features_lens".to_string(),
            elem_type: 7, // Int64
            dims: vec![OnnxDim::Dynamic("batch".to_string())],
        },
        OnnxInputTensorInfo {
            name: "language".to_string(),
            elem_type: 7,
            dims: vec![OnnxDim::Fixed(1)],
        },
        OnnxInputTensorInfo {
            name: "textnorm".to_string(),
            elem_type: 7,
            dims: vec![OnnxDim::Fixed(1)],
        },
    ];

    let audio_profiles =
        synthesize_trt_profiles(&audio_inputs, 8, 512).expect("Synthesis should succeed");
    assert_eq!(
        audio_profiles.min_shapes,
        "features:1x10x80,features_lens:1,language:1,textnorm:1"
    );
    assert_eq!(
        audio_profiles.opt_shapes,
        "features:1x1000x80,features_lens:1,language:1,textnorm:1"
    );
    assert_eq!(
        audio_profiles.max_shapes,
        "features:1x3000x80,features_lens:1,language:1,textnorm:1"
    );

    // 2. Transformer NLP / Embedding Model (Qwen / BGE / BERT)
    let nlp_inputs = vec![
        OnnxInputTensorInfo {
            name: "input_ids".to_string(),
            elem_type: 7,
            dims: vec![
                OnnxDim::Dynamic("batch".to_string()),
                OnnxDim::Dynamic("seq".to_string()),
            ],
        },
        OnnxInputTensorInfo {
            name: "attention_mask".to_string(),
            elem_type: 7,
            dims: vec![
                OnnxDim::Dynamic("batch".to_string()),
                OnnxDim::Dynamic("seq".to_string()),
            ],
        },
    ];

    let nlp_profiles =
        synthesize_trt_profiles(&nlp_inputs, 8, 512).expect("Synthesis should succeed");
    assert_eq!(nlp_profiles.min_shapes, "input_ids:1x1,attention_mask:1x1");
    assert_eq!(
        nlp_profiles.opt_shapes,
        "input_ids:1x256,attention_mask:1x256"
    );
    assert_eq!(
        nlp_profiles.max_shapes,
        "input_ids:8x512,attention_mask:8x512"
    );

    // 2b. Decision Model (e.g. Julia-1 with options / markers)
    let decision_inputs = vec![
        OnnxInputTensorInfo {
            name: "input_ids".to_string(),
            elem_type: 7,
            dims: vec![
                OnnxDim::Dynamic("batch".to_string()),
                OnnxDim::Dynamic("tokens".to_string()),
            ],
        },
        OnnxInputTensorInfo {
            name: "marker_pos".to_string(),
            elem_type: 7,
            dims: vec![
                OnnxDim::Dynamic("batch".to_string()),
                OnnxDim::Dynamic("options".to_string()),
            ],
        },
        OnnxInputTensorInfo {
            name: "qtype".to_string(),
            elem_type: 7,
            dims: vec![OnnxDim::Dynamic("batch".to_string())],
        },
    ];
    let decision_profiles =
        synthesize_trt_profiles(&decision_inputs, 4, 1024).expect("Decision synthesis should succeed");
    assert_eq!(
        decision_profiles.min_shapes,
        "input_ids:1x1,marker_pos:1x1,qtype:1"
    );
    assert_eq!(
        decision_profiles.opt_shapes,
        "input_ids:1x512,marker_pos:1x4,qtype:1"
    );
    assert_eq!(
        decision_profiles.max_shapes,
        "input_ids:4x1024,marker_pos:4x32,qtype:4"
    );

    // 3. Vision Model (e.g. ResNet / ViT)
    let vision_inputs = vec![OnnxInputTensorInfo {
        name: "pixel_values".to_string(),
        elem_type: 1,
        dims: vec![
            OnnxDim::Dynamic("batch".to_string()),
            OnnxDim::Fixed(3),
            OnnxDim::Dynamic("h".to_string()),
            OnnxDim::Dynamic("w".to_string()),
        ],
    }];

    let vision_profiles =
        synthesize_trt_profiles(&vision_inputs, 4, 512).expect("Synthesis should succeed");
    assert_eq!(vision_profiles.min_shapes, "pixel_values:1x3x128x128");
    assert_eq!(vision_profiles.opt_shapes, "pixel_values:1x3x224x224");
    assert_eq!(vision_profiles.max_shapes, "pixel_values:4x3x512x512");
}

#[test]
fn test_onnx_introspect_and_einsum_detection() {
    use jetort::engine::onnx_introspect::{
        introspect_onnx_inputs, parse_value_info_bytes, OnnxDim,
    };
    use jetort::engine::provider::detect_einsum_in_file;

    fn encode_varint(mut val: u64) -> Vec<u8> {
        let mut buf = Vec::new();
        while val >= 0x80 {
            buf.push((val as u8 & 0x7f) | 0x80);
            val >>= 7;
        }
        buf.push(val as u8);
        buf
    }

    fn encode_field(field_num: u32, wire_type: u32, payload: &[u8]) -> Vec<u8> {
        let mut buf = encode_varint(((field_num as u64) << 3) | (wire_type as u64));
        if wire_type == 2 {
            buf.extend(encode_varint(payload.len() as u64));
        }
        buf.extend_from_slice(payload);
        buf
    }

    // Build Dimension: dim_value=80 (tag 1, wire 0)
    let mut dim_80 = encode_varint(1 << 3);
    dim_80.extend(encode_varint(80));
    let dim_80_field = encode_field(1, 2, &dim_80);

    // Build Dimension: dim_param="batch" (tag 2, wire 2)
    let dim_batch = encode_field(2, 2, b"batch");
    let dim_batch_field = encode_field(1, 2, &dim_batch);

    // Build TensorShapeProto: dim=[batch, 80]
    let mut shape_proto = Vec::new();
    shape_proto.extend_from_slice(&dim_batch_field);
    shape_proto.extend_from_slice(&dim_80_field);
    let shape_field = encode_field(2, 2, &shape_proto);

    // Build Tensor: elem_type=1 (Float), shape
    let mut tensor_proto = encode_field(1, 0, &encode_varint(1));
    tensor_proto.extend_from_slice(&shape_field);
    let tensor_field = encode_field(1, 2, &tensor_proto);

    // Build TypeProto: tensor_type
    let type_field = encode_field(2, 2, &tensor_field);

    // Build ValueInfoProto: name="test_mel", type
    let name_field = encode_field(1, 2, b"test_mel");
    let mut vi_proto = Vec::new();
    vi_proto.extend_from_slice(&name_field);
    vi_proto.extend_from_slice(&type_field);

    // 1. Direct ValueInfoProto parsing test
    let parsed_vi = parse_value_info_bytes(&vi_proto).expect("Failed to parse ValueInfo bytes");
    assert_eq!(parsed_vi.name, "test_mel");
    assert_eq!(parsed_vi.elem_type, 1);
    assert_eq!(parsed_vi.dims.len(), 2);
    assert_eq!(parsed_vi.dims[0], OnnxDim::Dynamic("batch".to_string()));
    assert_eq!(parsed_vi.dims[1], OnnxDim::Fixed(80));

    // 2. Full ONNX file simulation:
    // ModelProto -> GraphProto with dummy initializer (tag 5, 2048 bytes), node with "Einsum", and input (tag 11)
    let mut graph_proto = Vec::new();

    // Large initializer (field 5, wire 2) to test zero-copy skipping
    let dummy_weights = vec![0xAAu8; 4096];
    graph_proto.extend_from_slice(&encode_field(5, 2, &dummy_weights));

    // Node (field 1, wire 2) containing "Einsum" and "equation"
    let node_with_einsum = b"mock_onnx_node_with_op_Einsum_and_equation_BD,BCD->BC";
    graph_proto.extend_from_slice(&encode_field(1, 2, node_with_einsum));

    // Input (field 11, wire 2)
    graph_proto.extend_from_slice(&encode_field(11, 2, &vi_proto));

    // ModelProto: field 7 is GraphProto
    let model_proto = encode_field(7, 2, &graph_proto);

    let temp_file =
        std::env::temp_dir().join(format!("test_onnx_introspect_{}.onnx", std::process::id()));
    std::fs::write(&temp_file, &model_proto).unwrap();

    // Test introspect_onnx_inputs
    let introspected =
        introspect_onnx_inputs(&temp_file).expect("Failed to introspect ONNX inputs");
    assert_eq!(introspected.len(), 1);
    assert_eq!(introspected[0].name, "test_mel");
    assert_eq!(introspected[0].dims.len(), 2);
    assert_eq!(introspected[0].dims[1], OnnxDim::Fixed(80));

    // Test detect_einsum_in_file
    assert!(detect_einsum_in_file(&temp_file, 1024 * 1024));

    // Test negative einsum detection
    let clean_temp =
        std::env::temp_dir().join(format!("test_clean_model_{}.bin", std::process::id()));
    std::fs::write(&clean_temp, b"standard_clean_model_without_einsum").unwrap();
    assert!(!detect_einsum_in_file(&clean_temp, 1024 * 1024));

    let _ = std::fs::remove_file(&temp_file);
    let _ = std::fs::remove_file(&clean_temp);
}

#[test]
fn test_comprehensive_onnx_model_inspection() {
    use jetort::config::ModelConfig;
    use jetort::engine::onnx_introspect::{inspect_onnx_model, QuantizationKind};

    fn encode_varint(mut val: u64) -> Vec<u8> {
        let mut res = Vec::new();
        loop {
            let mut byte = (val & 0x7F) as u8;
            val >>= 7;
            if val != 0 {
                byte |= 0x80;
                res.push(byte);
            } else {
                res.push(byte);
                break;
            }
        }
        res
    }

    fn encode_field(field_num: u32, wire_type: u32, data: &[u8]) -> Vec<u8> {
        let tag = (field_num << 3) | wire_type;
        let mut res = encode_varint(tag as u64);
        if wire_type == 2 {
            res.extend(encode_varint(data.len() as u64));
        }
        res.extend_from_slice(data);
        res
    }

    // 1. Build a synthetic ONNX ModelProto with CPU dynamic quantization (MatMulInteger)
    let mut graph_bytes = Vec::new();

    // Node 1: MatMulInteger (field 4 in NodeProto is op_type)
    let node1_data = encode_field(4, 2, b"MatMulInteger");
    graph_bytes.extend(encode_field(1, 2, &node1_data));

    // Node 2: DynamicQuantizeLinear
    let node2_data = encode_field(4, 2, b"DynamicQuantizeLinear");
    graph_bytes.extend(encode_field(1, 2, &node2_data));

    // Input: features [Dynamic("batch"), Dynamic("time"), Fixed(80)]
    let mut dim1 = Vec::new();
    dim1.extend(encode_field(2, 2, b"batch"));
    let mut dim2 = Vec::new();
    dim2.extend(encode_field(2, 2, b"time"));
    let mut dim3 = Vec::new();
    dim3.extend(encode_field(1, 0, &encode_varint(80)));

    let mut shape_bytes = Vec::new();
    shape_bytes.extend(encode_field(1, 2, &dim1));
    shape_bytes.extend(encode_field(1, 2, &dim2));
    shape_bytes.extend(encode_field(1, 2, &dim3));

    let mut tensor_type = Vec::new();
    tensor_type.extend(encode_field(1, 0, &encode_varint(1))); // Float
    tensor_type.extend(encode_field(2, 2, &shape_bytes));

    let mut val_info = Vec::new();
    val_info.extend(encode_field(1, 2, b"features"));
    val_info.extend(encode_field(2, 2, &encode_field(1, 2, &tensor_type)));

    graph_bytes.extend(encode_field(11, 2, &val_info));

    // Build ModelProto
    let mut model_bytes = Vec::new();
    // ir_version: 8 (field 1, wire 0)
    model_bytes.extend(encode_field(1, 0, &encode_varint(8)));
    // producer_name: "funasr_export" (field 2, wire 2)
    model_bytes.extend(encode_field(2, 2, b"funasr_export"));
    // producer_version: "2.1.0" (field 3, wire 2)
    model_bytes.extend(encode_field(3, 2, b"2.1.0"));
    // opset_import: field 8, wire 2
    let mut opset_entry = Vec::new();
    opset_entry.extend(encode_field(1, 2, b"ai.onnx"));
    opset_entry.extend(encode_field(2, 0, &encode_varint(17)));
    model_bytes.extend(encode_field(8, 2, &opset_entry));
    // metadata_props: "task"="asr" (field 14, wire 2)
    let mut meta_entry = Vec::new();
    meta_entry.extend(encode_field(1, 2, b"task"));
    meta_entry.extend(encode_field(2, 2, b"asr"));
    model_bytes.extend(encode_field(14, 2, &meta_entry));
    // graph: field 7, wire 2
    model_bytes.extend(encode_field(7, 2, &graph_bytes));

    let temp_path =
        std::env::temp_dir().join(format!("test_dynamic_model_{}.onnx", std::process::id()));
    std::fs::write(&temp_path, &model_bytes).unwrap();

    let config: ModelConfig = serde_yaml::from_str("name: test-sensevoice\n").unwrap();
    let inspection = inspect_onnx_model(&temp_path, &config).expect("Inspection should succeed");

    assert_eq!(inspection.ir_version, Some(8));
    assert_eq!(inspection.opset_version, Some(17));
    assert_eq!(inspection.producer_name.as_deref(), Some("funasr_export"));
    assert_eq!(inspection.producer_version.as_deref(), Some("2.1.0"));
    assert_eq!(
        inspection.metadata_props.get("task").map(|s| s.as_str()),
        Some("asr")
    );
    assert_eq!(
        inspection.quantization_kind,
        QuantizationKind::CpuDynamicInt8
    );
    assert!(inspection.is_audio_model);
    assert!(inspection.trt_compilation_feasible);
    assert_eq!(
        inspection.candidate_providers,
        jetort::engine::provider::get_candidate_providers(&config)
    );

    // Verify dynamic profile synthesis bounds audio batch to 1
    let profiles = inspection
        .synthesized_profiles
        .expect("Profiles should be synthesized");
    assert_eq!(profiles.min_shapes, "features:1x10x80");
    assert_eq!(profiles.opt_shapes, "features:1x1000x80");
    assert_eq!(profiles.max_shapes, "features:1x3000x80");

    let _ = std::fs::remove_file(&temp_path);
}

#[test]
fn test_header_prober_bundle_and_paradigm_detection() {
    use jetort::config::EngineKind;
    use jetort::engine::header_prober::probe_model_bundle;
    use std::fs::File;
    use std::io::Write;

    let dir = std::env::temp_dir().join(format!("test_bundle_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);

    let model_path = dir.join("model.onnx");
    let mut f = File::create(&model_path).unwrap();
    f.write_all(b"dummy onnx content").unwrap();

    let probe = probe_model_bundle(&model_path, None, None);
    assert_eq!(probe.kind, EngineKind::OneShot);
    assert!(!probe.has_genai_config);

    // A genai_config.json that does not describe a decoder loop stays one-shot.
    let genai_cfg_path = dir.join("genai_config.json");
    let mut cfg_f = File::create(&genai_cfg_path).unwrap();
    cfg_f.write_all(b"{\"model\": {}}").unwrap();

    let probe2 = probe_model_bundle(&model_path, None, None);
    assert_eq!(probe2.kind, EngineKind::OneShot);
    assert!(probe2.has_genai_config);
    assert!(!probe2.genai_config_substantive);

    // Clean up
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_resolve_local_model_and_relative_path() {
    use jetort::config::ModelConfig;
    use jetort::engine::session::resolve_model_and_tokenizer_path;
    use std::fs::File;
    use std::io::Write;

    let base_temp =
        std::env::temp_dir().join(format!("jetort_local_model_test_{}", std::process::id()));
    let model_dir = base_temp.join("models").join("sensevoice");
    std::fs::create_dir_all(&model_dir).expect("Failed to create test dir");

    let onnx_path = model_dir.join("model_int8_qdq.onnx");
    let mut f = File::create(&onnx_path).unwrap();
    f.write_all(b"fake onnx weights data").unwrap();

    let vocab_path = model_dir.join("vocab.txt");
    let mut vf = File::create(&vocab_path).unwrap();
    vf.write_all(b"token1\ntoken2\n").unwrap();

    // 1. Directory path + explicit file
    let yaml1 = format!(
        r#"
name: SenseVoiceLocal
path: "{}"
file: "model_int8_qdq.onnx"
"#,
        model_dir.display()
    );
    let cfg1: ModelConfig = serde_yaml::from_str(&yaml1).unwrap();
    let (resolved1, tok1) = resolve_model_and_tokenizer_path(&cfg1).unwrap();
    assert_eq!(resolved1, onnx_path);
    assert_eq!(tok1, Some(vocab_path.clone()));

    // 2. Direct path to ONNX file (file omitted)
    let yaml2 = format!(
        r#"
name: SenseVoiceDirect
path: "{}"
"#,
        onnx_path.display()
    );
    let cfg2: ModelConfig = serde_yaml::from_str(&yaml2).unwrap();
    let (resolved2, tok2) = resolve_model_and_tokenizer_path(&cfg2).unwrap();
    assert_eq!(resolved2, onnx_path);
    assert_eq!(tok2, Some(vocab_path.clone()));

    // 3. Relative path resolved against config_dir
    let yaml3 = r#"
name: SenseVoiceRel
path: "./models/sensevoice"
file: "model_int8_qdq.onnx"
"#;
    let mut cfg3: ModelConfig = serde_yaml::from_str(yaml3).unwrap();
    cfg3.config_dir = Some(base_temp.clone());
    let (resolved3, tok3) = resolve_model_and_tokenizer_path(&cfg3).unwrap();
    assert_eq!(resolved3, onnx_path);
    assert_eq!(tok3, Some(vocab_path.clone()));

    // 4. Directory without file (auto-discovery of model_int8_qdq.onnx)
    let yaml4 = format!(
        r#"
name: SenseVoiceAuto
path: "{}"
"#,
        model_dir.display()
    );
    let cfg4: ModelConfig = serde_yaml::from_str(&yaml4).unwrap();
    let (resolved4, tok4) = resolve_model_and_tokenizer_path(&cfg4).unwrap();
    assert_eq!(resolved4, onnx_path);
    assert_eq!(tok4, Some(vocab_path));

    // 5. Using aliases: 'model_path' and 'model_file'
    let yaml5 = format!(
        r#"
name: SenseVoiceAlias
model_path: "{}"
model_file: "model_int8_qdq.onnx"
"#,
        model_dir.display()
    );
    let cfg5: ModelConfig = serde_yaml::from_str(&yaml5).unwrap();
    let (resolved5, _) = resolve_model_and_tokenizer_path(&cfg5).unwrap();
    assert_eq!(resolved5, onnx_path);

    // Clean up
    let _ = std::fs::remove_dir_all(&base_temp);
}

#[test]
fn test_n_ctx_config_parsing_and_aliases() {
    use jetort::config::ModelConfig;

    let yaml = r#"
name: test-ctx
ctx_size: 4096
batch_size: 256
alias: public-name
"#;
    let cfg: ModelConfig = serde_yaml::from_str(yaml).unwrap();
    assert_eq!(cfg.effective_n_ctx(), Some(4096));
    assert_eq!(cfg.n_batch, 256);
    assert_eq!(cfg.alias.as_deref(), Some("public-name"));

    for legacy_field in [
        "n_ctx: 8192",
        "n_batch: 128",
        "preload: false",
        "aliases: [legacy-name]",
        "pinned: true",
    ] {
        let yaml = format!("name: test-legacy\n{legacy_field}\n");
        assert!(
            serde_yaml::from_str::<ModelConfig>(&yaml).is_err(),
            "legacy field should be rejected: {legacy_field}"
        );
    }

    let legacy_server = r#"
server:
  memory_min_available_mb: 2048
"#;
    assert!(serde_yaml::from_str::<jetort::config::AppConfig>(legacy_server).is_err());
}
