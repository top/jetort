use jetort::engine::header_prober::{probe_model_bundle, ModelParadigm};
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn create_temp_test_dir(prefix: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("jetort-hybrid-{}-{}", prefix, nanos));
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn test_probe_hybrid_bundle_with_vision_encoder() {
    let dir = create_temp_test_dir("vision");
    let model_dir = dir.join("minicpmv");
    fs::create_dir_all(&model_dir).unwrap();

    let model_onnx = model_dir.join("model.onnx");
    let genai_config = model_dir.join("genai_config.json");
    let vision_encoder = model_dir.join("vision_encoder.onnx");

    fs::write(&model_onnx, b"dummy onnx backbone").unwrap();
    fs::write(&genai_config, b"{}").unwrap();
    fs::write(&vision_encoder, b"dummy vision encoder").unwrap();

    let probe = probe_model_bundle(&model_onnx, None);
    assert_eq!(probe.paradigm, ModelParadigm::Hybrid);
    assert!(probe.has_genai_config);
    assert!(probe.has_vision_encoder);
    assert!(!probe.has_audio_encoder);
    assert_eq!(probe.framework_type.as_deref(), Some("hybrid_multimodal"));
    assert!(probe.detection_reason.contains("Found multimodal encoder"));
}

#[test]
fn test_probe_hybrid_bundle_with_audio_encoder() {
    let dir = create_temp_test_dir("audio");
    let model_dir = dir.join("whisper_hybrid");
    fs::create_dir_all(&model_dir).unwrap();

    let model_onnx = model_dir.join("model.onnx");
    let genai_config = model_dir.join("genai_config.json");
    let audio_encoder = model_dir.join("audio_encoder.onnx");

    fs::write(&model_onnx, b"dummy onnx backbone").unwrap();
    fs::write(&genai_config, b"{}").unwrap();
    fs::write(&audio_encoder, b"dummy audio encoder").unwrap();

    let probe = probe_model_bundle(&model_onnx, None);
    assert_eq!(probe.paradigm, ModelParadigm::Hybrid);
    assert!(probe.has_genai_config);
    assert!(!probe.has_vision_encoder);
    assert!(probe.has_audio_encoder);
    assert_eq!(probe.framework_type.as_deref(), Some("hybrid_multimodal"));
}

#[test]
fn test_probe_pure_genai_vs_hybrid() {
    let dir = create_temp_test_dir("pure_llm");
    let model_dir = dir.join("pure_llm");
    fs::create_dir_all(&model_dir).unwrap();

    let model_onnx = model_dir.join("model.onnx");
    let genai_config = model_dir.join("genai_config.json");

    fs::write(&model_onnx, b"dummy onnx backbone").unwrap();
    fs::write(&genai_config, b"{}").unwrap();

    let probe = probe_model_bundle(&model_onnx, None);
    assert_eq!(probe.paradigm, ModelParadigm::AutoregressiveGenAi);
    assert!(probe.has_genai_config);
    assert!(!probe.has_vision_encoder);
    assert!(!probe.has_audio_encoder);
}
