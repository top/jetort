use jetort::config::EngineKind;
use jetort::engine::header_prober::probe_model_bundle;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn create_temp_test_dir(prefix: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("jetort-engine-{prefix}-{nanos}"));
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn arch_thin_genai_config_with_encoder_is_not_autoregressive() {
    let dir = create_temp_test_dir("vision");
    fs::write(dir.join("model.onnx"), b"dummy onnx backbone").unwrap();
    fs::write(dir.join("genai_config.json"), b"{}").unwrap();
    fs::write(dir.join("vision_encoder.onnx"), b"dummy vision encoder").unwrap();

    let probe = probe_model_bundle(&dir.join("model.onnx"), None, None);
    assert_eq!(probe.kind, EngineKind::OneShot);
    assert!(probe.has_genai_config);
    assert!(probe.has_vision_encoder);
    assert!(!probe.genai_config_substantive);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn arch_encoder_decoder_is_autoregressive() {
    let dir = create_temp_test_dir("audio");
    fs::write(dir.join("model.onnx"), b"dummy onnx backbone").unwrap();
    fs::write(
        dir.join("genai_config.json"),
        r#"{"model":{"type":"lfm2_audio","decoder":{"num_hidden_layers":2}}}"#,
    )
    .unwrap();
    fs::write(dir.join("audio_encoder.onnx"), b"dummy audio encoder").unwrap();

    let probe = probe_model_bundle(&dir.join("model.onnx"), None, None);
    assert_eq!(probe.kind, EngineKind::Autoregressive);
    assert!(probe.has_audio_encoder);
    assert!(probe.genai_config_substantive);
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn arch_genai_config_alone_does_not_force_autoregressive() {
    let dir = create_temp_test_dir("pure");
    fs::write(dir.join("model.onnx"), b"dummy onnx backbone").unwrap();
    fs::write(dir.join("genai_config.json"), b"{}").unwrap();

    let probe = probe_model_bundle(&dir.join("model.onnx"), None, None);
    assert_eq!(probe.kind, EngineKind::OneShot);
    assert!(probe.has_genai_config);
    assert!(!probe.has_vision_encoder);
    assert!(!probe.has_audio_encoder);
    let _ = fs::remove_dir_all(dir);
}
