use crate::config::EngineKind;
use crate::engine::onnx_introspect::OnnxModelInspection;
use std::path::Path;
use tracing::{debug, info};

/// How a bundle was classified. `ambiguous` is the only case where
/// `ModelConfig::engine_kind` is allowed to decide.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelBundleProbe {
    pub kind: EngineKind,
    pub ambiguous: bool,
    pub has_genai_config: bool,
    pub genai_config_substantive: bool,
    pub has_vision_encoder: bool,
    pub has_audio_encoder: bool,
    pub detection_reason: String,
    pub used_config_override: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenaiConfigSignal {
    Absent,
    /// File exists but does not declare an autoregressive decoder loop.
    Thin,
    /// `model.decoder`, or a non-empty `model.type` plus generation settings.
    DecoderLoop,
}

/// Task used for built-in aliases. Shape rank alone is not a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTask {
    Audio,
    Embedding,
    Unknown,
}

impl std::fmt::Display for EngineKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineKind::OneShot => write!(f, "One-Shot (ONNX Runtime)"),
            EngineKind::Autoregressive => write!(f, "Autoregressive (refused)"),
        }
    }
}

pub fn engine_label(kind: EngineKind) -> &'static str {
    match kind {
        EngineKind::OneShot => "onnxruntime",
        EngineKind::Autoregressive => "autoregressive",
    }
}

/// `whisper-1` / `whisper` and `text-embedding-*` match a task, not "any model"
/// and not "any rank-2 output".
pub fn builtin_alias_matches(requested: &str, task: ModelTask) -> bool {
    if requested.eq_ignore_ascii_case("whisper-1") || requested.eq_ignore_ascii_case("whisper") {
        return task == ModelTask::Audio;
    }
    if requested.starts_with("text-embedding-") {
        return task == ModelTask::Embedding;
    }
    false
}

pub fn task_from_inspection(inspection: Option<&OnnxModelInspection>) -> ModelTask {
    let Some(inspection) = inspection else {
        return ModelTask::Unknown;
    };
    if inspection.is_audio_model {
        return ModelTask::Audio;
    }
    if inspection
        .outputs
        .iter()
        .any(|output| output.name == "sentence_embedding" || output.name == "last_hidden_state")
    {
        return ModelTask::Embedding;
    }
    ModelTask::Unknown
}

pub fn task_from_configured_alias(alias: Option<&str>) -> ModelTask {
    let Some(alias) = alias else {
        return ModelTask::Unknown;
    };
    if alias.eq_ignore_ascii_case("whisper-1") || alias.eq_ignore_ascii_case("whisper") {
        ModelTask::Audio
    } else if alias.starts_with("text-embedding-") {
        ModelTask::Embedding
    } else {
        ModelTask::Unknown
    }
}

/// Inspection wins. A configured alias supplies the task when the file cannot be read.
pub fn resolve_model_task(
    inspection: Option<&OnnxModelInspection>,
    alias: Option<&str>,
    has_audio_encoder: bool,
) -> ModelTask {
    if has_audio_encoder {
        return ModelTask::Audio;
    }
    match task_from_inspection(inspection) {
        ModelTask::Unknown => task_from_configured_alias(alias),
        task => task,
    }
}

fn read_genai_signal(model_dir: &Path) -> GenaiConfigSignal {
    let path = model_dir.join("genai_config.json");
    if !path.exists() {
        return GenaiConfigSignal::Absent;
    }
    let Ok(text) = std::fs::read_to_string(&path) else {
        return GenaiConfigSignal::Thin;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
        return GenaiConfigSignal::Thin;
    };
    let Some(model) = json.get("model").and_then(|value| value.as_object()) else {
        return GenaiConfigSignal::Thin;
    };
    let has_decoder = model
        .get("decoder")
        .map(|decoder| decoder.is_object())
        .unwrap_or(false);
    let type_name = model
        .get("type")
        .and_then(|value| value.as_str())
        .unwrap_or("")
        .trim();
    let has_generation =
        has_decoder || json.get("search").is_some() || model.get("context_length").is_some();
    if has_decoder || (!type_name.is_empty() && has_generation) {
        GenaiConfigSignal::DecoderLoop
    } else {
        GenaiConfigSignal::Thin
    }
}

fn paired_kv(inspection: &OnnxModelInspection) -> bool {
    let past = inspection.inputs.iter().any(|input| {
        let name = input.name.to_ascii_lowercase();
        name.starts_with("past_key_values") || name.starts_with("past_key.")
    });
    let present = inspection.outputs.iter().any(|output| {
        let name = output.name.to_ascii_lowercase();
        name.starts_with("present")
    });
    past && present
}

fn framework_marker(inspection: &OnnxModelInspection) -> Option<EngineKind> {
    let value = inspection
        .metadata_props
        .get("jetort.framework_type")
        .or_else(|| inspection.metadata_props.get("framework_type"))?;
    match value.to_ascii_lowercase().as_str() {
        "autoregressive" | "decoder_only" | "decoder-only" | "encoder_decoder"
        | "encoder-decoder" => Some(EngineKind::Autoregressive),
        "encoder_only" | "one_shot" | "oneshot" | "embedding" => Some(EngineKind::OneShot),
        _ => None,
    }
}

fn looks_like_embedding(inspection: &OnnxModelInspection) -> bool {
    inspection
        .outputs
        .iter()
        .any(|output| output.name == "sentence_embedding" || output.name == "last_hidden_state")
}

fn finish(
    kind: EngineKind,
    ambiguous: bool,
    reason: String,
    override_kind: Option<EngineKind>,
    has_genai_config: bool,
    substantive: bool,
    has_vision_encoder: bool,
    has_audio_encoder: bool,
) -> ModelBundleProbe {
    if ambiguous {
        if let Some(kind) = override_kind {
            info!(
                engine_kind = %kind,
                "Engine kind is ambiguous; using config engine_kind"
            );
            return ModelBundleProbe {
                kind,
                ambiguous: true,
                has_genai_config,
                genai_config_substantive: substantive,
                has_vision_encoder,
                has_audio_encoder,
                detection_reason: format!("{reason} Resolved by config engine_kind={kind}."),
                used_config_override: true,
            };
        }
    }
    ModelBundleProbe {
        kind,
        ambiguous,
        has_genai_config,
        genai_config_substantive: substantive,
        has_vision_encoder,
        has_audio_encoder,
        detection_reason: reason,
        used_config_override: false,
    }
}

/// Classifies a bundle as one-shot or autoregressive.
///
/// 1. Paired `past_key_values` / `present` tensors, or a trustworthy framework
///    marker (ONNX metadata, or a genai config that actually describes a decoder loop).
/// 2. A `genai_config.json` file by itself is not that marker. A thin file must
///    not take an embedding model.
/// 3. `engine_kind` in config applies only when those signals conflict.
///
/// Encoder-decoder bundles (substantive genai config plus a vision or audio
/// encoder file) are autoregressive. JetOrt refuses to load them.
pub fn probe_model_bundle(
    model_path: &Path,
    inspection: Option<&OnnxModelInspection>,
    engine_kind_override: Option<EngineKind>,
) -> ModelBundleProbe {
    let parent = model_path.parent();
    let (has_genai_config, has_vision_encoder, has_audio_encoder, signal) =
        if let Some(dir) = parent {
            let has_genai = dir.join("genai_config.json").exists();
            (
                has_genai,
                dir.join("vision_encoder.onnx").exists(),
                dir.join("audio_encoder.onnx").exists(),
                read_genai_signal(dir),
            )
        } else {
            (false, false, false, GenaiConfigSignal::Absent)
        };
    let substantive = signal == GenaiConfigSignal::DecoderLoop;

    let kv = inspection.map(paired_kv).unwrap_or(false);
    let marker = inspection.and_then(framework_marker);
    let embedding_graph = inspection.map(looks_like_embedding).unwrap_or(false);

    let flags = (
        has_genai_config,
        substantive,
        has_vision_encoder,
        has_audio_encoder,
    );

    if kv && marker == Some(EngineKind::OneShot) {
        debug!(path = %model_path.display(), "KV cache conflicts with one-shot framework marker");
        return finish(
            EngineKind::Autoregressive,
            false,
            "Graph has paired KV cache but metadata says one-shot.".to_string(),
            engine_kind_override,
            flags.0,
            flags.1,
            flags.2,
            flags.3,
        );
    }

    if kv {
        info!(path = %model_path.display(), "Paired past_key_values and present tensors");
        return finish(
            EngineKind::Autoregressive,
            false,
            "Graph contains paired past_key_values and present tensors.".to_string(),
            engine_kind_override,
            flags.0,
            flags.1,
            flags.2,
            flags.3,
        );
    }

    if marker == Some(EngineKind::Autoregressive) && embedding_graph {
        return finish(
            EngineKind::OneShot,
            true,
            "Metadata says autoregressive but the graph outputs embeddings.".to_string(),
            engine_kind_override,
            flags.0,
            flags.1,
            flags.2,
            flags.3,
        );
    }

    if marker == Some(EngineKind::Autoregressive) {
        info!(path = %model_path.display(), "Trustworthy autoregressive framework marker");
        return finish(
            EngineKind::Autoregressive,
            false,
            "ONNX metadata declares an autoregressive decoder.".to_string(),
            engine_kind_override,
            flags.0,
            flags.1,
            flags.2,
            flags.3,
        );
    }

    if marker == Some(EngineKind::OneShot) {
        return finish(
            EngineKind::OneShot,
            false,
            "ONNX metadata declares a one-shot model.".to_string(),
            engine_kind_override,
            flags.0,
            flags.1,
            flags.2,
            flags.3,
        );
    }

    // Encoder-decoder: sidecar encoder plus a genai config that describes a decoder.
    // A thin genai_config next to an encoder file is not enough.
    if (has_vision_encoder || has_audio_encoder) && substantive && !embedding_graph {
        info!(
            path = %model_path.display(),
            vision = has_vision_encoder,
            audio = has_audio_encoder,
            "Encoder-decoder bundle classified as autoregressive"
        );
        return finish(
            EngineKind::Autoregressive,
            false,
            "Encoder file plus a genai config that describes a decoder loop.".to_string(),
            engine_kind_override,
            flags.0,
            flags.1,
            flags.2,
            flags.3,
        );
    }

    // A decoder-loop genai config without an embedding graph is a trustworthy marker.
    // File existence alone (Thin) is not.
    if substantive && !embedding_graph {
        info!(path = %model_path.display(), "genai_config describes a decoder loop");
        return finish(
            EngineKind::Autoregressive,
            false,
            "genai_config.json describes an autoregressive decoder loop.".to_string(),
            engine_kind_override,
            flags.0,
            flags.1,
            flags.2,
            flags.3,
        );
    }

    if substantive && embedding_graph {
        return finish(
            EngineKind::OneShot,
            true,
            "genai_config describes a decoder but the graph outputs embeddings.".to_string(),
            engine_kind_override,
            flags.0,
            flags.1,
            flags.2,
            flags.3,
        );
    }

    debug!(
        path = %model_path.display(),
        genai = ?signal_name(signal),
        "No autoregressive graph or framework marker"
    );
    let reason = if signal == GenaiConfigSignal::Thin {
        "genai_config.json is present but does not describe a decoder loop; classified as one-shot."
            .to_string()
    } else {
        "No autoregressive indicators; classified as one-shot.".to_string()
    };
    finish(
        EngineKind::OneShot,
        false,
        reason,
        engine_kind_override,
        flags.0,
        flags.1,
        flags.2,
        flags.3,
    )
}

fn signal_name(signal: GenaiConfigSignal) -> &'static str {
    match signal {
        GenaiConfigSignal::Absent => "absent",
        GenaiConfigSignal::Thin => "thin",
        GenaiConfigSignal::DecoderLoop => "decoder",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::onnx_introspect::{OnnxDim, OnnxInputTensorInfo, QuantizationKind};
    use std::collections::{HashMap, HashSet};

    fn inspection(
        metadata: HashMap<String, String>,
        inputs: Vec<&str>,
        outputs: Vec<&str>,
    ) -> OnnxModelInspection {
        OnnxModelInspection {
            ir_version: Some(8),
            opset_version: Some(19),
            producer_name: Some("test".to_string()),
            producer_version: None,
            metadata_props: metadata,
            inputs: inputs
                .into_iter()
                .map(|name| OnnxInputTensorInfo {
                    name: name.to_string(),
                    elem_type: 7,
                    dims: vec![OnnxDim::Fixed(1), OnnxDim::Dynamic("seq".to_string())],
                })
                .collect(),
            outputs: outputs
                .into_iter()
                .map(|name| OnnxInputTensorInfo {
                    name: name.to_string(),
                    elem_type: 1,
                    dims: vec![OnnxDim::Fixed(1), OnnxDim::Fixed(384)],
                })
                .collect(),
            op_types: HashSet::new(),
            quantization_kind: QuantizationKind::None,
            has_uppercase_einsum: false,
            is_audio_model: false,
            recommended_provider: "CPU".to_string(),
            candidate_providers: vec!["CPU".to_string()],
            trt_compilation_feasible: true,
            trt_skip_reason: None,
            synthesized_profiles: None,
        }
    }

    fn temp_bundle(prefix: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("jetort-probe-{prefix}-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn arch_probe_oneshot_default() {
        let insp = inspection(
            HashMap::new(),
            vec!["input_ids", "attention_mask"],
            vec!["sentence_embedding"],
        );
        let probe = probe_model_bundle(Path::new("/dummy/model.onnx"), Some(&insp), None);
        assert_eq!(probe.kind, EngineKind::OneShot);
        assert!(!probe.ambiguous);
    }

    #[test]
    fn arch_probe_metadata_autoregressive() {
        let mut meta = HashMap::new();
        meta.insert(
            "jetort.framework_type".to_string(),
            "autoregressive".to_string(),
        );
        let insp = inspection(meta, vec!["input_ids"], vec!["logits"]);
        let probe = probe_model_bundle(Path::new("/dummy/model.onnx"), Some(&insp), None);
        assert_eq!(probe.kind, EngineKind::Autoregressive);
    }

    #[test]
    fn arch_probe_encoder_decoder_metadata() {
        let mut meta = HashMap::new();
        meta.insert("framework_type".to_string(), "encoder_decoder".to_string());
        let insp = inspection(meta, vec!["input_ids"], vec!["logits"]);
        let probe = probe_model_bundle(Path::new("/dummy/model.onnx"), Some(&insp), None);
        assert_eq!(probe.kind, EngineKind::Autoregressive);
    }

    #[test]
    fn arch_probe_kv_cache_tensors() {
        let insp = inspection(
            HashMap::new(),
            vec![
                "input_ids",
                "past_key_values.0.key",
                "past_key_values.0.value",
            ],
            vec!["logits", "present.0.key", "present.0.value"],
        );
        let probe = probe_model_bundle(Path::new("/dummy/model.onnx"), Some(&insp), None);
        assert_eq!(probe.kind, EngineKind::Autoregressive);
    }

    #[test]
    fn arch_genai_config_file_alone_is_not_autoregressive() {
        let dir = temp_bundle("thin");
        let model = dir.join("model.onnx");
        std::fs::write(&model, b"not a real onnx").unwrap();
        std::fs::write(dir.join("genai_config.json"), b"{}").unwrap();
        let insp = inspection(
            HashMap::new(),
            vec!["input_ids", "attention_mask"],
            vec!["sentence_embedding"],
        );
        let probe = probe_model_bundle(&model, Some(&insp), None);
        assert_eq!(probe.kind, EngineKind::OneShot);
        assert!(probe.has_genai_config);
        assert!(!probe.genai_config_substantive);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn arch_thin_genai_plus_encoder_file_does_not_steal() {
        let dir = temp_bundle("encoder-thin");
        let model = dir.join("model.onnx");
        std::fs::write(&model, b"onnx").unwrap();
        std::fs::write(dir.join("genai_config.json"), b"{}").unwrap();
        std::fs::write(dir.join("vision_encoder.onnx"), b"encoder").unwrap();
        let insp = inspection(
            HashMap::new(),
            vec!["input_ids"],
            vec!["sentence_embedding"],
        );
        let probe = probe_model_bundle(&model, Some(&insp), None);
        assert_eq!(probe.kind, EngineKind::OneShot);
        assert!(probe.has_vision_encoder);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn arch_encoder_decoder_bundle_is_autoregressive() {
        let dir = temp_bundle("encoder-decoder");
        let model = dir.join("model.onnx");
        std::fs::write(&model, b"onnx").unwrap();
        std::fs::write(
            dir.join("genai_config.json"),
            r#"{"model":{"type":"lfm2_audio","decoder":{"num_hidden_layers":1}}}"#,
        )
        .unwrap();
        std::fs::write(dir.join("audio_encoder.onnx"), b"encoder").unwrap();
        let probe = probe_model_bundle(&model, None, None);
        assert_eq!(probe.kind, EngineKind::Autoregressive);
        assert!(probe.has_audio_encoder);
        assert!(probe.genai_config_substantive);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn arch_override_only_when_ambiguous() {
        let mut meta = HashMap::new();
        meta.insert(
            "jetort.framework_type".to_string(),
            "autoregressive".to_string(),
        );
        let conflicting = inspection(meta, vec!["input_ids"], vec!["sentence_embedding"]);
        let forced = probe_model_bundle(
            Path::new("/dummy/model.onnx"),
            Some(&conflicting),
            Some(EngineKind::Autoregressive),
        );
        assert!(forced.ambiguous);
        assert!(forced.used_config_override);
        assert_eq!(forced.kind, EngineKind::Autoregressive);

        let decisive = inspection(
            HashMap::new(),
            vec!["input_ids", "past_key_values.0.key"],
            vec!["logits", "present.0.key"],
        );
        let ignored = probe_model_bundle(
            Path::new("/dummy/model.onnx"),
            Some(&decisive),
            Some(EngineKind::OneShot),
        );
        assert_eq!(ignored.kind, EngineKind::Autoregressive);
        assert!(!ignored.used_config_override);
    }

    #[test]
    fn arch_alias_matches_task_not_rank() {
        assert!(builtin_alias_matches(
            "text-embedding-ada-002",
            ModelTask::Embedding
        ));
        assert!(!builtin_alias_matches(
            "text-embedding-3-small",
            ModelTask::Audio
        ));
        assert!(!builtin_alias_matches(
            "text-embedding-3-small",
            ModelTask::Unknown
        ));
        assert!(builtin_alias_matches("whisper-1", ModelTask::Audio));
        assert!(!builtin_alias_matches("whisper-1", ModelTask::Embedding));
        assert!(!builtin_alias_matches("whisper", ModelTask::Unknown));

        let mut embedding = inspection(
            HashMap::new(),
            vec!["input_ids"],
            vec!["sentence_embedding"],
        );
        embedding.outputs[0].dims = vec![OnnxDim::Fixed(1), OnnxDim::Fixed(384)];
        assert_eq!(task_from_inspection(Some(&embedding)), ModelTask::Embedding);
        let logits = inspection(HashMap::new(), vec!["input_ids"], vec!["logits"]);
        assert_eq!(task_from_inspection(Some(&logits)), ModelTask::Unknown);
        assert_eq!(
            task_from_configured_alias(Some("text-embedding-3-small")),
            ModelTask::Embedding
        );
    }
}
