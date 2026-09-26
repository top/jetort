use crate::engine::onnx_introspect::OnnxModelInspection;
use std::path::Path;
use tracing::{debug, info};

/// Execution paradigm of a model in JetOrt
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelParadigm {
    /// Pure non-autoregressive forward model (Embeddings, Reranker, ASR, Vision, Raw Tensors)
    OneShot,
    /// Autoregressive generative model (LLM / VLM with KV-cache / token-by-token generation)
    AutoregressiveGenAi,
    /// Hybrid multimodal pipeline (One-Shot Vision/Audio Encoder + Autoregressive LLM Backbone)
    Hybrid,
}

impl std::fmt::Display for ModelParadigm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelParadigm::OneShot => write!(f, "One-Shot Forward (Native ORT)"),
            ModelParadigm::AutoregressiveGenAi => write!(f, "Autoregressive GenAI (ORT-GenAI)"),
            ModelParadigm::Hybrid => write!(f, "Hybrid Multimodal (One-Shot Encoder + GenAI Backbone)"),
        }
    }
}

/// Metadata and structural inspection result of a model bundle
#[derive(Debug, Clone)]
pub struct ModelBundleProbe {
    pub paradigm: ModelParadigm,
    pub has_genai_config: bool,
    pub has_vision_encoder: bool,
    pub has_audio_encoder: bool,
    pub framework_type: Option<String>,
    pub detection_reason: String,
}

/// Probes a model file and its companion bundle directory to classify its paradigm.
///
/// Inspection hierarchy:
/// 1. Bundle signature: presence of `genai_config.json` + `vision_encoder.onnx`/`audio_encoder.onnx` -> Hybrid
/// 2. Bundle signature: presence of `genai_config.json` in the model directory -> GenAI
/// 3. ONNX Protobuf `metadata_props`: `jetort.framework_type` or `framework_type` == "autoregressive" -> GenAI
/// 4. ONNX Graph I/O signatures: presence of `past_key_values.*` inputs and `present.*` outputs -> GenAI
/// 5. Default: One-Shot Forward model
pub fn probe_model_bundle(
    model_path: &Path,
    inspection: Option<&OnnxModelInspection>,
) -> ModelBundleProbe {
    let parent_dir = model_path.parent();

    // 1. Check directory bundle signatures
    let (has_genai_config, has_vision_encoder, has_audio_encoder) = if let Some(dir) = parent_dir {
        let genai_cfg = dir.join("genai_config.json").exists();
        let vision_enc = dir.join("vision_encoder.onnx").exists();
        let audio_enc = dir.join("audio_encoder.onnx").exists();
        (genai_cfg, vision_enc, audio_enc)
    } else {
        (false, false, false)
    };

    if has_genai_config && (has_vision_encoder || has_audio_encoder) {
        info!(
            path = %model_path.display(),
            vision = has_vision_encoder,
            audio = has_audio_encoder,
            "Detected multimodal hybrid bundle (encoder + genai_config.json): classified as Hybrid"
        );
        return ModelBundleProbe {
            paradigm: ModelParadigm::Hybrid,
            has_genai_config: true,
            has_vision_encoder,
            has_audio_encoder,
            framework_type: Some("hybrid_multimodal".to_string()),
            detection_reason: "Found multimodal encoder (vision/audio) and genai_config.json in bundle".to_string(),
        };
    }

    if has_genai_config {
        info!(
            path = %model_path.display(),
            "Detected 'genai_config.json' in model directory: classified as AutoregressiveGenAi"
        );
        return ModelBundleProbe {
            paradigm: ModelParadigm::AutoregressiveGenAi,
            has_genai_config: true,
            has_vision_encoder,
            has_audio_encoder,
            framework_type: Some("autoregressive".to_string()),
            detection_reason: "Found standard genai_config.json in model bundle directory".to_string(),
        };
    }

    // 2. Check metadata_props if ONNX header was inspected
    if let Some(insp) = inspection {
        if let Some(fw_type) = insp
            .metadata_props
            .get("jetort.framework_type")
            .or_else(|| insp.metadata_props.get("framework_type"))
        {
            if fw_type.eq_ignore_ascii_case("autoregressive")
                || fw_type.eq_ignore_ascii_case("decoder_only")
            {
                info!(
                    path = %model_path.display(),
                    framework_type = %fw_type,
                    "Detected autoregressive framework_type in ONNX metadata_props: classified as AutoregressiveGenAi"
                );
                return ModelBundleProbe {
                    paradigm: ModelParadigm::AutoregressiveGenAi,
                    has_genai_config,
                    has_vision_encoder,
                    has_audio_encoder,
                    framework_type: Some(fw_type.clone()),
                    detection_reason: format!("ONNX metadata_props declares framework_type='{}'", fw_type),
                };
            } else if fw_type.eq_ignore_ascii_case("encoder_only")
                || fw_type.eq_ignore_ascii_case("one_shot")
            {
                debug!(
                    path = %model_path.display(),
                    framework_type = %fw_type,
                    "Detected encoder_only in ONNX metadata_props: classified as OneShot"
                );
                return ModelBundleProbe {
                    paradigm: ModelParadigm::OneShot,
                    has_genai_config,
                    has_vision_encoder,
                    has_audio_encoder,
                    framework_type: Some(fw_type.clone()),
                    detection_reason: format!("ONNX metadata_props declares framework_type='{}'", fw_type),
                };
            }
        }

        // 3. Check graph recurrent tensor signature (past_key_values / present)
        let has_past_kv = insp.inputs.iter().any(|i| {
            i.name.starts_with("past_key_values")
                || i.name.contains(".key")
                || i.name.contains(".value")
        });
        let has_present_kv = insp.outputs.iter().any(|o| {
            o.name.starts_with("present")
                || o.name.contains(".key")
                || o.name.contains(".value")
        });

        if has_past_kv && has_present_kv {
            info!(
                path = %model_path.display(),
                "Detected past_key_values inputs and present outputs in computational graph: classified as AutoregressiveGenAi"
            );
            return ModelBundleProbe {
                paradigm: ModelParadigm::AutoregressiveGenAi,
                has_genai_config,
                has_vision_encoder,
                has_audio_encoder,
                framework_type: Some("autoregressive".to_string()),
                detection_reason: "Graph contains paired past_key_values and present KV-cache tensors".to_string(),
            };
        }
    }

    // 4. Default: One-Shot Forward model
    debug!(
        path = %model_path.display(),
        "No autoregressive indicators found: classified as OneShot"
    );
    ModelBundleProbe {
        paradigm: ModelParadigm::OneShot,
        has_genai_config,
        has_vision_encoder,
        has_audio_encoder,
        framework_type: None,
        detection_reason: "Defaulting to One-Shot forward pass (no autoregressive indicators found)".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::onnx_introspect::{OnnxDim, OnnxInputTensorInfo, QuantizationKind};
    use std::collections::{HashMap, HashSet};

    fn make_dummy_inspection(
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
                    elem_type: 7, // INT64
                    dims: vec![OnnxDim::Fixed(1), OnnxDim::Dynamic("seq".to_string())],
                })
                .collect(),
            outputs: outputs
                .into_iter()
                .map(|name| OnnxInputTensorInfo {
                    name: name.to_string(),
                    elem_type: 1, // FLOAT
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

    #[test]
    fn test_probe_oneshot_default() {
        let insp = make_dummy_inspection(
            HashMap::new(),
            vec!["input_ids", "attention_mask"],
            vec!["sentence_embedding"],
        );
        let probe = probe_model_bundle(Path::new("/dummy/model.onnx"), Some(&insp));
        assert_eq!(probe.paradigm, ModelParadigm::OneShot);
    }

    #[test]
    fn test_probe_metadata_autoregressive() {
        let mut meta = HashMap::new();
        meta.insert("jetort.framework_type".to_string(), "autoregressive".to_string());
        let insp = make_dummy_inspection(
            meta,
            vec!["input_ids"],
            vec!["logits"],
        );
        let probe = probe_model_bundle(Path::new("/dummy/model.onnx"), Some(&insp));
        assert_eq!(probe.paradigm, ModelParadigm::AutoregressiveGenAi);
    }

    #[test]
    fn test_probe_kv_cache_tensors() {
        let insp = make_dummy_inspection(
            HashMap::new(),
            vec!["input_ids", "past_key_values.0.key", "past_key_values.0.value"],
            vec!["logits", "present.0.key", "present.0.value"],
        );
        let probe = probe_model_bundle(Path::new("/dummy/model.onnx"), Some(&insp));
        assert_eq!(probe.paradigm, ModelParadigm::AutoregressiveGenAi);
    }
}
