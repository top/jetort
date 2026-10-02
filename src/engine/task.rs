use std::path::Path;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use crate::engine::onnx_introspect::{OnnxDim, OnnxInputTensorInfo, OnnxOutputTensorInfo};

/// Canonical task/paradigm for models served by JetOrt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelTask {
    Auto,
    Decision,
    Rerank,
    Embedding,
    Classification,
    Asr,
    Custom,
}

impl ModelTask {
    pub const AUDIO: ModelTask = ModelTask::Asr;
    pub const UNKNOWN: ModelTask = ModelTask::Custom;
    #[allow(non_upper_case_globals)]
    pub const Audio: ModelTask = ModelTask::Asr;
    #[allow(non_upper_case_globals)]
    pub const Unknown: ModelTask = ModelTask::Custom;

    pub fn as_str(&self) -> &'static str {
        match self {
            ModelTask::Auto => "auto",
            ModelTask::Decision => "decision",
            ModelTask::Rerank => "rerank",
            ModelTask::Embedding => "embedding",
            ModelTask::Classification => "classification",
            ModelTask::Asr => "asr",
            ModelTask::Custom => "custom",
        }
    }

    pub fn from_str_tolerant(s: &str) -> Self {
        let lower = s.trim().to_ascii_lowercase().replace(['-', '_'], "");
        match lower.as_str() {
            "decision" | "decider" | "systemone" | "system1" | "choice" => ModelTask::Decision,
            "rerank" | "reranker" | "crossencoder" => ModelTask::Rerank,
            "embedding" | "embeddings" | "featureextraction" => ModelTask::Embedding,
            "classification" | "classifier" | "sequenceclassification" | "textclassification" => {
                ModelTask::Classification
            }
            "asr" | "audio" | "speech" | "transcription" => ModelTask::Asr,
            "custom" | "raw" | "tensor" => ModelTask::Custom,
            "auto" | "default" | "" => ModelTask::Auto,
            _ => ModelTask::Auto,
        }
    }
}

impl Serialize for ModelTask {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ModelTask {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(ModelTask::from_str_tolerant(&s))
    }
}

impl std::fmt::Display for ModelTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Detect task purely from companion files in the model folder.
/// Isolated for easy maintenance, testability, and future updates.
pub fn detect_task_from_path(model_dir: &Path) -> Option<ModelTask> {
    let dir = if model_dir.is_file() {
        model_dir.parent()?
    } else {
        model_dir
    };

    // 1. deployment_manifest.json (explicit packaging declaration)
    if let Ok(manifest_text) = std::fs::read_to_string(dir.join("deployment_manifest.json")) {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&manifest_text) {
            if let Some(task_str) = val.get("task").and_then(|t| t.as_str()) {
                let parsed = ModelTask::from_str_tolerant(task_str);
                if parsed != ModelTask::Auto {
                    return Some(parsed);
                }
            }
        }
    }

    // 2. OpenJev / Decision model config files
    if dir.join("rl_agent_config.json").exists()
        || dir.join("source").join("rl_agent_config.json").exists()
    {
        return Some(ModelTask::Decision);
    }
    if let Ok(prov_text) = std::fs::read_to_string(dir.join("provenance.json")) {
        if prov_text.contains("open-jev") || prov_text.contains("typed:choice") {
            return Some(ModelTask::Decision);
        }
    }

    // 3. Embedding model companion files
    if dir.join("1_Pooling").is_dir()
        || dir.join("sentence_bert_config.json").exists()
        || dir.join("modules.json").is_file()
            && std::fs::read_to_string(dir.join("modules.json"))
                .map(|t| t.contains("SentenceTransformer"))
                .unwrap_or(false)
    {
        return Some(ModelTask::Embedding);
    }

    // 4. Reranker prompt template
    if dir.join("chat_template.jinja").exists() {
        if let Ok(tmpl) = std::fs::read_to_string(dir.join("chat_template.jinja")) {
            let lower = tmpl.to_ascii_lowercase();
            if lower.contains("document") && lower.contains("query") {
                return Some(ModelTask::Rerank);
            }
        }
    }

    // 5. Hugging Face config.json
    if let Ok(cfg_text) = std::fs::read_to_string(dir.join("config.json")) {
        if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&cfg_text) {
            if let Some(model_type) = cfg.get("model_type").and_then(|v| v.as_str()) {
                if model_type.eq_ignore_ascii_case("gliclass") {
                    return Some(ModelTask::Decision);
                }
            }
            if let Some(archs) = cfg.get("architectures").and_then(|v| v.as_array()) {
                for a in archs {
                    if let Some(s) = a.as_str() {
                        if s.contains("ForSequenceClassification") {
                            // Check if configured as reranker
                            if dir.join("chat_template.jinja").exists() {
                                return Some(ModelTask::Rerank);
                            }
                            return Some(ModelTask::Classification);
                        }
                    }
                }
            }
        }
    }

    None
}

/// Detect task from ONNX computational graph tensor signatures (Structural DNA).
pub fn detect_task_from_tensors(
    inputs: &[OnnxInputTensorInfo],
    outputs: &[OnnxOutputTensorInfo],
) -> Option<ModelTask> {
    // 1. Decision / Choice models: distinctive marker_pos, marker_mask, qtype, or options dimension
    let has_decision_inputs = inputs.iter().any(|inp| {
        let name = inp.name.to_ascii_lowercase();
        name == "marker_pos"
            || name == "marker_mask"
            || name == "qtype"
            || inp.dims.iter().any(|d| match d {
                OnnxDim::Dynamic(s) => {
                    let s_low = s.to_ascii_lowercase();
                    s_low.contains("option") || s_low.contains("choice") || s_low.contains("candidate")
                }
                _ => false,
            })
    });
    let has_decision_outputs = outputs.iter().any(|out| {
        out.name.to_ascii_lowercase().contains("option")
            || out.dims.iter().any(|d| match d {
                OnnxDim::Dynamic(s) => s.to_ascii_lowercase().contains("option"),
                _ => false,
            })
    });
    if has_decision_inputs || has_decision_outputs {
        return Some(ModelTask::Decision);
    }

    // 2. ASR / Audio models
    let is_audio = inputs.iter().any(|inp| {
        let name = inp.name.to_ascii_lowercase();
        name == "features"
            || name == "speech"
            || name == "waveform"
            || (inp.dims.len() == 3
                && match inp.dims.last() {
                    Some(OnnxDim::Fixed(c)) => *c == 80 || *c == 128 || *c == 64,
                    _ => false,
                })
    });
    if is_audio {
        return Some(ModelTask::Asr);
    }

    // 3. Embedding models: distinctive pooled or token representation outputs
    let is_embedding = outputs.iter().any(|out| {
        out.name == "sentence_embedding"
            || out.name == "last_hidden_state"
            || out.name == "pooler_output"
    });
    if is_embedding {
        return Some(ModelTask::Embedding);
    }

    // 4. Reranker models: relevance_score or scalar score output
    let is_reranker = outputs.iter().any(|out| {
        let name = out.name.to_ascii_lowercase();
        name == "relevance_score" || name == "scores" || name == "score"
    });
    if is_reranker {
        return Some(ModelTask::Rerank);
    }

    None
}

/// Unified task resolver:
/// 1. User explicit config (task: "decision", "rerank", etc.) has absolute priority.
/// 2. Computational graph DNA (structural tensor fingerprinting).
/// 3. Companion files in the model path (manifests, configs, templates).
/// 4. Model alias heuristic (e.g. "whisper*", "text-embedding*").
/// 5. Output shape heuristic (standard sequence classification vs custom).
pub fn resolve_task(
    configured_task: Option<ModelTask>,
    model_path: Option<&Path>,
    inputs: &[OnnxInputTensorInfo],
    outputs: &[OnnxOutputTensorInfo],
    alias: Option<&str>,
) -> ModelTask {
    // 1. Explicit configuration wins unconditionally
    if let Some(task) = configured_task {
        if task != ModelTask::Auto {
            return task;
        }
    }

    // 2. Graph tensor signature (strongest intrinsic signal)
    if let Some(task) = detect_task_from_tensors(inputs, outputs) {
        return task;
    }

    // 3. Companion files
    if let Some(path) = model_path {
        if let Some(task) = detect_task_from_path(path) {
            return task;
        }
    }

    // 4. Model alias
    if let Some(alias) = alias {
        let low = alias.to_ascii_lowercase();
        if low.contains("whisper") || low.contains("sensevoice") || low.contains("paraformer") {
            return ModelTask::Asr;
        }
        if low.contains("embedding") || low.starts_with("text-embedding-") {
            return ModelTask::Embedding;
        }
        if low.contains("rerank") {
            return ModelTask::Rerank;
        }
        if low.contains("decision") || low.contains("decider") || low.contains("julia") || low.contains("laya") {
            return ModelTask::Decision;
        }
    }

    // 5. If outputs have logits with [batch, seq] text inputs
    let has_logits = outputs.iter().any(|o| o.name == "logits");
    let has_text_inputs = inputs.iter().any(|i| i.name == "input_ids");
    if has_logits && has_text_inputs {
        // Standard text sequence classification
        return ModelTask::Classification;
    }

    ModelTask::Custom
}

/// Returns the primary endpoints exposed for a given ModelTask.
pub fn endpoints_for_task(task: ModelTask) -> Vec<String> {
    match task {
        ModelTask::Decision => vec!["/v1/decision".to_string(), "/v1/systemone".to_string()],
        ModelTask::Rerank => vec!["/v1/rerank".to_string()],
        ModelTask::Embedding => vec!["/v1/embeddings".to_string()],
        ModelTask::Classification => vec!["/v1/classify".to_string()],
        ModelTask::Asr => vec!["/v1/audio/transcriptions".to_string()],
        ModelTask::Auto | ModelTask::Custom => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_task_aliases() {
        assert_eq!(ModelTask::from_str_tolerant("decision"), ModelTask::Decision);
        assert_eq!(ModelTask::from_str_tolerant("decider"), ModelTask::Decision);
        assert_eq!(ModelTask::from_str_tolerant("systemone"), ModelTask::Decision);
        assert_eq!(ModelTask::from_str_tolerant("system_one"), ModelTask::Decision);
        assert_eq!(ModelTask::from_str_tolerant("rerank"), ModelTask::Rerank);
        assert_eq!(ModelTask::from_str_tolerant("reranker"), ModelTask::Rerank);
        assert_eq!(ModelTask::from_str_tolerant("embedding"), ModelTask::Embedding);
        assert_eq!(ModelTask::from_str_tolerant("feature_extraction"), ModelTask::Embedding);
        assert_eq!(ModelTask::from_str_tolerant("classification"), ModelTask::Classification);
        assert_eq!(ModelTask::from_str_tolerant("sequence-classification"), ModelTask::Classification);
        assert_eq!(ModelTask::from_str_tolerant("asr"), ModelTask::Asr);
        assert_eq!(ModelTask::from_str_tolerant("audio"), ModelTask::Asr);
        assert_eq!(ModelTask::from_str_tolerant("auto"), ModelTask::Auto);
    }

    #[test]
    fn detects_decision_from_tensors() {
        let inputs = vec![
            OnnxInputTensorInfo {
                name: "input_ids".into(),
                elem_type: 7,
                dims: vec![OnnxDim::Dynamic("batch".into()), OnnxDim::Dynamic("tokens".into())],
            },
            OnnxInputTensorInfo {
                name: "marker_pos".into(),
                elem_type: 7,
                dims: vec![OnnxDim::Dynamic("batch".into()), OnnxDim::Dynamic("options".into())],
            },
            OnnxInputTensorInfo {
                name: "qtype".into(),
                elem_type: 7,
                dims: vec![OnnxDim::Dynamic("batch".into())],
            },
        ];
        let outputs = vec![OnnxOutputTensorInfo {
            name: "logits".into(),
            elem_type: 1,
            dims: vec![OnnxDim::Dynamic("batch".into()), OnnxDim::Dynamic("options".into())],
        }];

        let task = detect_task_from_tensors(&inputs, &outputs);
        assert_eq!(task, Some(ModelTask::Decision));
    }
}
