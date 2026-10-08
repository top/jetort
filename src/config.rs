use crate::error::AppError;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Classification of a model graph. Autoregressive models are refused at load.
/// Set `engine_kind` only when automatic detection cannot decide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EngineKind {
    #[serde(rename = "oneshot", alias = "one_shot", alias = "one-shot")]
    OneShot,
    #[serde(rename = "autoregressive")]
    Autoregressive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,

    #[serde(default = "default_port")]
    pub port: u16,

    #[serde(default = "default_inference_threads")]
    pub inference_threads: usize,

    /// Log filter used when RUST_LOG is not set.
    #[serde(default = "default_log_level")]
    pub log_level: String,

    /// Optional path to libonnxruntime.so (if omitted, auto-discovered from Python/system)
    #[serde(default)]
    pub ort_dylib_path: Option<String>,
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_inference_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn default_log_level() -> String {
    "info".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub name: String,

    /// Explicit local path to ONNX model or model directory (can be relative or absolute)
    #[serde(
        default,
        alias = "model_path",
        alias = "model-path",
        alias = "dir",
        alias = "model_dir"
    )]
    pub path: Option<String>,

    /// Hugging Face Hub repository identifier (e.g. "BAAI/bge-small-en-v1.5")
    #[serde(default, alias = "hf-repo")]
    pub hf_repo: Option<String>,

    /// Specific ONNX file name within the HF repository or local directory (e.g. "model_int8.onnx", "model.onnx")
    #[serde(
        default,
        alias = "hf-file",
        alias = "file",
        alias = "model_file",
        alias = "model-file",
        alias = "onnx_file",
        alias = "onnx-file"
    )]
    pub hf_file: Option<String>,

    /// Optional llama.cpp-style public model alias.
    #[serde(default)]
    pub alias: Option<String>,

    /// Explicit task/paradigm ("auto", "decision", "rerank", "embedding", "classification", "asr")
    #[serde(default, alias = "type")]
    pub task: Option<crate::engine::task::ModelTask>,

    /// Optional decision-model protocol. This is intentionally configured in
    /// config.yaml rather than discovered from files beside the ONNX graph.
    /// Supported values are contract families such as "gliclass",
    /// "openjev", "candidate_logits", and "d1".
    #[serde(default, alias = "decision-contract", alias = "decision_type")]
    pub decision_contract: Option<crate::engine::task::DecisionContract>,

    /// Load this model during startup after all TensorRT cache compilation completes.
    #[serde(default = "default_true")]
    pub load_on_startup: bool,

    /// Path to tokenizer or vocabulary file (optional, automatically inferred from HF cache)
    #[serde(default, alias = "tokenizer", alias = "vocab_path", alias = "vocab")]
    pub tokenizer_path: Option<String>,

    /// Execution providers / runtime list. Defaults to ["auto"]
    #[serde(default = "default_providers", alias = "runtime")]
    pub providers: Vec<String>,

    /// Context window size (llama.cpp style n_ctx, ctx_size). If None or 0, auto-detected from model metadata.
    #[serde(default, rename = "ctx_size")]
    pub n_ctx: Option<usize>,

    /// Prompt processing batch size (llama.cpp style n_batch). Defaults to 512.
    #[serde(default = "default_n_batch", rename = "batch_size")]
    pub n_batch: usize,

    /// Optional sequence length padding/truncation (e.g. 512 for static sequence models)
    #[serde(
        default,
        alias = "max_seq_len",
        alias = "max_length",
        alias = "pad_to_length"
    )]
    pub pad_seq_len: Option<usize>,

    /// Maximum micro-batch size for inference (defaults to 4 to prevent GPU memory spikes)
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    #[serde(default)]
    pub trt_cache_dir: Option<String>,

    #[serde(default)]
    pub trt_profile_min_shapes: Option<String>,

    #[serde(default)]
    pub trt_profile_opt_shapes: Option<String>,

    #[serde(default)]
    pub trt_profile_max_shapes: Option<String>,

    #[serde(default = "default_intra_threads")]
    pub intra_threads: usize,

    #[serde(default = "default_inter_threads")]
    pub inter_threads: usize,

    /// Optional limit on GPU device memory arena in MB for CUDA EP (e.g. 1500 for 1.5GB)
    #[serde(
        default,
        alias = "gpu_mem_limit",
        alias = "gpu_mem_limit_mb",
        alias = "cuda_memory_limit"
    )]
    pub cuda_memory_limit_mb: Option<usize>,

    /// Maximum workspace size in MB for TensorRT builder (defaults to 1024 MB)
    #[serde(
        default = "default_trt_max_workspace_mb",
        alias = "trt_max_workspace",
        alias = "trt_workspace_mb"
    )]
    pub trt_max_workspace_mb: Option<usize>,

    /// Enable memory pattern optimization (default: false to prevent memory leaks on dynamic shapes)
    #[serde(default = "default_enable_mem_pattern")]
    pub enable_mem_pattern: bool,

    /// Enable TensorRT context memory sharing across models and subgraphs (defaults to true)
    #[serde(default = "default_context_memory_sharing")]
    pub context_memory_sharing: bool,

    /// Concurrency limit for this model session (defaults to 1 to match Mutex<Session>)
    #[serde(default)]
    pub concurrency_limit: Option<usize>,

    /// Size of the concurrent session pool (defaults to 1 for edge/Jetson memory safety)
    #[serde(
        default = "default_session_pool_size",
        alias = "pool_size",
        alias = "session_pool_size"
    )]
    pub session_pool_size: usize,

    /// llama.cpp-style default sampling temperature for this model. Applied when a request
    /// doesn't specify its own `temperature`. Some models (e.g. reasoning/tool-calling
    /// fine-tunes) produce malformed output unless run at a specific temperature, so this
    /// lets an operator pin per-model defaults the same way llama.cpp's `models.ini` does.
    #[serde(default, alias = "temp")]
    pub temperature: Option<f32>,

    /// llama.cpp-style default nucleus sampling (top-p) for this model.
    #[serde(default, alias = "top-p")]
    pub top_p: Option<f32>,

    /// llama.cpp-style default top-k sampling for this model.
    #[serde(default, alias = "top-k")]
    pub top_k: Option<i32>,

    /// llama.cpp-style default repetition penalty for this model.
    #[serde(
        default,
        alias = "repeat-penalty",
        alias = "repeat_penalty",
        alias = "repetition-penalty"
    )]
    pub repetition_penalty: Option<f32>,

    /// Base configuration directory (populated at load time for relative path resolution)
    #[serde(skip)]
    pub config_dir: Option<PathBuf>,

    /// Used only when graph and framework detection disagree. Not a third engine.
    #[serde(default)]
    pub engine_kind: Option<EngineKind>,
}

pub const CONCURRENCY_CAP: usize = 32;

fn default_session_pool_size() -> usize {
    1
}

/// Jetson 1, discrete NVIDIA GPU 2, CPU 1. Never above [`CONCURRENCY_CAP`].
pub fn default_concurrency_limit() -> usize {
    if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        1
    } else if cfg!(feature = "cuda") || cfg!(feature = "tensorrt") {
        2
    } else {
        1
    }
}

fn default_trt_max_workspace_mb() -> Option<usize> {
    Some(512)
}

fn default_enable_mem_pattern() -> bool {
    false
}

fn default_context_memory_sharing() -> bool {
    true
}

fn default_true() -> bool {
    true
}

fn default_providers() -> Vec<String> {
    vec!["auto".to_string()]
}

fn default_max_batch_size() -> usize {
    4
}

fn default_intra_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().clamp(1, 16))
        .unwrap_or(8)
}

fn default_inter_threads() -> usize {
    1
}

fn default_n_batch() -> usize {
    512
}

impl ModelConfig {
    /// Context length. `pad_seq_len` is a separate static-shape hint and is not a context.
    pub fn effective_n_ctx(&self) -> Option<usize> {
        self.n_ctx.filter(|&v| v > 0)
    }

    pub fn effective_concurrency(&self) -> usize {
        self.concurrency_limit
            .unwrap_or_else(default_concurrency_limit)
            .clamp(1, CONCURRENCY_CAP)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub models: Vec<ModelConfig>,
    #[serde(skip)]
    pub config_dir: Option<PathBuf>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            inference_threads: default_inference_threads(),
            log_level: default_log_level(),
            ort_dylib_path: None,
        }
    }
}

impl AppConfig {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, AppError> {
        let p_ref = path.as_ref();
        let content = fs::read_to_string(p_ref).map_err(|e| {
            AppError::Config(format!(
                "Failed to read configuration file '{}': {}",
                p_ref.display(),
                e
            ))
        })?;

        let mut config: AppConfig = serde_yaml::from_str(&content)
            .map_err(|e| AppError::Config(format!("Failed to parse YAML configuration: {}", e)))?;

        let config_dir = p_ref.parent().map(|p| {
            if p.as_os_str().is_empty() {
                PathBuf::from(".")
            } else {
                p.to_path_buf()
            }
        });

        config.config_dir = config_dir.clone();
        for m in &mut config.models {
            m.config_dir = config_dir.clone();
        }
        config.validate()?;

        Ok(config)
    }

    fn validate(&mut self) -> Result<(), AppError> {
        for model in &mut self.models {
            if model.session_pool_size == 0 || model.session_pool_size > CONCURRENCY_CAP {
                return Err(AppError::Config(format!(
                    "Model '{}' session_pool_size must be 1..={CONCURRENCY_CAP} (small pool, not a queue of sessions)",
                    model.name
                )));
            }
            let concurrency = model
                .concurrency_limit
                .unwrap_or_else(default_concurrency_limit);
            if concurrency == 0 || concurrency > CONCURRENCY_CAP {
                return Err(AppError::Config(format!(
                    "Model '{}' concurrency_limit must be 1..={CONCURRENCY_CAP}",
                    model.name
                )));
            }
            model.concurrency_limit = Some(concurrency);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::task::DecisionContract;

    #[test]
    fn decision_contract_is_optional_and_yaml_configured() {
        let config: AppConfig = serde_yaml::from_str(
            r#"
models:
  - name: d1
    path: /models/d1/model.onnx
    task: decision
    decision_contract: d1
"#,
        )
        .unwrap();
        assert_eq!(
            config.models[0].task,
            Some(crate::engine::task::ModelTask::Decision)
        );
        assert_eq!(config.models[0].decision_contract, Some(DecisionContract::D1));

        let ordinary: AppConfig = serde_yaml::from_str(
            r#"
models:
  - name: embedding
    path: /models/embedding/model.onnx
"#,
        )
        .unwrap();
        assert_eq!(ordinary.models[0].decision_contract, None);
    }
}
