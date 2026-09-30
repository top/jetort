use crate::config::ModelConfig;
use crate::engine::provider::{
    applied_trt_profiles, configure_builder_for_provider,
    configure_builder_for_provider_with_precision, current_trt_manifest, detect_model_precision,
    get_candidate_providers, has_cached_trt_engine, is_corrupt_trt_engine_error,
    is_missing_trt_shape_profile_error, quarantine_trt_cache, sanitize_einsum_equations,
    trt_cache_dir, trt_profile_signature_from_shapes, trt_runtime_profile_bounds,
    write_trt_manifest,
};
use crate::engine::provider_chain::{self, FallbackReason, ShapeRoute, TrtProfileBounds};
use crate::error::AppError;
use crate::protocol::kserve_v2::TensorMetadata;
use half::f16;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use ort::value::{DynValue, Tensor, ValueType};
use parking_lot::Mutex;
use std::collections::HashMap;
use tokenizers::Tokenizer;
use tracing::{info, warn};

#[derive(Debug, Clone)]
pub struct DecisionTemperatures {
    pub default: f32,
    pub per_candidate_count: HashMap<usize, f32>,
}

/// Calibration for the bespoke "OpenJev rl-agent" System One contract (e.g. Laya): a
/// ModernBERT-style encoder + decision head that scores per-option `[MASK]` marker positions,
/// distinct from the GLiClass zero-shot-classification contract. See
/// `rl_common.py`'s `RLAgent`/`build_sequence`/`temp_bucket` for the authoritative reference
/// this mirrors.
#[derive(Debug, Clone)]
pub struct OpenJevCalibration {
    /// Total token budget for one encoded sequence (`cfg["max_len"]`); the ONNX bundle observed
    /// in practice is a fixed-shape export at this exact length (no padding to a shorter length).
    pub max_len: usize,
    /// Token budget shared between the head/instructions text and the rendered option texts
    /// (`cfg["head_max_len"]`); the remaining budget after that goes to the serialized `state`.
    pub head_max_len: usize,
    /// Fallback per-qtype (choice=0, score=1, noul=2) temperature, used when no
    /// `temperature_by_options` bucket matches.
    pub temperature: [f32; 3],
    /// Per-`"<qtype>:<option-count-bucket>"` calibrated temperature (`cfg["temperature_by_options"]`).
    pub temperature_by_options: HashMap<String, f32>,
}

fn load_openjev_calibration(model_dir: &std::path::Path) -> Option<OpenJevCalibration> {
    let candidates = [
        model_dir.join("rl_agent_config.json"),
        model_dir.join("source").join("rl_agent_config.json"),
    ];
    let raw = candidates
        .iter()
        .find_map(|path| std::fs::read_to_string(path).ok())?;
    let cfg: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let max_len = cfg.get("max_len")?.as_u64()? as usize;
    let head_max_len = cfg.get("head_max_len")?.as_u64()? as usize;
    let temp_values = cfg.get("temperature")?.as_array()?;
    if temp_values.len() < 3 {
        return None;
    }
    let mut temperature = [1.0f32; 3];
    for (slot, value) in temperature.iter_mut().zip(temp_values.iter()) {
        let parsed = value.as_f64()?;
        if !parsed.is_finite() || parsed <= 0.0 {
            return None;
        }
        *slot = parsed as f32;
    }
    let temperature_by_options = cfg
        .get("temperature_by_options")
        .and_then(|value| value.as_object())
        .map(|map| {
            map.iter()
                .filter_map(|(key, value)| {
                    let parsed = value.as_f64()?;
                    (parsed.is_finite() && parsed > 0.0).then_some((key.clone(), parsed as f32))
                })
                .collect()
        })
        .unwrap_or_default();
    Some(OpenJevCalibration {
        max_len,
        head_max_len,
        temperature,
        temperature_by_options,
    })
}

fn load_decision_temperatures(model_dir: &std::path::Path) -> Option<DecisionTemperatures> {
    let config = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
    let config: serde_json::Value = serde_json::from_str(&config).ok()?;
    let is_gliclass = config
        .get("model_type")
        .and_then(|value| value.as_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("gliclass"))
        || config
            .get("architectures")
            .and_then(|value| value.as_array())
            .is_some_and(|values| {
                values
                    .iter()
                    .filter_map(|value| value.as_str())
                    .any(|value| value.eq_ignore_ascii_case("GLiClassModel"))
            });
    if !is_gliclass {
        return None;
    }

    let calibrator = std::fs::read_to_string(model_dir.join("calibrator.json")).ok()?;
    let calibrator: serde_json::Value = serde_json::from_str(&calibrator).ok()?;
    let default = calibrator
        .get("temperature")
        .and_then(|value| value.as_f64())
        .filter(|value| value.is_finite() && *value > 0.0)? as f32;
    let per_candidate_count = calibrator
        .get("per_k")
        .and_then(|value| value.as_object())
        .map(|values| {
            values
                .iter()
                .filter_map(|(count, temperature)| {
                    Some((
                        count.parse::<usize>().ok()?,
                        temperature
                            .as_f64()
                            .filter(|value| value.is_finite() && *value > 0.0)?
                            as f32,
                    ))
                })
                .collect()
        })
        .unwrap_or_default();

    Some(DecisionTemperatures {
        default,
        per_candidate_count,
    })
}

/// Searches a directory for the primary ONNX model file.
pub fn find_local_onnx_file(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let priority_names = [
        "model_int8_qdq.onnx",
        "model_quant_int8.onnx",
        "model_int8.onnx",
        "model_fp16.onnx",
        "model_quantized.onnx",
        "model.onnx",
    ];
    for name in &priority_names {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }

    // Direct check for any .onnx file in the root directory
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut onnx_files = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().is_some_and(|ext| ext == "onnx") {
                onnx_files.push(path);
            }
        }
        if !onnx_files.is_empty() {
            onnx_files.sort();
            return onnx_files.into_iter().next();
        }
    }

    // Check subfolders (e.g. dir/onnx/model.onnx)
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                for name in &priority_names {
                    let p = path.join(name);
                    if p.is_file() {
                        return Some(p);
                    }
                }
                if let Ok(sub_entries) = std::fs::read_dir(&path) {
                    for sub_entry in sub_entries.flatten() {
                        let sub_path = sub_entry.path();
                        if sub_path.is_file()
                            && sub_path.extension().is_some_and(|ext| ext == "onnx")
                        {
                            return Some(sub_path);
                        }
                    }
                }
            }
        }
    }

    None
}

/// Searches a directory for standard tokenizer or vocabulary files.
pub fn find_local_tokenizer_file(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let priority_names = [
        "tokenizer.json",
        "vocab.txt",
        "tokenizer.model",
        "vocab.json",
        "tokens.txt",
        "dict.txt",
        "qwen.tiktoken",
        "sentencepiece.bpe.model",
    ];

    for name in &priority_names {
        let p = dir.join(name);
        if p.is_file() {
            return Some(p);
        }
    }

    // Check subfolders
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                for name in &priority_names {
                    let p = path.join(name);
                    if p.is_file() {
                        return Some(p);
                    }
                }
                // RLAgent bundles keep the HF tokenizer at `<model>/tokenizer` or
                // `<model>/source/tokenizer` (AutoTokenizer.from_pretrained).
                let nested = path.join("tokenizer");
                if nested.is_dir() {
                    for name in &priority_names {
                        let p = nested.join(name);
                        if p.is_file() {
                            return Some(p);
                        }
                    }
                }
            }
        }
    }

    None
}

/// Resolves a potentially relative path against config_dir and current working directory.
pub fn resolve_relative_path(
    path: &std::path::Path,
    config_dir: Option<&std::path::Path>,
) -> std::path::PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }

    // 1. Try relative to config_dir if provided and exists
    if let Some(base) = config_dir {
        let candidate = base.join(path);
        if candidate.exists() {
            return candidate;
        }
    }

    // 2. Try relative to current working directory if exists
    if path.exists() {
        if let Ok(cwd) = std::env::current_dir() {
            let candidate = cwd.join(path);
            if candidate.exists() {
                return candidate;
            }
        }
        return path.to_path_buf();
    }

    // 3. Fallback: join to config_dir if available, else current_dir or as-is
    if let Some(base) = config_dir {
        base.join(path)
    } else if let Ok(cwd) = std::env::current_dir() {
        cwd.join(path)
    } else {
        path.to_path_buf()
    }
}

pub fn resolve_local_model_and_tokenizer(
    config: &ModelConfig,
) -> Result<(std::path::PathBuf, Option<std::path::PathBuf>), AppError> {
    let cfg_dir = config.config_dir.as_deref();

    // Determine the base path and model file
    let (model_path, search_dir) = if let Some(ref raw_path_str) = config.path {
        let raw_path = std::path::Path::new(raw_path_str);
        let resolved_base = resolve_relative_path(raw_path, cfg_dir);

        if let Some(ref file_name) = config.hf_file {
            // Both path and file are specified
            if resolved_base.is_dir() {
                let full = resolved_base.join(file_name);
                let dir = resolved_base;
                (full, dir)
            } else if resolved_base.is_file() {
                // If resolved_base already ends with file_name, use it directly
                if resolved_base.file_name().and_then(|n| n.to_str()) == Some(file_name.as_str()) {
                    let dir = resolved_base
                        .parent()
                        .map(|p| p.to_path_buf())
                        .unwrap_or_else(|| std::path::PathBuf::from("."));
                    (resolved_base, dir)
                } else if let Some(parent) = resolved_base.parent() {
                    let candidate = parent.join(file_name);
                    if candidate.exists() {
                        (candidate, parent.to_path_buf())
                    } else {
                        let full = resolved_base.join(file_name);
                        (full, resolved_base)
                    }
                } else {
                    let full = resolved_base.join(file_name);
                    (full, resolved_base)
                }
            } else {
                let full = resolved_base.join(file_name);
                let dir = resolved_base;
                (full, dir)
            }
        } else {
            // Only path is specified (file is omitted)
            if resolved_base.is_file() || raw_path_str.ends_with(".onnx") {
                let dir = resolved_base
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| std::path::PathBuf::from("."));
                (resolved_base, dir)
            } else if resolved_base.is_dir() {
                if let Some(found) = find_local_onnx_file(&resolved_base) {
                    info!(
                        model = %config.name,
                        discovered_onnx = %found.display(),
                        "Discovered ONNX model in directory"
                    );
                    let dir = resolved_base;
                    (found, dir)
                } else {
                    return Err(AppError::Config(format!(
                        "Model '{}': Directory '{}' does not contain an ONNX model file. Please specify 'file: <filename.onnx>' in configuration.",
                        config.name,
                        resolved_base.display()
                    )));
                }
            } else {
                let dir = resolved_base
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| std::path::PathBuf::from("."));
                (resolved_base, dir)
            }
        }
    } else if let Some(ref file_name) = config.hf_file {
        // path is None, but file is specified directly
        let raw_file = std::path::Path::new(file_name);
        let resolved_file = resolve_relative_path(raw_file, cfg_dir);
        let dir = resolved_file
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        (resolved_file, dir)
    } else {
        return Err(AppError::Config(format!(
            "Model '{}': Either 'path' or 'hf_repo' must be provided in configuration",
            config.name
        )));
    };

    info!(
        model = %config.name,
        model_path = %model_path.display(),
        "Resolved local ONNX model path"
    );

    // Resolve or auto-discover tokenizer
    let tokenizer_path = if let Some(ref tok_str) = config.tokenizer_path {
        let tok_p = std::path::Path::new(tok_str);
        if tok_p.is_absolute() {
            Some(tok_p.to_path_buf())
        } else {
            let candidate_in_dir = search_dir.join(tok_p);
            if candidate_in_dir.exists() {
                Some(candidate_in_dir)
            } else {
                Some(resolve_relative_path(tok_p, cfg_dir))
            }
        }
    } else {
        find_local_tokenizer_file(&search_dir)
            .or_else(|| model_path.parent().and_then(find_local_tokenizer_file))
    };

    if let Some(ref tok) = tokenizer_path {
        info!(
            model = %config.name,
            tokenizer = %tok.display(),
            "Associated tokenizer/vocabulary from local directory"
        );
    }

    Ok((model_path, tokenizer_path))
}

pub fn resolve_model_and_tokenizer_path(
    config: &ModelConfig,
) -> Result<(std::path::PathBuf, Option<std::path::PathBuf>), AppError> {
    // 1. Try Hugging Face cache first if hf_repo is specified
    if let Some(ref hf_repo) = config.hf_repo {
        if let Some(resolved) =
            crate::engine::hf::resolve_from_hf_cache(hf_repo, config.hf_file.as_deref())
        {
            return Ok((resolved.model_path, resolved.tokenizer_path));
        }
    }

    // 2. Resolve local model path (or fallback if hf_repo was not found in cache)
    if config.path.is_some() || (config.hf_repo.is_none() && config.hf_file.is_some()) {
        return resolve_local_model_and_tokenizer(config);
    }

    // 3. Fallback error reporting
    if let Some(ref hf_repo) = config.hf_repo {
        Err(AppError::Config(format!(
            "Model '{}': Unable to find model in local Hugging Face cache for repo '{}'. Run 'huggingface-cli download {}' first, or provide a local 'path' in configuration.",
            config.name, hf_repo, hf_repo
        )))
    } else {
        Err(AppError::Config(format!(
            "Model '{}': Either 'path' or 'hf_repo' must be provided in configuration",
            config.name
        )))
    }
}

pub fn validate_model_file(path: &std::path::Path, model_name: &str) -> Result<(), AppError> {
    if !path.exists() {
        return Err(AppError::Config(format!(
            "Model file '{}' does not exist",
            path.display()
        )));
    }

    let file_len = path.metadata().map(|m| m.len()).unwrap_or(0);
    if file_len == 0 {
        return Err(AppError::Config(format!(
            "Model file '{}' is empty (0 bytes). Please re-download the model.",
            path.display()
        )));
    }

    if file_len < 1024 {
        if let Ok(content) = std::fs::read_to_string(path) {
            if content.starts_with("version https://git-lfs") {
                return Err(AppError::Config(format!(
                    "Model file '{}' is a Git LFS pointer text file ({} bytes), not actual weights! Please download full weights with 'git lfs pull' or 'huggingface-cli download'.",
                    path.display(),
                    file_len
                )));
            }
        }
    }

    info!(
        model = %model_name,
        file_size_bytes = file_len,
        "Validated model file integrity"
    );
    Ok(())
}

pub fn inspect_or_fallback(
    path: &std::path::Path,
    config: &ModelConfig,
) -> crate::engine::onnx_introspect::OnnxModelInspection {
    match crate::engine::onnx_introspect::inspect_onnx_model(path, config) {
        Ok(insp) => insp,
        Err(e) => {
            warn!(
                model = %config.name,
                error = %e,
                "Failed to perform zero-copy ONNX header inspection; falling back to basic heuristics"
            );
            crate::engine::onnx_introspect::OnnxModelInspection {
                ir_version: None,
                opset_version: None,
                producer_name: None,
                producer_version: None,
                metadata_props: std::collections::HashMap::new(),
                inputs: vec![],
                outputs: vec![],
                op_types: std::collections::HashSet::new(),
                quantization_kind: crate::engine::onnx_introspect::QuantizationKind::None,
                has_uppercase_einsum: false,
                is_audio_model: false,
                recommended_provider: "CPU".to_string(),
                candidate_providers: get_candidate_providers(config),
                trt_compilation_feasible: true,
                trt_skip_reason: None,
                synthesized_profiles: None,
            }
        }
    }
}

pub struct OneShotSession {
    pub name: String,
    pub aliases: Vec<String>,
    pub pool: crate::engine::session_pool::SessionPool,
    pub active_provider: String,
    pub precision: Option<String>,
    pub inputs_metadata: Vec<TensorMetadata>,
    pub outputs_metadata: Vec<TensorMetadata>,
    pub tokenizer: Option<Tokenizer>,
    pub vocab_table: Option<HashMap<u32, String>>,
    pub pad_seq_len: Option<usize>,
    pub max_batch_size: usize,
    pub inference_semaphore: std::sync::Arc<tokio::sync::Semaphore>,
    pub n_ctx_train: usize,
    pub n_ctx: usize,
    pub n_batch: usize,
    pub model_path: std::path::PathBuf,
    /// ONNX file the primary pool was committed from (may be a sanitized cache copy).
    effective_onnx_path: std::path::PathBuf,
    /// Config snapshot used to lazily build the next-provider session for
    /// requests whose shapes exceed the active TensorRT profile.
    model_config: ModelConfig,
    /// Bounds of the loaded TensorRT engine profile (None when not on TensorRT).
    pub trt_profile: Option<TrtProfileBounds>,
    /// Lazily created next-backend pool (CUDA then CPU) for over-profile shapes.
    fallback_pool: Mutex<Option<(String, crate::engine::session_pool::SessionPool)>>,
    pub decision_temperatures: Option<DecisionTemperatures>,
    pub openjev: Option<OpenJevCalibration>,
}

impl OneShotSession {
    pub fn load(config: &ModelConfig) -> Result<Self, AppError> {
        let (resolved_model_path, inferred_tokenizer_path) =
            resolve_model_and_tokenizer_path(config)?;
        validate_model_file(&resolved_model_path, &config.name)?;
        let inspection = inspect_or_fallback(&resolved_model_path, config);
        Self::load_with_inspection(
            config,
            resolved_model_path,
            inferred_tokenizer_path,
            inspection,
        )
    }

    pub fn load_with_inspection(
        config: &ModelConfig,
        resolved_model_path: std::path::PathBuf,
        inferred_tokenizer_path: Option<std::path::PathBuf>,
        inspection: crate::engine::onnx_introspect::OnnxModelInspection,
    ) -> Result<Self, AppError> {
        let file_len = resolved_model_path.metadata().map(|m| m.len()).unwrap_or(0);
        inspection.print_inspection_banner(&config.name);

        let candidate_providers = inspection.candidate_providers.clone();

        let cache_dir = format!(".cache/{}", config.name);
        let sanitized_filename = resolved_model_path
            .file_name()
            .unwrap_or(std::ffi::OsStr::new("model.onnx"));
        let sanitized_path = std::path::PathBuf::from(&cache_dir).join(sanitized_filename);

        let effective_model_path = if sanitized_path.exists() {
            info!(
                model = %config.name,
                sanitized_path = %sanitized_path.display(),
                "Using existing cached sanitized ONNX model"
            );
            sanitized_path
        } else if inspection.has_uppercase_einsum
            || crate::engine::provider::detect_einsum_in_file(&resolved_model_path, 8 * 1024 * 1024)
        {
            if let Ok(mut bytes) = std::fs::read(&resolved_model_path) {
                let einsum_count = sanitize_einsum_equations(&mut bytes);
                if einsum_count > 0 {
                    let _ = std::fs::create_dir_all(&cache_dir);
                    match std::fs::write(&sanitized_path, &bytes) {
                        Ok(_) => {
                            info!(
                                model = %config.name,
                                count = einsum_count,
                                sanitized_path = %sanitized_path.display(),
                                "Normalized uppercase Einsum equations to lowercase and cached sanitized model"
                            );
                            drop(bytes);
                            crate::engine::memory::trim_allocator_heap();
                            sanitized_path
                        }
                        Err(e) => {
                            warn!(
                                model = %config.name,
                                "Failed to write sanitized model to '{}': {}. Using original file.",
                                sanitized_path.display(), e
                            );
                            resolved_model_path.clone()
                        }
                    }
                } else {
                    drop(bytes);
                    crate::engine::memory::trim_allocator_heap();
                    resolved_model_path.clone()
                }
            } else {
                resolved_model_path.clone()
            }
        } else {
            resolved_model_path.clone()
        };

        // Guarantee heap memory is returned to the OS before initializing execution providers
        crate::engine::memory::trim_allocator_heap();

        // Model precision determined from inspection or path heuristic:
        // In TensorRT, only models with explicit INT8-QDQ nodes (QuantizeLinear/DequantizeLinear)
        // should activate native TensorRT INT8 mode (with_int8(true)).
        // Models using CPU dynamic quantization (MatMulInteger) or static QLinear operators do NOT have
        // explicit TensorRT QDQ calibration graphs; attempting with_int8(true) on them forces TensorRT's
        // builder to attempt dynamic range calibration, exhausting memory (12+ GB).
        // Therefore, non-QDQ models (including CPU dynamic INT8 models like SenseVoice) run on TensorRT
        // in high-speed FP16 mixed precision, matching pre-compiled engine caches.
        let detected_precision = match inspection.quantization_kind {
            crate::engine::onnx_introspect::QuantizationKind::ExplicitQdqInt8 => {
                crate::engine::provider::ModelPrecision::Int8
            }
            crate::engine::onnx_introspect::QuantizationKind::CpuDynamicInt8
            | crate::engine::onnx_introspect::QuantizationKind::StaticQLinearInt8 => {
                crate::engine::provider::ModelPrecision::Fp16
            }
            crate::engine::onnx_introspect::QuantizationKind::Fp16 => {
                crate::engine::provider::ModelPrecision::Fp16
            }
            crate::engine::onnx_introspect::QuantizationKind::None => {
                detect_model_precision(&effective_model_path)
            }
        };
        crate::engine::memory::trim_allocator_heap();

        info!(
            model = %config.name,
            model_path = %effective_model_path.display(),
            detected_precision = %detected_precision,
            "🎯 Model precision resolved: '{}'",
            detected_precision
        );

        let mut session_opt = None;
        let mut active_provider = provider_chain::Backend::Cpu.canonical_name().to_string();
        let mut active_precision: Option<String> = None;
        let mut last_err = None;
        let mut trt_cache_retried = false;
        let selection = provider_chain::selection_plan(&config.providers);
        for warning in &selection.fallback_warnings {
            warn!(model = %config.name, "{warning}");
        }

        let create_base_builder = |cfg: &ModelConfig| -> Result<SessionBuilder, String> {
            let mut builder = Session::builder()
                .map_err(|e| format!("Failed to initialize session builder: {}", e))?;

            builder = builder
                .with_optimization_level(GraphOptimizationLevel::Level1)
                .map_err(|e| format!("Failed to set optimization level: {}", e))?;

            builder = builder
                .with_intra_threads(cfg.intra_threads)
                .map_err(|e| format!("Failed to set intra threads: {}", e))?;

            builder = builder
                .with_inter_threads(cfg.inter_threads)
                .map_err(|e| format!("Failed to set inter threads: {}", e))?;

            builder = builder
                .with_config_entry("session.intra_op.allow_spinning", "0")
                .map_err(|e| format!("Failed to set intra_op allow_spinning: {}", e))?;

            builder = builder
                .with_config_entry("session.inter_op.allow_spinning", "0")
                .map_err(|e| format!("Failed to set inter_op allow_spinning: {}", e))?;

            // Disable memory pattern optimization on dynamic shapes to prevent memory leaks in BFC arena
            builder = builder
                .with_memory_pattern(cfg.enable_mem_pattern)
                .map_err(|e| format!("Failed to set memory pattern optimization: {}", e))?;

            // Allocate initializers (model weights) directly in device memory to prevent arena pollution
            builder = builder
                .with_device_allocated_initializers()
                .map_err(|e| format!("Failed to set device allocated initializers: {}", e))?;

            Ok(builder)
        };

        info!(
            model = %config.name,
            providers = ?candidate_providers,
            configured = ?config.providers,
            automatic = config.providers.iter().any(|provider| provider.eq_ignore_ascii_case("auto")),
            "execution provider selection started"
        );

        for (idx, provider_name) in candidate_providers.iter().enumerate() {
            let p_lower = provider_name.to_lowercase();
            if p_lower == "tensorrt" || p_lower == "trt" {
                let (feasible, reason) = crate::engine::provider::is_trt_compilation_feasible(
                    config,
                    Some(&effective_model_path),
                    file_len,
                );
                if !feasible {
                    let next_provider = candidate_providers.get(idx + 1);
                    if let Some(next) = next_provider {
                        let line = FallbackReason::DeviceMissing {
                            detail: reason.clone(),
                        }
                        .warning_line(&provider_chain::canonical_provider_name(next));
                        warn!(
                            model = %config.name,
                            reason = %reason,
                            fallback_to = %provider_chain::canonical_provider_name(next),
                            "{line}"
                        );
                    } else {
                        info!(
                            model = %config.name,
                            reason = %reason,
                            "⚡ TensorRT cold compilation pre-flight skipped: {}. No alternate candidate provider configured.",
                            reason
                        );
                    }
                    continue;
                }

                // TensorRT compilation: attempt with the detected model precision.
                // NOTE on Jetson UMA architecture: We do NOT attempt serial cold builds of different precisions
                // in the same process loop, because each failed TensorRT engine compilation retains workspace
                // and optimizer scratch in device memory until process exit, rapidly leading to OOM crash.
                // A failed startup build drops the builder, trims the heap, and
                // lets this load attempt continue down the provider chain.
                let target_precision = detected_precision;

                info!(
                    model = %config.name,
                    target_precision = %target_precision,
                    "Attempting TensorRT engine build with precision: '{}'...",
                    target_precision
                );

                let base_builder = match create_base_builder(config) {
                    Ok(b) => b,
                    Err(e) => {
                        last_err = Some(e);
                        continue;
                    }
                };

                let mut configured_builder = match configure_builder_for_provider_with_precision(
                    base_builder,
                    config,
                    provider_name,
                    target_precision,
                    Some(&effective_model_path),
                ) {
                    Ok(b) => b,
                    Err(e) => {
                        warn!(
                            model = %config.name,
                            precision = %target_precision,
                            error = %e,
                            "TensorRT EP setup failed with precision '{}': {}",
                            target_precision, e
                        );
                        last_err = Some(e.to_string());
                        continue;
                    }
                };

                info!(
                    model = %config.name,
                    provider = %provider_name,
                    precision = %target_precision,
                    model_path = %effective_model_path.display(),
                    "Compiling and initializing ONNX model session via TensorRT ({})...",
                    target_precision
                );

                // Spawn a background thread to log deep memory telemetry during the blocking compilation call
                let telemetry_active =
                    std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
                let telemetry_active_clone = telemetry_active.clone();
                let model_name_clone = config.name.clone();
                let telemetry_thread = std::thread::spawn(move || {
                    let mut seconds = 0;
                    while telemetry_active_clone.load(std::sync::atomic::Ordering::Relaxed) {
                        std::thread::sleep(std::time::Duration::from_millis(500));
                        seconds += 1;
                        if seconds % 10 == 0 {
                            // Log every 5 seconds (10 * 500ms)
                            let snap = crate::engine::memory::get_memory_snapshot();
                            tracing::info!(
                                model = %model_name_clone,
                                rss_mb = snap.process_rss_mb,
                                system_available_mb = snap.system_available_mb,
                                system_total_mb = snap.system_total_mb,
                                "TensorRT engine compilation in progress"
                            );
                        }
                    }
                });

                let commit_result = configured_builder.commit_from_file(&effective_model_path);

                // Stop telemetry thread
                telemetry_active.store(false, std::sync::atomic::Ordering::Relaxed);
                let _ = telemetry_thread.join();

                match commit_result {
                    Ok(sess) => {
                        info!(
                            model = %config.name,
                            provider = provider_chain::Backend::TensorRt.canonical_name(),
                            precision = %target_precision,
                            priority = idx + 1,
                            "✨ Model successfully initialized and active on provider 'TensorRT'"
                        );
                        session_opt = Some(sess);
                        active_provider = provider_chain::Backend::TensorRt
                            .canonical_name()
                            .to_string();
                        active_precision = Some(target_precision.to_string());
                        let cache_dir = trt_cache_dir(config);
                        let applied =
                            applied_trt_profiles(config, &cache_dir, Some(&effective_model_path));
                        let profile_sig = trt_profile_signature_from_shapes(
                            &applied.min_shapes,
                            &applied.opt_shapes,
                            &applied.max_shapes,
                        );
                        if let Some(manifest) = current_trt_manifest(
                            &effective_model_path,
                            &target_precision.to_string(),
                            &profile_sig,
                        ) {
                            if let Err(err) = write_trt_manifest(&cache_dir, &manifest) {
                                warn!(
                                    model = %config.name,
                                    error = %err,
                                    "Failed to write TensorRT cache manifest"
                                );
                            }
                        }
                        break;
                    }
                    Err(e) => {
                        // Immediately drop builder and reclaim heap to protect Tegra unified memory
                        drop(configured_builder);
                        crate::engine::memory::trim_allocator_heap();

                        let cache_dir = trt_cache_dir(config);
                        let err_text = e.to_string();

                        // Missing EP shape profiles is a session-config bug, not a bad engine.
                        // Keep the cache, rebuild options (applied_trt_profiles covers all ONNX
                        // inputs), and retry the SAME engine once. Never quarantine for this.
                        if !trt_cache_retried
                            && has_cached_trt_engine(&cache_dir)
                            && is_missing_trt_shape_profile_error(&err_text)
                        {
                            trt_cache_retried = true;
                            warn!(
                                model = %config.name,
                                provider = "TensorRT",
                                error = %err_text,
                                "⚠️ TensorRT session create failed: EP options omitted a dynamic input profile; retrying SAME cache with complete profiles (engine not quarantined)"
                            );
                            let mut retry_builder =
                                match create_base_builder(config).and_then(|b| {
                                    configure_builder_for_provider_with_precision(
                                        b,
                                        config,
                                        provider_name,
                                        target_precision,
                                        Some(&effective_model_path),
                                    )
                                    .map_err(|err| err.to_string())
                                }) {
                                    Ok(builder) => builder,
                                    Err(err) => {
                                        last_err = Some(err);
                                        continue;
                                    }
                                };
                            match retry_builder.commit_from_file(&effective_model_path) {
                                Ok(sess) => {
                                    info!(
                                        model = %config.name,
                                        provider = provider_chain::Backend::TensorRt.canonical_name(),
                                        precision = %target_precision,
                                        "✨ Model successfully initialized and active on provider 'TensorRT' (cache reused after profile-option fix)"
                                    );
                                    session_opt = Some(sess);
                                    active_provider = provider_chain::Backend::TensorRt
                                        .canonical_name()
                                        .to_string();
                                    active_precision = Some(target_precision.to_string());
                                    if let Some(manifest) = {
                                        let applied = applied_trt_profiles(
                                            config,
                                            &cache_dir,
                                            Some(&effective_model_path),
                                        );
                                        let profile_sig = trt_profile_signature_from_shapes(
                                            &applied.min_shapes,
                                            &applied.opt_shapes,
                                            &applied.max_shapes,
                                        );
                                        current_trt_manifest(
                                            &effective_model_path,
                                            &target_precision.to_string(),
                                            &profile_sig,
                                        )
                                    } {
                                        let _ = write_trt_manifest(&cache_dir, &manifest);
                                    }
                                    break;
                                }
                                Err(retry_err) => {
                                    crate::engine::memory::trim_allocator_heap();
                                    last_err = Some(format!(
                                        "TensorRT profile-option retry failed: {retry_err}"
                                    ));
                                    // Fall through to CUDA/CPU — do not quarantine a good engine.
                                    if let Some(next) = candidate_providers.get(idx + 1) {
                                        let line = provider_chain::classify_provider_failure(
                                            &retry_err.to_string(),
                                        )
                                        .warning_line(&provider_chain::canonical_provider_name(
                                            next,
                                        ));
                                        warn!(model = %config.name, "{line}");
                                    }
                                    continue;
                                }
                            }
                        }

                        // Only quarantine when the engine file itself looks corrupt / incompatible.
                        if !trt_cache_retried
                            && has_cached_trt_engine(&cache_dir)
                            && is_corrupt_trt_engine_error(&err_text)
                        {
                            trt_cache_retried = true;
                            match quarantine_trt_cache(&cache_dir) {
                                Ok(dest) => {
                                    warn!(
                                        model = %config.name,
                                        provider = "TensorRT",
                                        quarantine = %dest.display(),
                                        error = %err_text,
                                        "⚠️ TensorRT engine deserialize failed (corrupt/incompatible cache); quarantined and recompiling once — this is not a fallback to TensorRT"
                                    );
                                }
                                Err(qerr) => {
                                    warn!(
                                        model = %config.name,
                                        error = %qerr,
                                        "Failed to quarantine TensorRT cache after deserialize error"
                                    );
                                }
                            }
                            // Rebuild once against an empty cache before leaving TensorRT.
                            let mut retry_builder =
                                match create_base_builder(config).and_then(|b| {
                                    configure_builder_for_provider_with_precision(
                                        b,
                                        config,
                                        provider_name,
                                        target_precision,
                                        Some(&effective_model_path),
                                    )
                                    .map_err(|err| err.to_string())
                                }) {
                                    Ok(builder) => builder,
                                    Err(err) => {
                                        last_err = Some(err);
                                        continue;
                                    }
                                };
                            match retry_builder.commit_from_file(&effective_model_path) {
                                Ok(sess) => {
                                    info!(
                                        model = %config.name,
                                        provider = provider_chain::Backend::TensorRt.canonical_name(),
                                        precision = %target_precision,
                                        "✨ Model successfully initialized and active on provider 'TensorRT'"
                                    );
                                    session_opt = Some(sess);
                                    active_provider = provider_chain::Backend::TensorRt
                                        .canonical_name()
                                        .to_string();
                                    active_precision = Some(target_precision.to_string());
                                    if let Some(manifest) = {
                                        let cache_dir = trt_cache_dir(config);
                                        let applied = applied_trt_profiles(
                                            config,
                                            &cache_dir,
                                            Some(&effective_model_path),
                                        );
                                        let profile_sig = trt_profile_signature_from_shapes(
                                            &applied.min_shapes,
                                            &applied.opt_shapes,
                                            &applied.max_shapes,
                                        );
                                        current_trt_manifest(
                                            &effective_model_path,
                                            &target_precision.to_string(),
                                            &profile_sig,
                                        )
                                    } {
                                        let _ =
                                            write_trt_manifest(&trt_cache_dir(config), &manifest);
                                    }
                                    break;
                                }
                                Err(retry_err) => {
                                    crate::engine::memory::trim_allocator_heap();
                                    last_err =
                                        Some(format!("TensorRT recompile failed: {retry_err}"));
                                    if let Some(next) = candidate_providers.get(idx + 1) {
                                        let line = FallbackReason::SessionCreateFailed {
                                            detail: retry_err.to_string(),
                                        }
                                        .warning_line(&provider_chain::canonical_provider_name(
                                            next,
                                        ));
                                        warn!(model = %config.name, "{line}");
                                    }
                                    continue;
                                }
                            }
                        }

                        let next_provider = candidate_providers.get(idx + 1);
                        if let Some(next) = next_provider {
                            let line = provider_chain::classify_provider_failure(&err_text)
                                .warning_line(&provider_chain::canonical_provider_name(next));
                            warn!(
                                model = %config.name,
                                failed_precision = %target_precision,
                                fallback_to = %provider_chain::canonical_provider_name(next),
                                error = %e,
                                "{line}"
                            );
                        } else {
                            warn!(
                                model = %config.name,
                                failed_precision = %target_precision,
                                error = %e,
                                "⚠️ TensorRT ({}) build failed for model '{}'. Error: {}",
                                target_precision, config.name, e
                            );
                        }
                        last_err = Some(format!("TensorRT ({}) failed: {}", target_precision, e));
                        continue;
                    }
                }
            }

            // Non-TensorRT candidate providers (CUDA, CoreML, DirectML, CPU)
            let base_builder = match create_base_builder(config) {
                Ok(b) => b,
                Err(e) => {
                    last_err = Some(e);
                    continue;
                }
            };

            let mut configured_builder =
                match configure_builder_for_provider(base_builder, config, provider_name) {
                    Ok(b) => b,
                    Err(e) => {
                        warn!(
                            model = %config.name,
                            provider = %provider_name,
                            error = %e,
                            "Provider '{}' setup failed: {}. Falling back to next candidate...",
                            provider_name, e
                        );
                        last_err = Some(e.to_string());
                        continue;
                    }
                };

            info!(
                model = %config.name,
                provider = %provider_name,
                model_path = %effective_model_path.display(),
                "Compiling and initializing ONNX model session on '{}'...",
                provider_name
            );

            let commit_result = configured_builder.commit_from_file(&effective_model_path);

            match commit_result {
                Ok(sess) => {
                    info!(
                        model = %config.name,
                        provider = %provider_name,
                        priority = idx + 1,
                        "✨ Model successfully initialized and active on provider '{}'",
                        provider_chain::canonical_provider_name(provider_name)
                    );
                    session_opt = Some(sess);
                    active_provider = provider_chain::canonical_provider_name(provider_name);
                    break;
                }
                Err(e) => {
                    let next_provider = candidate_providers.get(idx + 1);
                    if let Some(next) = next_provider {
                        let line = provider_chain::classify_provider_failure(&e.to_string())
                            .warning_line(&provider_chain::canonical_provider_name(next));
                        warn!(
                            model = %config.name,
                            failed_provider = %provider_chain::canonical_provider_name(provider_name),
                            fallback_to = %provider_chain::canonical_provider_name(next),
                            error = %e,
                            "{line}"
                        );
                    } else {
                        warn!(
                            model = %config.name,
                            failed_provider = %provider_name,
                            error = %e,
                            "❌ All candidate execution providers failed to initialize for model '{}'. Final error: {}",
                            config.name, e
                        );
                    }
                    last_err = Some(format!("{} failed: {}", provider_name, e));
                }
            }
        }

        let session = session_opt.ok_or_else(|| {
            AppError::Inference(format!(
                "Failed to load model '{}' with any candidate provider {:?}. Last error: {:?}",
                config.name, candidate_providers, last_err
            ))
        })?;

        // Inspect metadata using session.inputs() and session.outputs()
        let mut inputs_metadata = Vec::new();
        for input in session.inputs() {
            let name = input.name().to_string();
            let (datatype, shape) = match input.dtype() {
                ValueType::Tensor { ty, shape, .. } => (
                    format!("{:?}", ty),
                    shape.iter().copied().collect::<Vec<i64>>(),
                ),
                _ => ("UNKNOWN".to_string(), vec![]),
            };
            inputs_metadata.push(TensorMetadata {
                name,
                datatype,
                shape,
            });
        }

        let mut outputs_metadata = Vec::new();
        for output in session.outputs() {
            let name = output.name().to_string();
            let (datatype, shape) = match output.dtype() {
                ValueType::Tensor { ty, shape, .. } => (
                    format!("{:?}", ty),
                    shape.iter().copied().collect::<Vec<i64>>(),
                ),
                _ => ("UNKNOWN".to_string(), vec![]),
            };
            outputs_metadata.push(TensorMetadata {
                name,
                datatype,
                shape,
            });
        }

        info!(
            model = %config.name,
            active_provider = %active_provider,
            inputs = ?inputs_metadata,
            outputs = ?outputs_metadata,
            "Model session loaded successfully"
        );

        // Optional tokenizer (either automatically resolved/inferred from model directory/HF cache or raw config)
        let effective_tokenizer_path = inferred_tokenizer_path
            .or_else(|| config.tokenizer_path.as_ref().map(std::path::PathBuf::from));

        let mut tokenizer = None;
        let mut vocab_table = None;

        if let Some(ref tok_path) = effective_tokenizer_path {
            if tok_path.exists() {
                let filename = tok_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("")
                    .to_lowercase();

                if filename.ends_with(".json") {
                    match Tokenizer::from_file(tok_path) {
                        Ok(tok) => {
                            info!(
                                model = %config.name,
                                tokenizer_path = %tok_path.display(),
                                "Tokenizer loaded successfully from json"
                            );
                            tokenizer = Some(tok);
                        }
                        Err(e) => {
                            // If HuggingFace Tokenizer parser fails, try parsing as vocab.json map
                            if let Some(table) = load_vocab_json(tok_path) {
                                info!(
                                    model = %config.name,
                                    vocab_size = table.len(),
                                    vocab_path = %tok_path.display(),
                                    "Vocabulary table loaded successfully from vocab.json"
                                );
                                vocab_table = Some(table);
                            } else {
                                warn!(
                                    model = %config.name,
                                    "Failed to parse tokenizer or vocab from '{}': {}. Text/ASR endpoints may be limited.",
                                    tok_path.display(),
                                    e
                                );
                            }
                        }
                    }
                } else if filename.ends_with(".txt") {
                    if let Some(table) = load_vocab_txt(tok_path) {
                        info!(
                            model = %config.name,
                            vocab_size = table.len(),
                            vocab_path = %tok_path.display(),
                            "Vocabulary table loaded successfully from txt"
                        );
                        vocab_table = Some(table);
                    }
                }
            } else {
                warn!(
                    model = %config.name,
                    "Tokenizer/vocabulary file '{}' does not exist",
                    tok_path.display()
                );
            }
        }

        if tokenizer.is_none() && vocab_table.is_none() {
            warn!(
                model = %config.name,
                "Model session loaded without tokenizer or vocabulary table. Text generation, embeddings, reranking, and ASR decoding will require a tokenizer or vocab file."
            );
        }

        let mut sessions = vec![session];
        let pool_target = config.session_pool_size;
        if pool_target > 1 {
            info!(
                model = %config.name,
                target_size = pool_target,
                provider = %active_provider,
                "Initializing additional sessions for session pool..."
            );
            for pool_idx in 1..pool_target {
                let builder_res = if provider_chain::is_tensorrt_name(&active_provider) {
                    create_base_builder(config).and_then(|b| {
                        configure_builder_for_provider_with_precision(
                            b,
                            config,
                            &active_provider,
                            detected_precision,
                            Some(&effective_model_path),
                        )
                        .map_err(|e| e.to_string())
                    })
                } else {
                    create_base_builder(config).and_then(|b| {
                        configure_builder_for_provider(b, config, &active_provider)
                            .map_err(|e| e.to_string())
                    })
                };

                match builder_res {
                    Ok(mut b) => match b.commit_from_file(&effective_model_path) {
                        Ok(extra_sess) => {
                            sessions.push(extra_sess);
                        }
                        Err(e) => {
                            warn!(
                                model = %config.name,
                                index = pool_idx,
                                error = %e,
                                "Failed to create extra session for pool. Capping pool at {} sessions.",
                                sessions.len()
                            );
                            break;
                        }
                    },
                    Err(e) => {
                        warn!(
                            model = %config.name,
                            index = pool_idx,
                            error = %e,
                            "Failed to configure builder for pool session {}. Capping pool at {} sessions.",
                            pool_idx, sessions.len()
                        );
                        break;
                    }
                }
            }
        }
        let pool = crate::engine::session_pool::SessionPool::from_sessions(&config.name, sessions);

        // Auto-detect n_ctx_train for OneShot models
        let mut n_ctx_train = 0;
        let model_dir = resolved_model_path
            .parent()
            .unwrap_or(std::path::Path::new("."));
        let tok_cfg = model_dir.join("tokenizer_config.json");
        if tok_cfg.exists() {
            if let Ok(c) = std::fs::read_to_string(&tok_cfg) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&c) {
                    if let Some(m) = v.get("model_max_length").and_then(|x| x.as_u64()) {
                        if m > 0 && m < 100_000_000 {
                            n_ctx_train = m as usize;
                        }
                    }
                }
            }
        }
        if n_ctx_train == 0 {
            let hf_cfg = model_dir.join("config.json");
            if hf_cfg.exists() {
                if let Ok(c) = std::fs::read_to_string(&hf_cfg) {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&c) {
                        if let Some(m) = v
                            .get("max_position_embeddings")
                            .or_else(|| v.get("max_sequence_length"))
                            .and_then(|x| x.as_u64())
                        {
                            n_ctx_train = m as usize;
                        }
                    }
                }
            }
        }
        if n_ctx_train == 0 {
            for input in &inputs_metadata {
                if (input.name.contains("input_ids") || input.name.contains("tokens"))
                    && input.shape.len() == 2
                {
                    if input.shape[1] > 0 {
                        n_ctx_train = input.shape[1] as usize;
                        break;
                    }
                }
            }
        }
        if n_ctx_train == 0 {
            n_ctx_train = 512;
        }

        let n_ctx = config.effective_n_ctx().unwrap_or(n_ctx_train);
        let n_batch = config.n_batch;
        let decision_temperatures = load_decision_temperatures(model_dir);
        if decision_temperatures.is_some() {
            info!(
                model = %config.name,
                "Detected calibrated GLiClass System One model contract"
            );
        }
        let openjev = load_openjev_calibration(model_dir);
        if openjev.is_some() {
            info!(
                model = %config.name,
                "Detected calibrated OpenJev rl-agent System One model contract"
            );
        }

        info!(
            model = %config.name,
            n_ctx_train = n_ctx_train,
            n_ctx = n_ctx,
            n_batch = n_batch,
            "llama_init_from_model: n_ctx_train = {}, n_ctx = {}, n_batch = {}",
            n_ctx_train, n_ctx, n_batch
        );

        let semaphore_permits = config.effective_concurrency();
        let inference_semaphore =
            std::sync::Arc::new(tokio::sync::Semaphore::new(semaphore_permits));

        let trt_profile = if provider_chain::is_tensorrt_name(&active_provider) {
            Some(trt_runtime_profile_bounds(&trt_cache_dir(config), config))
        } else {
            None
        };
        if let Some(profile) = trt_profile {
            info!(
                model = %config.name,
                max_batch = profile.max_batch,
                max_seq = profile.max_seq,
                range = %profile.range_label(),
                "TensorRT runtime profile bounds active"
            );
        }

        Ok(Self {
            name: config.name.clone(),
            aliases: config.alias.iter().cloned().collect(),
            pool,
            active_provider,
            precision: active_precision,
            inputs_metadata,
            outputs_metadata,
            tokenizer,
            vocab_table,
            pad_seq_len: config.pad_seq_len,
            max_batch_size: config.max_batch_size,
            inference_semaphore,
            n_ctx_train,
            n_ctx,
            n_batch,
            model_path: resolved_model_path,
            effective_onnx_path: effective_model_path,
            model_config: config.clone(),
            trt_profile,
            fallback_pool: Mutex::new(None),
            decision_temperatures,
            openjev,
        })
    }

    pub async fn acquire_permit(&self) -> Result<tokio::sync::OwnedSemaphorePermit, AppError> {
        self.inference_semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| AppError::Internal(format!("Inference semaphore closed: {}", e)))
    }

    /// Lazily build the next backend after TensorRT (CUDA then CPU) for over-profile shapes.
    fn ensure_fallback_pool(
        &self,
    ) -> Result<(String, crate::engine::session_pool::SessionPool), AppError> {
        {
            let guard = self.fallback_pool.lock();
            if let Some((name, pool)) = guard.as_ref() {
                return Ok((name.clone(), pool.clone()));
            }
        }
        let plan = provider_chain::selection_plan(&self.model_config.providers);
        let candidates: Vec<_> = plan
            .walk
            .into_iter()
            .filter(|b| *b != provider_chain::Backend::TensorRt)
            .collect();
        let mut last_err = String::from("no fallback provider available");
        for backend in candidates {
            let name = backend.canonical_name().to_string();
            let builder_res =
                Session::builder()
                    .map_err(|e| e.to_string())
                    .and_then(|mut builder| {
                        builder = builder
                            .with_optimization_level(GraphOptimizationLevel::Level1)
                            .map_err(|e| e.to_string())?;
                        builder = builder
                            .with_intra_threads(self.model_config.intra_threads)
                            .map_err(|e| e.to_string())?;
                        builder = builder
                            .with_inter_threads(self.model_config.inter_threads)
                            .map_err(|e| e.to_string())?;
                        builder = builder
                            .with_memory_pattern(self.model_config.enable_mem_pattern)
                            .map_err(|e| e.to_string())?;
                        configure_builder_for_provider(builder, &self.model_config, &name)
                            .map_err(|e| e.to_string())
                    });
            match builder_res {
                Ok(mut b) => match b.commit_from_file(&self.effective_onnx_path) {
                    Ok(sess) => {
                        let pool = crate::engine::session_pool::SessionPool::from_sessions(
                            format!("{}-fallback-{}", self.name, name),
                            vec![sess],
                        );
                        *self.fallback_pool.lock() = Some((name.clone(), pool.clone()));
                        info!(
                            model = %self.name,
                            fallback = %name,
                            "Created request-time fallback session pool for over-profile shapes"
                        );
                        return Ok((name, pool));
                    }
                    Err(e) => {
                        last_err = format!("{name}: {e}");
                        crate::engine::memory::trim_allocator_heap();
                    }
                },
                Err(e) => {
                    last_err = format!("{name}: {e}");
                }
            }
        }
        Err(AppError::Inference(format!(
            "Failed to create TensorRT shape-fallback session for '{}': {last_err}",
            self.name
        )))
    }

    pub fn infer_raw(
        &self,
        inputs: HashMap<String, DynValue>,
    ) -> Result<HashMap<String, DynValue>, AppError> {
        self.infer_raw_with_targets(inputs, None)
    }

    pub fn infer_raw_with_targets(
        &self,
        inputs: HashMap<String, DynValue>,
        target_outputs: Option<&[&str]>,
    ) -> Result<HashMap<String, DynValue>, AppError> {
        self.infer_raw_on_pool(&self.pool, inputs, target_outputs)
    }

    fn infer_raw_on_pool(
        &self,
        pool: &crate::engine::session_pool::SessionPool,
        inputs: HashMap<String, DynValue>,
        target_outputs: Option<&[&str]>,
    ) -> Result<HashMap<String, DynValue>, AppError> {
        let mut lock = pool.acquire();

        let run_options = ort::session::RunOptions::new()
            .map_err(|e| AppError::Inference(format!("Failed to create RunOptions: {}", e)))?;

        let run_options = if let Some(targets) = target_outputs {
            let matching_targets: Vec<&str> = targets
                .iter()
                .copied()
                .filter(|&t| self.outputs_metadata.iter().any(|m| m.name == t))
                .collect();

            if !matching_targets.is_empty() {
                let mut selector = ort::session::OutputSelector::no_default();
                for t in matching_targets {
                    selector = selector.with(t);
                }
                run_options.with_outputs(selector)
            } else {
                let mut selector = ort::session::OutputSelector::default();
                for m in &self.outputs_metadata {
                    if m.name.starts_with("present.") {
                        selector = selector.without(&m.name);
                    }
                }
                run_options.with_outputs(selector)
            }
        } else {
            let mut selector = ort::session::OutputSelector::default();
            for m in &self.outputs_metadata {
                if m.name.starts_with("present.") {
                    selector = selector.without(&m.name);
                }
            }
            run_options.with_outputs(selector)
        };

        let mut outputs = lock.run_with_options(inputs, &run_options).map_err(|e| {
            AppError::Inference(format!(
                "Forward execution failed on model '{}': {}",
                self.name, e
            ))
        })?;

        let mut result = HashMap::new();
        for output_meta in &self.outputs_metadata {
            if let Some(val) = outputs.remove(&output_meta.name) {
                result.insert(output_meta.name.clone(), val);
            }
        }

        if result.is_empty() {
            for (name, val) in outputs {
                result.insert(name.to_string(), val);
            }
        }

        Ok(result)
    }

    /// Executes inference with zero-copy IoBinding, reducing host-device copies where supported.
    /// Transparently falls back to standard execution if IoBinding creation or binding encounters unsupported dynamic shapes.
    pub fn infer_with_io_binding(
        &self,
        inputs: HashMap<String, DynValue>,
        target_outputs: Option<&[&str]>,
    ) -> Result<HashMap<String, DynValue>, AppError> {
        let mut session = self.pool.acquire();
        let mut binding = match session.create_binding() {
            Ok(b) => b,
            Err(e) => {
                warn!(
                    model = %self.name,
                    error = %e,
                    "Failed to create IoBinding; falling back to standard session.run"
                );
                drop(session);
                return self.infer_raw_with_targets(inputs, target_outputs);
            }
        };

        for (name, val) in &inputs {
            if let Err(e) = binding.bind_input(name, val) {
                warn!(
                    model = %self.name,
                    input = %name,
                    error = %e,
                    "IoBinding bind_input failed; falling back to standard session.run"
                );
                drop(session);
                return self.infer_raw_with_targets(inputs, target_outputs);
            }
        }

        let targets: Vec<String> = if let Some(req_targets) = target_outputs {
            req_targets
                .iter()
                .filter(|&t| self.outputs_metadata.iter().any(|m| &m.name == t))
                .map(|t| t.to_string())
                .collect()
        } else {
            self.outputs_metadata
                .iter()
                .filter(|m| !m.name.starts_with("present."))
                .map(|m| m.name.clone())
                .collect()
        };

        let mem_info = match ort::memory::MemoryInfo::new(
            ort::memory::AllocationDevice::CPU,
            0,
            ort::memory::AllocatorType::Device,
            ort::memory::MemoryType::Default,
        ) {
            Ok(m) => m,
            Err(e) => {
                warn!(
                    model = %self.name,
                    error = %e,
                    "Failed to allocate MemoryInfo for IoBinding; falling back to standard session.run"
                );
                drop(session);
                return self.infer_raw_with_targets(inputs, target_outputs);
            }
        };

        for target in &targets {
            if let Err(e) = binding.bind_output_to_device(target, &mem_info) {
                warn!(
                    model = %self.name,
                    target = %target,
                    error = %e,
                    "IoBinding bind_output_to_device failed; falling back to standard session.run"
                );
                drop(session);
                return self.infer_raw_with_targets(inputs, target_outputs);
            }
        }

        let (results, run_err) = {
            match session.run_binding(&binding) {
                Ok(mut bound_outputs) => {
                    let mut results = HashMap::new();
                    for target in targets {
                        if let Some(val) = bound_outputs.remove(&target) {
                            results.insert(target, val);
                        }
                    }
                    (Some(results), None)
                }
                Err(e) => (None, Some(e)),
            }
        };

        drop(binding);
        drop(session);

        if let Some(results) = results {
            return Ok(results);
        }

        let run_error = run_err.unwrap();
        warn!(
            model = %self.name,
            error = %run_error,
            "IoBinding run_binding failed; falling back to standard session.run"
        );
        self.infer_raw_with_targets(inputs, target_outputs)
    }

    fn prepare_transformer_inputs(
        &self,
        batch_size: usize,
        effective_seq_len: usize,
        input_ids: Vec<i64>,
        attention_mask: Vec<i64>,
    ) -> Result<HashMap<String, ort::value::DynValue>, AppError> {
        let mut input_map = HashMap::new();

        // 1. input_ids with type matching
        let ids_datatype = self
            .inputs_metadata
            .iter()
            .find(|m| m.name == "input_ids")
            .map(|m| m.datatype.to_uppercase())
            .unwrap_or_else(|| "INT64".to_string());

        if ids_datatype.contains("INT32") || ids_datatype.contains("I32") {
            let ids_i32: Vec<i32> = input_ids.iter().map(|&x| x as i32).collect();
            let val =
                Tensor::from_array(([batch_size, effective_seq_len], ids_i32.into_boxed_slice()))
                    .map_err(|e| AppError::Inference(e.to_string()))?;
            input_map.insert("input_ids".to_string(), val.into_dyn());
        } else {
            let val = Tensor::from_array((
                [batch_size, effective_seq_len],
                input_ids.into_boxed_slice(),
            ))
            .map_err(|e| AppError::Inference(e.to_string()))?;
            input_map.insert("input_ids".to_string(), val.into_dyn());
        }

        // 2. attention_mask with type matching (Int64, Int32, Bool, Float32)
        let mask_datatype = self
            .inputs_metadata
            .iter()
            .find(|m| m.name == "attention_mask")
            .map(|m| m.datatype.to_uppercase())
            .unwrap_or_else(|| "INT64".to_string());

        if mask_datatype.contains("INT32") || mask_datatype.contains("I32") {
            let mask_i32: Vec<i32> = attention_mask.iter().map(|&x| x as i32).collect();
            let val =
                Tensor::from_array(([batch_size, effective_seq_len], mask_i32.into_boxed_slice()))
                    .map_err(|e| AppError::Inference(e.to_string()))?;
            input_map.insert("attention_mask".to_string(), val.into_dyn());
        } else if mask_datatype.contains("BOOL") {
            let mask_bool: Vec<bool> = attention_mask.iter().map(|&x| x != 0).collect();
            let val = Tensor::from_array((
                [batch_size, effective_seq_len],
                mask_bool.into_boxed_slice(),
            ))
            .map_err(|e| AppError::Inference(e.to_string()))?;
            input_map.insert("attention_mask".to_string(), val.into_dyn());
        } else if mask_datatype.contains("FLOAT") || mask_datatype.contains("FP32") {
            let mask_f32: Vec<f32> = attention_mask.iter().map(|&x| x as f32).collect();
            let val =
                Tensor::from_array(([batch_size, effective_seq_len], mask_f32.into_boxed_slice()))
                    .map_err(|e| AppError::Inference(e.to_string()))?;
            input_map.insert("attention_mask".to_string(), val.into_dyn());
        } else {
            let val = Tensor::from_array((
                [batch_size, effective_seq_len],
                attention_mask.into_boxed_slice(),
            ))
            .map_err(|e| AppError::Inference(e.to_string()))?;
            input_map.insert("attention_mask".to_string(), val.into_dyn());
        }

        // 3. Automatically supply additional standard inputs if required by the ONNX model graph
        for meta in &self.inputs_metadata {
            if meta.name == "position_ids" && !input_map.contains_key("position_ids") {
                let dt = meta.datatype.to_uppercase();
                if dt.contains("INT32") || dt.contains("I32") {
                    let mut pos_ids = Vec::with_capacity(batch_size * effective_seq_len);
                    for _ in 0..batch_size {
                        for i in 0..effective_seq_len {
                            pos_ids.push(i as i32);
                        }
                    }
                    let val = Tensor::from_array((
                        [batch_size, effective_seq_len],
                        pos_ids.into_boxed_slice(),
                    ))
                    .map_err(|e| AppError::Inference(e.to_string()))?;
                    input_map.insert("position_ids".to_string(), val.into_dyn());
                } else {
                    let mut pos_ids = Vec::with_capacity(batch_size * effective_seq_len);
                    for _ in 0..batch_size {
                        for i in 0..effective_seq_len {
                            pos_ids.push(i as i64);
                        }
                    }
                    let val = Tensor::from_array((
                        [batch_size, effective_seq_len],
                        pos_ids.into_boxed_slice(),
                    ))
                    .map_err(|e| AppError::Inference(e.to_string()))?;
                    input_map.insert("position_ids".to_string(), val.into_dyn());
                }
            } else if meta.name == "token_type_ids" && !input_map.contains_key("token_type_ids") {
                let dt = meta.datatype.to_uppercase();
                if dt.contains("INT32") || dt.contains("I32") {
                    let type_ids = vec![0i32; batch_size * effective_seq_len];
                    let val = Tensor::from_array((
                        [batch_size, effective_seq_len],
                        type_ids.into_boxed_slice(),
                    ))
                    .map_err(|e| AppError::Inference(e.to_string()))?;
                    input_map.insert("token_type_ids".to_string(), val.into_dyn());
                } else {
                    let type_ids = vec![0i64; batch_size * effective_seq_len];
                    let val = Tensor::from_array((
                        [batch_size, effective_seq_len],
                        type_ids.into_boxed_slice(),
                    ))
                    .map_err(|e| AppError::Inference(e.to_string()))?;
                    input_map.insert("token_type_ids".to_string(), val.into_dyn());
                }
            } else if meta.name == "marker_pos" && !input_map.contains_key("marker_pos") {
                let pos_count = if meta.shape.len() > 1 && meta.shape[1] > 0 {
                    meta.shape[1] as usize
                } else {
                    2
                };
                let marker_pos = vec![0i64; batch_size * pos_count];
                let val =
                    Tensor::from_array(([batch_size, pos_count], marker_pos.into_boxed_slice()))
                        .map_err(|e| AppError::Inference(e.to_string()))?;
                input_map.insert("marker_pos".to_string(), val.into_dyn());
            } else if meta.name == "marker_mask" && !input_map.contains_key("marker_mask") {
                let mask_count = if meta.shape.len() > 1 && meta.shape[1] > 0 {
                    meta.shape[1] as usize
                } else {
                    2
                };
                let marker_mask = vec![true; batch_size * mask_count];
                let val =
                    Tensor::from_array(([batch_size, mask_count], marker_mask.into_boxed_slice()))
                        .map_err(|e| AppError::Inference(e.to_string()))?;
                input_map.insert("marker_mask".to_string(), val.into_dyn());
            } else if meta.name == "qtype" && !input_map.contains_key("qtype") {
                let qtypes = vec![0i64; batch_size];
                let val = Tensor::from_array(([batch_size], qtypes.into_boxed_slice()))
                    .map_err(|e| AppError::Inference(e.to_string()))?;
                input_map.insert("qtype".to_string(), val.into_dyn());
            } else if meta.name.starts_with("past_key_values.")
                && !input_map.contains_key(&meta.name)
            {
                let num_heads = if meta.shape.len() > 1 && meta.shape[1] > 0 {
                    meta.shape[1] as usize
                } else {
                    8
                };
                let head_dim = if meta.shape.len() > 3 && meta.shape[3] > 0 {
                    meta.shape[3] as usize
                } else {
                    128
                };
                let dt = meta.datatype.to_uppercase();
                if dt.contains("FLOAT16") || dt.contains("FP16") || dt.contains("HALF") {
                    let empty_kv = Tensor::from_array((
                        [batch_size, num_heads, 0usize, head_dim],
                        Box::new([]) as Box<[f16]>,
                    ))
                    .map_err(|e| AppError::Inference(e.to_string()))?;
                    input_map.insert(meta.name.clone(), empty_kv.into_dyn());
                } else {
                    let empty_kv = Tensor::from_array((
                        [batch_size, num_heads, 0usize, head_dim],
                        Box::new([]) as Box<[f32]>,
                    ))
                    .map_err(|e| AppError::Inference(e.to_string()))?;
                    input_map.insert(meta.name.clone(), empty_kv.into_dyn());
                }
            }
        }

        Ok(input_map)
    }

    /// Returns `(embeddings, prompt_tokens, backend_that_ran)`.
    pub fn embed_texts(
        &self,
        texts: Vec<String>,
    ) -> Result<(Vec<Vec<f32>>, usize, String), AppError> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            AppError::Tokenizer(format!(
                "Model '{}' does not have a configured tokenizer for embedding",
                self.name
            ))
        })?;

        if texts.is_empty() {
            return Ok((vec![], 0, self.active_provider.clone()));
        }

        let all_encodings = tokenizer
            .encode_batch(texts, true)
            .map_err(|e| AppError::Tokenizer(format!("Tokenization error: {}", e)))?;

        let total_prompt_tokens: usize = all_encodings.iter().map(|e| e.get_ids().len()).sum();
        let max_seq_raw = all_encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(1);
        let max_batch_limit = self.max_batch_size.max(1);
        let max_tile_bytes: usize = 32 * 1024 * 1024;

        // One place decides TensorRT vs next backend from the loaded profile bounds.
        let (run_provider, run_pool, microbatch_cap, profile_ref): (
            String,
            crate::engine::session_pool::SessionPool,
            usize,
            Option<TrtProfileBounds>,
        ) = if provider_chain::is_tensorrt_name(&self.active_provider) {
            if let Some(profile) = self.trt_profile {
                match provider_chain::plan_shape_route(
                    all_encodings.len().min(max_batch_limit).max(1),
                    max_seq_raw.max(1),
                    &profile,
                ) {
                    ShapeRoute::TensorRt { microbatch, .. } => (
                        self.active_provider.clone(),
                        self.pool.clone(),
                        microbatch,
                        Some(profile),
                    ),
                    ShapeRoute::Fallback {
                        reason,
                        actual_batch,
                        actual_seq,
                    } => {
                        let (fb_name, fb_pool) = self.ensure_fallback_pool()?;
                        let line = provider_chain::shape_fallback_warning(
                            &fb_name,
                            &format!(
                                "actual=[{actual_batch},{actual_seq}] profile={} ({reason})",
                                profile.range_label()
                            ),
                        );
                        warn!(
                            model = %self.name,
                            target_backend = %fb_name,
                            actual_batch,
                            actual_seq,
                            profile = %profile.range_label(),
                            "{line}"
                        );
                        (fb_name, fb_pool, max_batch_limit, None)
                    }
                }
            } else {
                (
                    self.active_provider.clone(),
                    self.pool.clone(),
                    max_batch_limit,
                    None,
                )
            }
        } else {
            (
                self.active_provider.clone(),
                self.pool.clone(),
                max_batch_limit,
                self.trt_profile,
            )
        };

        let mut all_embeddings = Vec::with_capacity(all_encodings.len());
        let mut current_chunk: Vec<&tokenizers::Encoding> = Vec::new();
        let mut current_max_len: usize = 0;

        let run_chunk = |chunk: &[&tokenizers::Encoding],
                         embeddings_out: &mut Vec<Vec<f32>>,
                         provider: &str,
                         pool: &crate::engine::session_pool::SessionPool,
                         profile: Option<&TrtProfileBounds>|
         -> Result<(), AppError> {
            let batch_size = chunk.len();
            if batch_size == 0 {
                return Ok(());
            }

            let max_len_in_chunk = chunk.iter().map(|e| e.get_ids().len()).max().unwrap_or(1);
            let aligned_len = max_len_in_chunk.div_ceil(8) * 8;
            let max_safe_len = (((max_tile_bytes / (batch_size.max(1) * 16 * 4)) as f64).sqrt()
                as usize)
                .clamp(128, 512);
            let effective_seq_len = provider_chain::resolve_seq_pad(
                self.pad_seq_len,
                provider,
                aligned_len,
                self.n_ctx,
                max_safe_len,
                profile,
            );

            let mut input_ids = Vec::with_capacity(batch_size * effective_seq_len);
            let mut attention_mask = Vec::with_capacity(batch_size * effective_seq_len);

            for enc in chunk {
                let ids = enc.get_ids();
                let mask = enc.get_attention_mask();
                for &id in ids.iter().take(effective_seq_len) {
                    input_ids.push(id as i64);
                }
                for _ in ids.len()..effective_seq_len {
                    input_ids.push(0);
                }
                for &m in mask.iter().take(effective_seq_len) {
                    attention_mask.push(m as i64);
                }
                for _ in mask.len()..effective_seq_len {
                    attention_mask.push(0);
                }
            }

            let input_map = self.prepare_transformer_inputs(
                batch_size,
                effective_seq_len,
                input_ids,
                attention_mask,
            )?;

            let raw_outputs = self.infer_raw_on_pool(
                pool,
                input_map,
                Some(&["sentence_embedding", "last_hidden_state"]),
            )?;

            let output_val = raw_outputs
                .get("sentence_embedding")
                .or_else(|| raw_outputs.get("last_hidden_state"))
                .or_else(|| {
                    raw_outputs
                        .iter()
                        .find(|(k, _)| !k.starts_with("present."))
                        .map(|(_, v)| v)
                })
                .ok_or_else(|| {
                    AppError::Inference("Model produced no output tensors".to_string())
                })?;

            let (shape, data): (Vec<i64>, Vec<f32>) =
                if let Ok((s, d)) = output_val.try_extract_tensor::<f32>() {
                    (s.to_vec(), d.to_vec())
                } else if let Ok((s, d)) = output_val.try_extract_tensor::<f16>() {
                    (s.to_vec(), d.iter().map(|&x| x.to_f32()).collect())
                } else {
                    return Err(AppError::Inference(
                        "Failed to extract float or float16 tensor for embedding".to_string(),
                    ));
                };

            if shape.len() == 3 {
                let out_batch = shape[0] as usize;
                let out_seq_len = shape[1] as usize;
                let hidden_dim = shape[2] as usize;

                if out_seq_len == 1 {
                    for b in 0..batch_size.min(out_batch) {
                        let offset = b * hidden_dim;
                        if offset + hidden_dim <= data.len() {
                            let mut row = data[offset..offset + hidden_dim].to_vec();
                            let norm: f32 = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
                            if norm > 0.0 {
                                for x in &mut row {
                                    *x /= norm;
                                }
                            }
                            embeddings_out.push(row);
                        }
                    }
                } else {
                    for b in 0..batch_size.min(out_batch) {
                        let mut pooled = vec![0.0f32; hidden_dim];
                        let mut valid_tokens = 0.0f32;
                        let token_count = effective_seq_len.min(out_seq_len);

                        for s in 0..token_count {
                            let mask =
                                chunk[b].get_attention_mask().get(s).copied().unwrap_or(0) as f32;
                            if mask > 0.0 {
                                valid_tokens += 1.0;
                                let offset = (b * out_seq_len + s) * hidden_dim;
                                if offset + hidden_dim <= data.len() {
                                    for h in 0..hidden_dim {
                                        pooled[h] += data[offset + h];
                                    }
                                }
                            }
                        }

                        if valid_tokens > 0.0 {
                            for h in 0..hidden_dim {
                                pooled[h] /= valid_tokens;
                            }
                        }

                        let norm: f32 = pooled.iter().map(|&x| x * x).sum::<f32>().sqrt();
                        if norm > 0.0 {
                            for h in 0..hidden_dim {
                                pooled[h] /= norm;
                            }
                        }

                        embeddings_out.push(pooled);
                    }
                }
            } else if shape.len() == 2 {
                let out_batch = shape[0] as usize;
                let hidden_dim = shape[1] as usize;
                for b in 0..batch_size.min(out_batch) {
                    let offset = b * hidden_dim;
                    if offset + hidden_dim <= data.len() {
                        let mut row = data[offset..offset + hidden_dim].to_vec();
                        let norm: f32 = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
                        if norm > 0.0 {
                            for x in &mut row {
                                *x /= norm;
                            }
                        }
                        embeddings_out.push(row);
                    }
                }
            } else {
                return Err(AppError::Inference(format!(
                    "Unexpected embedding output shape: {:?}",
                    shape
                )));
            }

            Ok(())
        };

        let batch_cap = microbatch_cap.min(max_batch_limit).max(1);
        for enc in &all_encodings {
            let enc_len = enc.get_ids().len();
            let cand_len = current_chunk.len() + 1;
            let cand_max = current_max_len.max(enc_len);
            let aligned = cand_max.div_ceil(8) * 8;
            let cand_seq_len = provider_chain::resolve_seq_pad(
                self.pad_seq_len,
                &run_provider,
                aligned,
                self.n_ctx,
                512,
                profile_ref.as_ref(),
            );
            let cand_est_bytes = cand_len * 16 * cand_seq_len * cand_seq_len * 4;

            if !current_chunk.is_empty()
                && (cand_len > batch_cap || cand_est_bytes > max_tile_bytes)
            {
                run_chunk(
                    &current_chunk,
                    &mut all_embeddings,
                    &run_provider,
                    &run_pool,
                    profile_ref.as_ref(),
                )?;
                current_chunk.clear();
                current_max_len = 0;
            }

            current_chunk.push(enc);
            current_max_len = current_max_len.max(enc_len);
        }

        if !current_chunk.is_empty() {
            run_chunk(
                &current_chunk,
                &mut all_embeddings,
                &run_provider,
                &run_pool,
                profile_ref.as_ref(),
            )?;
        }

        Ok((all_embeddings, total_prompt_tokens, run_provider))
    }

    /// Returns `(scores, prompt_tokens, backend_that_ran)`.
    pub fn rerank_documents(
        &self,
        query: &str,
        documents: &[String],
    ) -> Result<(Vec<f32>, usize, String), AppError> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            AppError::Tokenizer(format!(
                "Model '{}' does not have a configured tokenizer for reranking",
                self.name
            ))
        })?;

        if documents.is_empty() {
            return Ok((vec![], 0, self.active_provider.clone()));
        }

        let pairs: Vec<(String, String)> = documents
            .iter()
            .map(|doc| (query.to_string(), doc.clone()))
            .collect();

        let all_encodings = tokenizer
            .encode_batch(pairs, true)
            .map_err(|e| AppError::Tokenizer(format!("Tokenization error in rerank: {}", e)))?;

        let total_prompt_tokens: usize = all_encodings.iter().map(|e| e.get_ids().len()).sum();
        let max_seq_raw = all_encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(1);
        let mut all_scores = Vec::with_capacity(all_encodings.len());
        let max_batch_limit = self.max_batch_size.max(1);
        let max_tile_bytes: usize = 32 * 1024 * 1024;

        let (run_provider, run_pool, microbatch_cap, profile_ref): (
            String,
            crate::engine::session_pool::SessionPool,
            usize,
            Option<TrtProfileBounds>,
        ) = if provider_chain::is_tensorrt_name(&self.active_provider) {
            if let Some(profile) = self.trt_profile {
                match provider_chain::plan_shape_route(
                    all_encodings.len().min(max_batch_limit).max(1),
                    max_seq_raw.max(1),
                    &profile,
                ) {
                    ShapeRoute::TensorRt { microbatch, .. } => (
                        self.active_provider.clone(),
                        self.pool.clone(),
                        microbatch,
                        Some(profile),
                    ),
                    ShapeRoute::Fallback {
                        reason,
                        actual_batch,
                        actual_seq,
                    } => {
                        let (fb_name, fb_pool) = self.ensure_fallback_pool()?;
                        let line = provider_chain::shape_fallback_warning(
                            &fb_name,
                            &format!(
                                "actual=[{actual_batch},{actual_seq}] profile={} ({reason})",
                                profile.range_label()
                            ),
                        );
                        warn!(
                            model = %self.name,
                            target_backend = %fb_name,
                            actual_batch,
                            actual_seq,
                            profile = %profile.range_label(),
                            "{line}"
                        );
                        (fb_name, fb_pool, max_batch_limit, None)
                    }
                }
            } else {
                (
                    self.active_provider.clone(),
                    self.pool.clone(),
                    max_batch_limit,
                    None,
                )
            }
        } else {
            (
                self.active_provider.clone(),
                self.pool.clone(),
                max_batch_limit,
                self.trt_profile,
            )
        };

        let mut current_chunk: Vec<&tokenizers::Encoding> = Vec::new();
        let mut current_max_len: usize = 0;

        let run_chunk = |chunk: &[&tokenizers::Encoding],
                         scores_out: &mut Vec<f32>,
                         provider: &str,
                         pool: &crate::engine::session_pool::SessionPool,
                         profile: Option<&TrtProfileBounds>|
         -> Result<(), AppError> {
            let batch_size = chunk.len();
            if batch_size == 0 {
                return Ok(());
            }

            let max_len_in_chunk = chunk.iter().map(|e| e.get_ids().len()).max().unwrap_or(1);
            let aligned_len = max_len_in_chunk.div_ceil(8) * 8;
            let max_safe_len = (((max_tile_bytes / (batch_size.max(1) * 16 * 4)) as f64).sqrt()
                as usize)
                .clamp(128, 512);
            let effective_seq_len = provider_chain::resolve_seq_pad(
                self.pad_seq_len,
                provider,
                aligned_len,
                self.n_ctx,
                max_safe_len,
                profile,
            );

            let mut input_ids = Vec::with_capacity(batch_size * effective_seq_len);
            let mut attention_mask = Vec::with_capacity(batch_size * effective_seq_len);

            for enc in chunk {
                let ids = enc.get_ids();
                let mask = enc.get_attention_mask();
                for &id in ids.iter().take(effective_seq_len) {
                    input_ids.push(id as i64);
                }
                for _ in ids.len()..effective_seq_len {
                    input_ids.push(0);
                }
                for &m in mask.iter().take(effective_seq_len) {
                    attention_mask.push(m as i64);
                }
                for _ in mask.len()..effective_seq_len {
                    attention_mask.push(0);
                }
            }

            let input_map = self.prepare_transformer_inputs(
                batch_size,
                effective_seq_len,
                input_ids,
                attention_mask,
            )?;

            let raw_outputs = self.infer_raw_on_pool(pool, input_map, Some(&["logits"]))?;
            let output_val = raw_outputs
                .get("logits")
                .or_else(|| {
                    raw_outputs
                        .iter()
                        .find(|(k, _)| !k.starts_with("present."))
                        .map(|(_, v)| v)
                })
                .ok_or_else(|| {
                    AppError::Inference("Model produced no output tensors".to_string())
                })?;

            let data: Vec<f32> = if let Ok((_shape, d)) = output_val.try_extract_tensor::<f32>() {
                d.to_vec()
            } else if let Ok((_shape, d)) = output_val.try_extract_tensor::<f16>() {
                d.iter().map(|&x| x.to_f32()).collect()
            } else {
                return Err(AppError::Inference(
                    "Failed to extract float or float16 scores for reranker".to_string(),
                ));
            };

            for &logit in data.iter().take(batch_size) {
                scores_out.push(1.0 / (1.0 + (-logit).exp()));
            }

            Ok(())
        };

        let batch_cap = microbatch_cap.min(max_batch_limit).max(1);
        for enc in &all_encodings {
            let enc_len = enc.get_ids().len();
            let cand_len = current_chunk.len() + 1;
            let cand_max = current_max_len.max(enc_len);
            let aligned = cand_max.div_ceil(8) * 8;
            let cand_seq_len = provider_chain::resolve_seq_pad(
                self.pad_seq_len,
                &run_provider,
                aligned,
                self.n_ctx,
                512,
                profile_ref.as_ref(),
            );
            let cand_est_bytes = cand_len * 16 * cand_seq_len * cand_seq_len * 4;

            if !current_chunk.is_empty()
                && (cand_len > batch_cap || cand_est_bytes > max_tile_bytes)
            {
                run_chunk(
                    &current_chunk,
                    &mut all_scores,
                    &run_provider,
                    &run_pool,
                    profile_ref.as_ref(),
                )?;
                current_chunk.clear();
                current_max_len = 0;
            }

            current_chunk.push(enc);
            current_max_len = current_max_len.max(enc_len);
        }

        if !current_chunk.is_empty() {
            run_chunk(
                &current_chunk,
                &mut all_scores,
                &run_provider,
                &run_pool,
                profile_ref.as_ref(),
            )?;
        }

        Ok((all_scores, total_prompt_tokens, run_provider))
    }

    pub fn supports_system_one(&self) -> bool {
        (self.decision_temperatures.is_some() || self.openjev.is_some())
            && self.tokenizer.is_some()
            && self
                .outputs_metadata
                .iter()
                .any(|output| output.name == "logits")
    }

    pub fn uses_openjev_contract(&self) -> bool {
        self.openjev.is_some()
    }

    pub fn system_one(
        &self,
        questions: Vec<crate::protocol::system_one::PreparedQuestion>,
    ) -> Result<
        (
            std::collections::BTreeMap<String, crate::protocol::system_one::SystemOneAnswer>,
            usize,
        ),
        AppError,
    > {
        if !self.supports_system_one() {
            return Err(AppError::Unprocessable(format!(
                "Model '{}' does not expose the calibrated GLiClass System One contract (use \
                 system_one_openjev for the OpenJev rl-agent contract)",
                self.name
            )));
        }
        let tokenizer = self.tokenizer.as_ref().expect("checked above");
        let temperatures = self.decision_temperatures.as_ref().expect("checked above");
        let mut answers = std::collections::BTreeMap::new();
        let mut total_tokens = 0;

        for chunk in questions.chunks(self.max_batch_size.max(1)) {
            let prompts: Vec<String> = chunk
                .iter()
                .map(|question| question.prompt.clone())
                .collect();
            let encodings = tokenizer.encode_batch(prompts, true).map_err(|error| {
                AppError::Tokenizer(format!("System One tokenization failed: {error}"))
            })?;
            if let Some(length) = encodings
                .iter()
                .map(|encoding| encoding.get_ids().len())
                .find(|length| *length > 512)
            {
                return Err(AppError::Unprocessable(format!(
                    "System One input is {length} tokens; this local GLiClass model supports at most 512 tokens per question"
                )));
            }
            total_tokens += encodings
                .iter()
                .map(|encoding| encoding.get_ids().len())
                .sum::<usize>();

            let batch_size = encodings.len();
            let max_length = encodings
                .iter()
                .map(|encoding| encoding.get_ids().len())
                .max()
                .unwrap_or(1)
                .min(512)
                .max(1);
            let mut input_ids = Vec::with_capacity(batch_size * max_length);
            let mut attention_mask = Vec::with_capacity(batch_size * max_length);
            for encoding in &encodings {
                input_ids.extend(
                    encoding
                        .get_ids()
                        .iter()
                        .take(max_length)
                        .map(|id| *id as i64),
                );
                input_ids.resize(
                    input_ids.len() + max_length.saturating_sub(encoding.get_ids().len()),
                    0,
                );
                attention_mask.extend(
                    encoding
                        .get_attention_mask()
                        .iter()
                        .take(max_length)
                        .map(|mask| *mask as i64),
                );
                attention_mask.resize(
                    attention_mask.len()
                        + max_length.saturating_sub(encoding.get_attention_mask().len()),
                    0,
                );
            }

            let inputs =
                self.prepare_transformer_inputs(batch_size, max_length, input_ids, attention_mask)?;
            let outputs = self.infer_raw_with_targets(inputs, Some(&["logits"]))?;
            let output = outputs
                .get("logits")
                .or_else(|| outputs.values().next())
                .ok_or_else(|| {
                    AppError::Inference("System One model produced no logits".to_string())
                })?;
            let (shape, logits): (Vec<i64>, Vec<f32>) =
                if let Ok((shape, values)) = output.try_extract_tensor::<f32>() {
                    (shape.to_vec(), values.to_vec())
                } else if let Ok((shape, values)) = output.try_extract_tensor::<f16>() {
                    (
                        shape.to_vec(),
                        values.iter().map(|value| value.to_f32()).collect(),
                    )
                } else {
                    return Err(AppError::Inference(
                        "System One logits must be FP32 or FP16".to_string(),
                    ));
                };
            if shape.len() != 2 || shape[0] as usize != batch_size {
                return Err(AppError::Inference(format!(
                    "Unexpected System One logits shape: {shape:?}"
                )));
            }
            let stride = shape[1] as usize;
            for (index, question) in chunk.iter().enumerate() {
                let start = index * stride;
                let end = start + stride;
                let temperature = temperatures
                    .per_candidate_count
                    .get(&question.candidate_ids.len())
                    .copied()
                    .unwrap_or(temperatures.default);
                let answer = crate::protocol::system_one::answer_question(
                    question,
                    &logits[start..end],
                    temperature,
                )?;
                answers.insert(question.id.clone(), answer);
            }
        }

        Ok((answers, total_tokens))
    }

    /// Scores questions against the bespoke OpenJev "rl-agent" contract (e.g. Laya): a
    /// ModernBERT-style encoder whose decision head reads per-option `[MASK]` marker positions.
    /// Mirrors `rl_agent_api.RLAgent.system_one` / `rl_common.build_sequence` exactly, since the
    /// fitted calibration temperatures assume that precise tokenization/marker-placement
    /// algorithm. Unlike the GLiClass contract there is no abstention candidate, and (per the
    /// fixed-shape ONNX export observed for this contract) each question is run as its own
    /// batch-of-1 inference call.
    pub fn system_one_openjev(
        &self,
        state: &serde_json::Value,
        questions: &std::collections::BTreeMap<
            String,
            crate::protocol::system_one::SystemOneQuestion,
        >,
    ) -> Result<
        (
            std::collections::BTreeMap<String, crate::protocol::system_one::SystemOneAnswer>,
            usize,
        ),
        AppError,
    > {
        use crate::protocol::system_one as s1;

        let calibration = self.openjev.as_ref().ok_or_else(|| {
            AppError::Unprocessable(format!(
                "Model '{}' does not expose the calibrated OpenJev rl-agent System One contract",
                self.name
            ))
        })?;
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            AppError::Tokenizer(format!(
                "Model '{}' has no tokenizer configured for OpenJev System One scoring",
                self.name
            ))
        })?;

        let cls_id = tokenizer.token_to_id("[CLS]").ok_or_else(|| {
            AppError::Tokenizer("OpenJev tokenizer is missing a [CLS] token".to_string())
        })?;
        let sep_id = tokenizer.token_to_id("[SEP]").ok_or_else(|| {
            AppError::Tokenizer("OpenJev tokenizer is missing a [SEP] token".to_string())
        })?;
        let mask_id = tokenizer.token_to_id("[MASK]").ok_or_else(|| {
            AppError::Tokenizer("OpenJev tokenizer is missing a [MASK] token".to_string())
        })?;
        let pad_id = tokenizer.token_to_id("[PAD]").unwrap_or(0);
        let mask_token_str = "[MASK]";

        let marker_capacity = self
            .inputs_metadata
            .iter()
            .find(|m| m.name == "marker_pos")
            .and_then(|m| m.shape.get(1).copied())
            .filter(|&n| n > 0)
            .map(|n| n as usize)
            .unwrap_or(4);
        let max_len = calibration.max_len;
        let head_max_len = calibration.head_max_len;

        let encode = |text: &str| -> Result<Vec<u32>, AppError> {
            tokenizer
                .encode(text, false)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|error| {
                    AppError::Tokenizer(format!("OpenJev tokenization failed: {error}"))
                })
        };
        let sanitize = |text: &str| text.replace(mask_token_str, " ");
        let state_text = sanitize(&s1::entry_text(state));

        let mut answers = std::collections::BTreeMap::new();
        let mut total_tokens = 0usize;

        for (id, question) in questions {
            let qtype = s1::openjev_qtype_index(question);
            let options = s1::openjev_render_options(question);
            let option_ids = s1::openjev_option_ids(question);
            if options.len() < 2 {
                return Err(AppError::Unprocessable(format!(
                    "questions.{id} must have at least 2 options for this model"
                )));
            }
            if options.len() > marker_capacity {
                return Err(AppError::Unprocessable(format!(
                    "questions.{id} has {} options, but this OpenJev model supports at most {} \
                     options per question (fixed-shape marker export)",
                    options.len(),
                    marker_capacity
                )));
            }

            let head_ids = encode(&sanitize(&s1::openjev_head_text(question)))?;
            let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(options.len());
            for option in &options {
                let mut ids = vec![mask_id];
                let tokens = encode(&sanitize(&format!(" {option}")))?;
                ids.extend(tokens.into_iter().take(48));
                opt_ids.push(ids);
            }

            let opt_len_sum: usize = opt_ids.iter().map(|o| o.len()).sum();
            let opt_budget = (head_max_len as i64) - (opt_len_sum as i64);
            if opt_budget < 16 {
                let per = ((head_max_len.saturating_sub(16)) / opt_ids.len().max(1)).max(4);
                for opt in &mut opt_ids {
                    opt.truncate(per);
                }
            }
            let opt_len_sum: usize = opt_ids.iter().map(|o| o.len()).sum();
            let opt_budget = head_max_len.saturating_sub(opt_len_sum).max(0);
            let head_take = opt_budget.max(8).min(head_ids.len());

            let mut ids: Vec<u32> = Vec::with_capacity(max_len);
            ids.push(cls_id);
            ids.extend_from_slice(&head_ids[..head_take]);
            ids.push(sep_id);
            let mut markers: Vec<usize> = Vec::with_capacity(opt_ids.len());
            for opt in &opt_ids {
                markers.push(ids.len());
                ids.extend_from_slice(opt);
            }
            ids.push(sep_id);
            let room = max_len.saturating_sub(ids.len() + 1);
            let state_ids = encode(&state_text)?;
            let state_take = state_ids.len().min(room);
            ids.extend_from_slice(&state_ids[..state_take]);
            ids.push(sep_id);
            ids.truncate(max_len);
            markers.retain(|&pos| pos < max_len);
            if markers.len() != opt_ids.len() {
                return Err(AppError::Unprocessable(format!(
                    "questions.{id}: options did not fit within this model's {max_len}-token \
                     sequence budget"
                )));
            }

            let seq_len = ids.len();
            total_tokens += seq_len;

            // Fixed-shape ONNX export: pad input_ids/attention_mask out to exactly `max_len`
            // tokens and marker_pos/marker_mask out to exactly `marker_capacity` slots.
            let mut input_ids = vec![pad_id as i64; max_len];
            let mut attention_mask = vec![0i64; max_len];
            for (index, token_id) in ids.iter().enumerate() {
                input_ids[index] = *token_id as i64;
                attention_mask[index] = 1;
            }
            let mut marker_pos = vec![0i64; marker_capacity];
            let mut marker_mask = vec![false; marker_capacity];
            for (index, pos) in markers.iter().enumerate() {
                marker_pos[index] = *pos as i64;
                marker_mask[index] = true;
            }
            let qtype_tensor = vec![qtype as i64];

            let mut input_map = HashMap::new();
            input_map.insert(
                "input_ids".to_string(),
                Tensor::from_array(([1usize, max_len], input_ids.into_boxed_slice()))
                    .map_err(|e| AppError::Inference(e.to_string()))?
                    .into_dyn(),
            );
            input_map.insert(
                "attention_mask".to_string(),
                Tensor::from_array(([1usize, max_len], attention_mask.into_boxed_slice()))
                    .map_err(|e| AppError::Inference(e.to_string()))?
                    .into_dyn(),
            );
            input_map.insert(
                "marker_pos".to_string(),
                Tensor::from_array(([1usize, marker_capacity], marker_pos.into_boxed_slice()))
                    .map_err(|e| AppError::Inference(e.to_string()))?
                    .into_dyn(),
            );
            input_map.insert(
                "marker_mask".to_string(),
                Tensor::from_array(([1usize, marker_capacity], marker_mask.into_boxed_slice()))
                    .map_err(|e| AppError::Inference(e.to_string()))?
                    .into_dyn(),
            );
            input_map.insert(
                "qtype".to_string(),
                Tensor::from_array(([1usize], qtype_tensor.into_boxed_slice()))
                    .map_err(|e| AppError::Inference(e.to_string()))?
                    .into_dyn(),
            );

            let outputs = self.infer_raw_with_targets(input_map, Some(&["logits"]))?;
            let output = outputs
                .get("logits")
                .or_else(|| outputs.values().next())
                .ok_or_else(|| {
                    AppError::Inference("OpenJev System One model produced no logits".to_string())
                })?;
            let (shape, logits): (Vec<i64>, Vec<f32>) =
                if let Ok((shape, values)) = output.try_extract_tensor::<f32>() {
                    (shape.to_vec(), values.to_vec())
                } else if let Ok((shape, values)) = output.try_extract_tensor::<f16>() {
                    (shape.to_vec(), values.iter().map(|v| v.to_f32()).collect())
                } else {
                    return Err(AppError::Inference(
                        "OpenJev System One logits must be FP32 or FP16".to_string(),
                    ));
                };
            if shape.len() != 2 || shape[1] as usize != marker_capacity {
                return Err(AppError::Inference(format!(
                    "Unexpected OpenJev System One logits shape: {shape:?}"
                )));
            }
            let k = options.len();
            let bucket = s1::openjev_temp_bucket(qtype, k);
            let temperature = calibration
                .temperature_by_options
                .get(&bucket)
                .copied()
                .unwrap_or(calibration.temperature[qtype.min(2)]);
            let probabilities = s1::calibrated_probabilities(&logits[..k], temperature);
            let answer = s1::answer_question_openjev(id, question, &option_ids, &probabilities)?;
            answers.insert(id.clone(), answer);
        }

        Ok((answers, total_tokens))
    }

    pub fn transcribe_audio(
        &self,
        pcm_samples: &[f32],
        language: Option<&str>,
    ) -> Result<(String, Vec<u32>, usize), AppError> {
        if self.tokenizer.is_none() && self.vocab_table.is_none() {
            return Err(AppError::Tokenizer(format!(
                "Model '{}' does not have an active tokenizer or vocabulary table. Please configure 'tokenizer_path' in config.yaml, or ensure 'vocab.txt' / 'tokenizer.json' is present in the Hugging Face cache for repo '{}'.",
                self.name,
                self.name
            )));
        }

        if pcm_samples.is_empty() {
            return Ok((String::new(), Vec::new(), 0));
        }

        let mut input_map = HashMap::new();

        // Check if model expects 80-channel Fbank features (like SenseVoice / Whisper / Paraformer)
        let first_input = self.inputs_metadata.first();
        let is_fbank_model = first_input
            .map(|m| {
                m.name == "features" || (m.shape.len() == 3 && m.shape.last().copied() == Some(80))
            })
            .unwrap_or(false);

        let total_frames = if is_fbank_model {
            let (fbank_data, num_frames) = crate::engine::audio::compute_fbank_80(pcm_samples);
            if num_frames == 0 {
                return Ok((String::new(), Vec::new(), 0));
            }

            let features_val =
                Tensor::from_array(([1usize, num_frames, 80usize], fbank_data.into_boxed_slice()))
                    .map_err(|e| AppError::Inference(e.to_string()))?;
            input_map.insert("features".to_string(), features_val.into_dyn());

            // Supply companion inputs if expected (e.g. SenseVoice features_lens, language, textnorm)
            for meta in &self.inputs_metadata {
                if meta.name == "features_lens" {
                    let lens_val =
                        Tensor::from_array(([1usize], Box::new([num_frames as i64]) as Box<[i64]>))
                            .map_err(|e| AppError::Inference(e.to_string()))?;
                    input_map.insert("features_lens".to_string(), lens_val.into_dyn());
                } else if meta.name == "language" {
                    let lang_code: i64 = match language {
                        Some("en") => 1,
                        Some("yue") => 2,
                        Some("ja") => 3,
                        Some("ko") => 4,
                        _ => 0,
                    };
                    let lang_val =
                        Tensor::from_array(([1usize], Box::new([lang_code]) as Box<[i64]>))
                            .map_err(|e| AppError::Inference(e.to_string()))?;
                    input_map.insert("language".to_string(), lang_val.into_dyn());
                } else if meta.name == "textnorm" {
                    let tn_val = Tensor::from_array(([1usize], Box::new([15i64]) as Box<[i64]>))
                        .map_err(|e| AppError::Inference(e.to_string()))?;
                    input_map.insert("textnorm".to_string(), tn_val.into_dyn());
                }
            }
            num_frames
        } else {
            let sample_count = pcm_samples.len();
            let waveform = Tensor::from_array((
                [1usize, sample_count],
                pcm_samples.to_vec().into_boxed_slice(),
            ))
            .map_err(|e| AppError::Inference(e.to_string()))?;
            let input_name = first_input.map(|m| m.name.as_str()).unwrap_or("speech");
            input_map.insert(input_name.to_string(), waveform.into_dyn());
            (sample_count / 160).max(1)
        };

        let raw_outputs =
            self.infer_raw_with_targets(input_map, Some(&["logprobs", "logits", "ctc_logits"]))?;

        let output_val = raw_outputs
            .get("logprobs")
            .or_else(|| raw_outputs.values().next())
            .ok_or_else(|| {
                AppError::Inference("ASR model produced no output tensors".to_string())
            })?;

        let token_ids: Vec<u32> = if let Ok((shape, data)) = output_val.try_extract_tensor::<f32>()
        {
            if shape.len() == 3 {
                let num_frames = shape[1] as usize;
                let vocab_size = shape[2] as usize;
                crate::engine::audio::ctc_greedy_decode(data, num_frames, vocab_size, 0)
            } else {
                Vec::new()
            }
        } else if let Ok((shape, data)) = output_val.try_extract_tensor::<f16>() {
            if shape.len() == 3 {
                let num_frames = shape[1] as usize;
                let vocab_size = shape[2] as usize;
                let data_f32: Vec<f32> = data.iter().map(|&x| x.to_f32()).collect();
                crate::engine::audio::ctc_greedy_decode(&data_f32, num_frames, vocab_size, 0)
            } else {
                Vec::new()
            }
        } else if let Ok((_shape, ids)) = output_val.try_extract_tensor::<i64>() {
            ids.iter().map(|&x| x as u32).collect()
        } else if let Ok((_shape, ids)) = output_val.try_extract_tensor::<i32>() {
            ids.iter().map(|&x| x as u32).collect()
        } else {
            return Err(AppError::Inference(
                "Unable to extract discrete token IDs from ASR model output".to_string(),
            ));
        };

        let decoded_text = if let Some(ref tokenizer) = self.tokenizer {
            tokenizer
                .decode(&token_ids, true)
                .map_err(|e| AppError::Tokenizer(format!("ASR token decoding failure: {}", e)))?
        } else if let Some(ref vocab) = self.vocab_table {
            let mut text = String::new();
            for &id in &token_ids {
                if let Some(tok) = vocab.get(&id) {
                    // Skip special control tokens (e.g. <|zh|>, <|en|>, <|NEUTRAL|>, <blk>, <s>, </s>)
                    if (tok.starts_with("<|") && tok.ends_with("|>"))
                        || (tok.starts_with('<') && tok.ends_with('>'))
                    {
                        continue;
                    }
                    let clean = tok.replace('\u{2581}', " ");
                    text.push_str(&clean);
                }
            }
            text.trim().to_string()
        } else {
            String::new()
        };

        Ok((decoded_text, token_ids, total_frames))
    }
}

/// Parses vocab.txt / tokens.txt / dict.txt into token-id to token-string map
fn load_vocab_txt(path: &std::path::Path) -> Option<HashMap<u32, String>> {
    let content = std::fs::read_to_string(path).ok()?;
    let mut map = HashMap::new();
    for (line_idx, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() >= 2 {
            // Case 1: "<token> <id>"
            if let Ok(id) = parts[parts.len() - 1].parse::<u32>() {
                let token = parts[..parts.len() - 1].join(" ");
                map.insert(id, token);
                continue;
            }
            // Case 2: "<id> <token>"
            if let Ok(id) = parts[0].parse::<u32>() {
                let token = parts[1..].join(" ");
                map.insert(id, token);
                continue;
            }
        }
        // Case 3: Line index is the ID
        map.insert(line_idx as u32, trimmed.to_string());
    }
    Some(map)
}

/// Parses vocab.json into token-id to token-string map
fn load_vocab_json(path: &std::path::Path) -> Option<HashMap<u32, String>> {
    let content = std::fs::read_to_string(path).ok()?;
    let val: serde_json::Value = serde_json::from_str(&content).ok()?;
    if let serde_json::Value::Object(map) = val {
        let mut vocab = HashMap::new();
        for (k, v) in map {
            if let Some(id) = v.as_u64() {
                vocab.insert(id as u32, k);
            } else if let Ok(id) = k.parse::<u32>() {
                if let Some(tok) = v.as_str() {
                    vocab.insert(id, tok.to_string());
                }
            }
        }
        if !vocab.is_empty() {
            return Some(vocab);
        }
    }
    None
}

/// Single-pass ONNX session. Autoregressive graphs are refused in [`ModelEngine::load`].
pub struct ModelEngine {
    session: std::sync::Arc<OneShotSession>,
}

/// Backward-compatible alias for ModelSession
pub type ModelSession = ModelEngine;

impl ModelEngine {
    pub fn load(config: &ModelConfig) -> Result<Self, AppError> {
        let (resolved_model_path, inferred_tokenizer_path) =
            resolve_model_and_tokenizer_path(config)?;
        validate_model_file(&resolved_model_path, &config.name)?;
        let inspection = inspect_or_fallback(&resolved_model_path, config);

        let probe = crate::engine::header_prober::probe_model_bundle(
            &resolved_model_path,
            Some(&inspection),
            config.engine_kind,
        );
        info!(
            model = %config.name,
            engine_kind = %probe.kind,
            reason = %probe.detection_reason,
            "Classified model engine"
        );

        match probe.kind {
            crate::config::EngineKind::OneShot => {
                let oneshot = OneShotSession::load_with_inspection(
                    config,
                    resolved_model_path,
                    inferred_tokenizer_path,
                    inspection,
                )?;
                Ok(ModelEngine {
                    session: std::sync::Arc::new(oneshot),
                })
            }
            crate::config::EngineKind::Autoregressive => Err(AppError::InvalidInput(format!(
                "Model '{}' is autoregressive ({}) and JetOrt only runs single-pass ONNX models. Refusing to load.",
                config.name, probe.detection_reason
            ))),
        }
    }

    pub fn name(&self) -> &str {
        &self.session.name
    }

    pub fn aliases(&self) -> &[String] {
        &self.session.aliases
    }

    pub fn active_provider(&self) -> &str {
        &self.session.active_provider
    }

    pub fn precision(&self) -> Option<&str> {
        self.session.precision.as_deref()
    }

    pub fn inputs_metadata(&self) -> &[TensorMetadata] {
        &self.session.inputs_metadata
    }

    pub fn outputs_metadata(&self) -> &[TensorMetadata] {
        &self.session.outputs_metadata
    }

    pub fn as_oneshot(&self) -> Result<&std::sync::Arc<OneShotSession>, AppError> {
        Ok(&self.session)
    }

    pub fn accepts_audio(&self) -> bool {
        self.session.inputs_metadata.iter().any(|input| {
            input.name == "features"
                || input.name == "speech"
                || (input.shape.len() == 3 && input.shape.last().copied() == Some(80))
        })
    }

    pub fn pad_seq_len(&self) -> Option<usize> {
        self.session.pad_seq_len
    }

    pub fn n_ctx(&self) -> Option<usize> {
        Some(self.session.n_ctx)
    }

    pub fn model_path(&self) -> &std::path::Path {
        &self.session.model_path
    }

    pub fn created_at(&self) -> u64 {
        self.model_path()
            .metadata()
            .ok()
            .and_then(|metadata| metadata.created().or_else(|_| metadata.modified()).ok())
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|duration| duration.as_secs())
            .unwrap_or(0)
    }

    pub fn model_size(&self) -> u64 {
        self.model_path()
            .metadata()
            .map(|metadata| metadata.len())
            .unwrap_or(0)
    }

    pub fn engine_name(&self) -> &'static str {
        "onnxruntime"
    }

    pub fn supports_system_one(&self) -> bool {
        self.session.supports_system_one()
    }

    pub fn supported_endpoints(&self) -> Vec<String> {
        let session = &self.session;
        let mut endpoints = Vec::new();
        if session.supports_system_one() {
            endpoints.push("/v1/systemone".to_string());
        } else {
            if session.outputs_metadata.iter().any(|output| {
                output.name == "sentence_embedding" || output.name == "last_hidden_state"
            }) {
                endpoints.push("/v1/embeddings".to_string());
            }
            if session
                .outputs_metadata
                .iter()
                .any(|output| output.name == "logits")
                && session.tokenizer.is_some()
            {
                endpoints.push("/v1/rerank".to_string());
            }
        }
        if session.inputs_metadata.iter().any(|input| {
            input.name == "features"
                || input.name == "speech"
                || (input.shape.len() == 3 && input.shape.last().copied() == Some(80))
        }) {
            endpoints.push("/v1/audio/transcriptions".to_string());
        }
        endpoints
    }

    pub async fn acquire_permit(&self) -> Result<tokio::sync::OwnedSemaphorePermit, AppError> {
        self.session.acquire_permit().await
    }

    pub fn infer_raw(
        &self,
        inputs: HashMap<String, DynValue>,
    ) -> Result<HashMap<String, DynValue>, AppError> {
        self.session.infer_raw(inputs)
    }

    pub fn infer_raw_with_targets(
        &self,
        inputs: HashMap<String, DynValue>,
        target_outputs: Option<&[&str]>,
    ) -> Result<HashMap<String, DynValue>, AppError> {
        self.session.infer_raw_with_targets(inputs, target_outputs)
    }

    pub fn embed_texts(
        &self,
        texts: Vec<String>,
    ) -> Result<(Vec<Vec<f32>>, usize, String), AppError> {
        self.session.embed_texts(texts)
    }

    pub fn rerank_documents(
        &self,
        query: &str,
        documents: &[String],
    ) -> Result<(Vec<f32>, usize, String), AppError> {
        self.session.rerank_documents(query, documents)
    }

    pub fn transcribe_audio(
        &self,
        pcm_samples: &[f32],
        language: Option<&str>,
    ) -> Result<(String, Vec<u32>, usize), AppError> {
        self.session.transcribe_audio(pcm_samples, language)
    }

    pub fn system_one(
        &self,
        questions: Vec<crate::protocol::system_one::PreparedQuestion>,
    ) -> Result<
        (
            std::collections::BTreeMap<String, crate::protocol::system_one::SystemOneAnswer>,
            usize,
        ),
        AppError,
    > {
        self.session.system_one(questions)
    }

    pub fn uses_openjev_contract(&self) -> bool {
        self.session.uses_openjev_contract()
    }

    pub fn system_one_openjev(
        &self,
        state: &serde_json::Value,
        questions: &std::collections::BTreeMap<
            String,
            crate::protocol::system_one::SystemOneQuestion,
        >,
    ) -> Result<
        (
            std::collections::BTreeMap<String, crate::protocol::system_one::SystemOneAnswer>,
            usize,
        ),
        AppError,
    > {
        self.session.system_one_openjev(state, questions)
    }
}

#[cfg(test)]
mod decision_tests {
    use super::*;

    #[test]
    fn finds_tokenizer_nested_under_source() {
        let path = std::env::temp_dir().join(format!("jetort-tok-{}", std::process::id()));
        let nested = path.join("source").join("tokenizer");
        std::fs::create_dir_all(&nested).unwrap();
        let file = nested.join("tokenizer.json");
        std::fs::write(&file, "{}").unwrap();
        let found = find_local_tokenizer_file(&path).unwrap();
        assert_eq!(found, file);
        std::fs::remove_dir_all(&path).ok();
    }

    #[test]
    fn detects_gliclass_calibrator() {
        let path = std::env::temp_dir().join(format!("jetort-systemone-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(
            path.join("config.json"),
            r#"{"model_type":"GLiClass","architectures":["GLiClassModel"]}"#,
        )
        .unwrap();
        std::fs::write(
            path.join("calibrator.json"),
            r#"{"temperature":2.5,"per_k":{"3":4.0}}"#,
        )
        .unwrap();

        let temperatures = load_decision_temperatures(&path).unwrap();
        assert_eq!(temperatures.default, 2.5);
        assert_eq!(temperatures.per_candidate_count[&3], 4.0);

        std::fs::remove_dir_all(path).unwrap();
    }
}
