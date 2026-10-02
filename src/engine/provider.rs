use crate::config::ModelConfig;
use ort::session::builder::SessionBuilder;
use std::io::Read;
use tracing::info;
#[cfg(feature = "tensorrt")]
use tracing::warn;

/// Detected precision of an ONNX model file, used to configure execution provider precision flags
/// and drive the TensorRT precision fallback chain.
///
/// Detection strategy (see [`detect_model_precision`]):
/// 1. Byte-level scan for QDQ (QuantizeLinear/DequantizeLinear) operator nodes → INT8
/// 2. Filename convention (HuggingFace ecosystem: `_int8`, `_fp16`, etc.) → corresponding precision
/// 3. Default → FP32
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelPrecision {
    Int8,
    Fp16,
    Fp32,
}

impl std::fmt::Display for ModelPrecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelPrecision::Int8 => write!(f, "INT8"),
            ModelPrecision::Fp16 => write!(f, "FP16"),
            ModelPrecision::Fp32 => write!(f, "FP32"),
        }
    }
}

impl ModelPrecision {
    /// Returns the TensorRT precision fallback chain starting from the detected precision.
    ///
    /// - INT8 models: INT8 (with FP16 mixed-precision) → FP16 → FP32
    /// - FP16 models: FP16 → FP32
    /// - FP32 models: FP16 (TRT's standard optimization for FP32 weights) → FP32 (pure)
    pub fn trt_precision_fallback_chain(self) -> &'static [ModelPrecision] {
        match self {
            ModelPrecision::Int8 => &[
                ModelPrecision::Int8,
                ModelPrecision::Fp16,
                ModelPrecision::Fp32,
            ],
            ModelPrecision::Fp16 => &[ModelPrecision::Fp16, ModelPrecision::Fp32],
            ModelPrecision::Fp32 => &[ModelPrecision::Fp16, ModelPrecision::Fp32],
        }
    }
}

pub fn get_candidate_providers(config: &ModelConfig) -> Vec<String> {
    crate::engine::provider_chain::candidate_provider_names(config)
}

// Candidates ordering is determined cleanly by user configuration or default platform priority:
// TensorRT -> CUDA -> CPU (on Linux / Jetson), or CoreML -> CPU (on macOS), etc.

/// Sanitizes uppercase Einsum equations in ONNX protobuf bytes (e.g. "BD,BCD->BC")
/// to lowercase ("bd,bcd->bc").
/// TensorRT's onnxOpCheckers.cpp strictly checks `islower(c)` for equation characters,
/// rejecting any equation with uppercase letters and causing unwanted graph partitioning.
/// Converting them in-place preserves protobuf wire format and allows TensorRT to compile
/// the Einsum node into a single unified engine.
pub fn sanitize_einsum_equations(bytes: &mut [u8]) -> usize {
    let mut modified = 0;
    // Match the full protobuf wire pattern for AttributeProto.name = "equation":
    //   tag 0x0a (field 1, wire type 2) + length 0x08 + "equation"
    let name_pattern: &[u8] = b"\x0a\x08equation";
    let len = bytes.len();
    let mut i = 0;

    while i + name_pattern.len() < len {
        if &bytes[i..i + name_pattern.len()] == name_pattern {
            let after_name = i + name_pattern.len();
            // Scan ahead for tag 0x22 = AttributeProto.s (field 4, wire type 2)
            // Other fields (e.g. type=0xa0 0x01 0x03) may appear in between.
            let scan_end = (after_name + 24).min(len);
            let mut found = false;
            let mut j = after_name;
            while j < scan_end {
                if bytes[j] == 0x22 && j + 1 < len {
                    // Decode varint length
                    let (str_len, varint_size) = decode_protobuf_varint(&bytes[j + 1..]);
                    if varint_size > 0 && str_len > 0 && str_len < 256 {
                        let data_start = j + 1 + varint_size;
                        let data_end = data_start + str_len;
                        if data_end <= len {
                            let slice = &bytes[data_start..data_end];
                            // Only modify if it looks like an Einsum equation (contains "->")
                            if slice.windows(2).any(|w| w == b"->") {
                                for b in &mut bytes[data_start..data_end] {
                                    if *b >= b'A' && *b <= b'Z' {
                                        *b = b.to_ascii_lowercase();
                                        modified += 1;
                                    }
                                }
                            }
                            i = data_end;
                            found = true;
                            break;
                        }
                    }
                    j += 1;
                } else {
                    j += 1;
                }
            }
            if !found {
                i = after_name;
            }
        } else {
            i += 1;
        }
    }
    modified
}

/// Decode a protobuf varint from a byte slice.
/// Returns (value, number_of_bytes_consumed). Returns (0, 0) on invalid input.
fn decode_protobuf_varint(bytes: &[u8]) -> (usize, usize) {
    let mut result: usize = 0;
    let mut shift = 0;
    for (idx, &b) in bytes.iter().enumerate().take(5) {
        result |= ((b & 0x7f) as usize) << shift;
        if b & 0x80 == 0 {
            return (result, idx + 1);
        }
        shift += 7;
    }
    (0, 0)
}

#[cfg(feature = "cuda")]
pub fn build_cuda_ep(config: &ModelConfig) -> ort::ep::CUDA {
    let mut cuda = ort::ep::CUDA::default()
        .with_arena_extend_strategy(ort::ep::ArenaExtendStrategy::SameAsRequested)
        .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
        .with_conv_max_workspace(false);

    if let Some(limit_mb) = config.cuda_memory_limit_mb {
        info!(
            model = %config.name,
            limit_mb = limit_mb,
            "Enforcing user-configured CUDA EP memory arena limit"
        );
        cuda = cuda.with_memory_limit(limit_mb * 1024 * 1024);
    }
    cuda
}

/// Checks if a valid pre-compiled TensorRT engine cache exists in the given directory.
pub fn has_cached_trt_engine(cache_dir: &str) -> bool {
    let path = std::path::Path::new(cache_dir);
    if !path.is_dir() {
        return false;
    }
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() {
                let is_engine = p.extension().map(|ext| ext == "engine").unwrap_or(false)
                    || p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|name| name.contains("engine"))
                        .unwrap_or(false);
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                if is_engine && size > 1024 {
                    return true;
                }
            }
        }
    }
    false
}

pub fn trt_cache_dir(config: &ModelConfig) -> String {
    config
        .trt_cache_dir
        .clone()
        .unwrap_or_else(|| format!(".cache/{}", config.name))
}

const TRT_MANIFEST_FILE: &str = "jetort-trt-manifest.json";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TrtCacheManifest {
    pub onnx_sha256: String,
    pub tensorrt_version: String,
    pub cuda_version: String,
    pub driver_version: String,
    pub compute_capability: String,
    pub precision: String,
    pub profile_shapes: String,
}

pub fn trt_profile_signature_from_shapes(
    min_shapes: &str,
    opt_shapes: &str,
    max_shapes: &str,
) -> String {
    format!("min={min_shapes};opt={opt_shapes};max={max_shapes};buckets=128,256,512,1024")
}

/// Signature used for cache invalidation when only the config is known (no introspection yet).
pub fn trt_profile_signature(config: &ModelConfig) -> String {
    let profiles =
        resolve_trt_dynamic_profiles(config, config.path.as_deref().map(std::path::Path::new));
    trt_profile_signature_from_shapes(
        &profiles.min_shapes,
        &profiles.opt_shapes,
        &profiles.max_shapes,
    )
}

/// Resolve the min/opt/max profile strings that will be (or were) applied to TensorRT.
/// This is the single source of truth for compile-time profiles and runtime bounds.
pub fn resolve_trt_dynamic_profiles(
    config: &ModelConfig,
    model_path: Option<&std::path::Path>,
) -> crate::engine::onnx_introspect::TrtDynamicProfiles {
    if let (Some(min), Some(opt), Some(max)) = (
        config.trt_profile_min_shapes.as_deref(),
        config.trt_profile_opt_shapes.as_deref(),
        config.trt_profile_max_shapes.as_deref(),
    ) {
        return crate::engine::onnx_introspect::TrtDynamicProfiles {
            min_shapes: min.to_string(),
            opt_shapes: opt.to_string(),
            max_shapes: max.to_string(),
        };
    }

    let len = crate::engine::provider_chain::trt_compile_max_seq(
        config.effective_n_ctx().or_else(|| {
            detect_model_max_position(
                model_path.or_else(|| config.path.as_deref().map(std::path::Path::new)),
            )
        }),
        config.pad_seq_len,
    );
    let max_batch = config.max_batch_size.clamp(1, 32);
    let path_to_inspect = model_path.or_else(|| config.path.as_deref().map(std::path::Path::new));

    if config.trt_profile_min_shapes.is_none() {
        if let Some(p) = path_to_inspect {
            if let Ok(inputs) = crate::engine::onnx_introspect::introspect_onnx_inputs(p) {
                if !inputs.is_empty() {
                    if let Some(auto) = crate::engine::onnx_introspect::synthesize_trt_profiles(
                        &inputs, max_batch, len,
                    ) {
                        // Allow partial overrides for opt/max when only some are set.
                        return crate::engine::onnx_introspect::TrtDynamicProfiles {
                            min_shapes: config
                                .trt_profile_min_shapes
                                .clone()
                                .unwrap_or(auto.min_shapes),
                            opt_shapes: config
                                .trt_profile_opt_shapes
                                .clone()
                                .unwrap_or(auto.opt_shapes),
                            max_shapes: config
                                .trt_profile_max_shapes
                                .clone()
                                .unwrap_or(auto.max_shapes),
                        };
                    }
                }
            }
        }
    }

    let min_shapes = config
        .trt_profile_min_shapes
        .clone()
        .unwrap_or_else(|| "input_ids:1x1,attention_mask:1x1".to_string());
    let opt_shapes = config.trt_profile_opt_shapes.clone().unwrap_or_else(|| {
        let opt_seq = (len / 2).clamp(64, len);
        format!("input_ids:1x{opt_seq},attention_mask:1x{opt_seq}")
    });
    let max_shapes = config
        .trt_profile_max_shapes
        .clone()
        .unwrap_or_else(|| format!("input_ids:{max_batch}x{len},attention_mask:{max_batch}x{len}"));
    crate::engine::onnx_introspect::TrtDynamicProfiles {
        min_shapes,
        opt_shapes,
        max_shapes,
    }
}

/// Read max_position_embeddings / tokenizer model_max_length from the model directory.
fn detect_model_max_position(model_path: Option<&std::path::Path>) -> Option<usize> {
    let path = model_path?;
    let dir = if path.is_file() { path.parent()? } else { path };
    for name in ["config.json", "tokenizer_config.json"] {
        let cfg_path = dir.join(name);
        let Ok(text) = std::fs::read_to_string(&cfg_path) else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
            continue;
        };
        for key in [
            "max_position_embeddings",
            "max_sequence_length",
            "model_max_length",
        ] {
            if let Some(m) = v.get(key).and_then(|x| x.as_u64()) {
                if m > 0 && m < 100_000_000 {
                    return Some(m as usize);
                }
            }
        }
    }
    None
}

pub fn manifest_matches(stored: &TrtCacheManifest, current: &TrtCacheManifest) -> bool {
    // profile_shapes describes the engine that was built. A newer compile-time preference
    // (e.g. widening max seq from 512 → 2048) must not force every Jetson cache to rebuild;
    // over-profile requests fall back to CUDA at runtime instead.
    stored.onnx_sha256 == current.onnx_sha256
        && stored.tensorrt_version == current.tensorrt_version
        && stored.cuda_version == current.cuda_version
        && stored.driver_version == current.driver_version
        && stored.compute_capability == current.compute_capability
        && stored.precision == current.precision
}

pub fn hash_file_sha256(path: &std::path::Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = std::io::Read::read(&mut file, &mut buf).ok()?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Some(format!("{:x}", hasher.finalize()))
}

pub fn current_trt_manifest(
    model_path: &std::path::Path,
    precision: &str,
    profile_shapes: &str,
) -> Option<TrtCacheManifest> {
    let (cuda_version, driver_version, compute_capability) = probe_nvidia_versions();
    Some(TrtCacheManifest {
        onnx_sha256: hash_file_sha256(model_path)?,
        tensorrt_version: probe_tensorrt_version(),
        cuda_version,
        driver_version,
        compute_capability,
        precision: precision.to_string(),
        profile_shapes: profile_shapes.to_string(),
    })
}

pub fn read_trt_manifest(cache_dir: &str) -> Option<TrtCacheManifest> {
    let path = std::path::Path::new(cache_dir).join(TRT_MANIFEST_FILE);
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

pub fn write_trt_manifest(cache_dir: &str, manifest: &TrtCacheManifest) -> std::io::Result<()> {
    let dir = std::path::Path::new(cache_dir);
    std::fs::create_dir_all(dir)?;
    let path = dir.join(TRT_MANIFEST_FILE);
    let body = serde_json::to_string_pretty(manifest)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
    std::fs::write(path, body)
}

/// Runtime profile bounds for a loaded (or about-to-load) TensorRT cache.
/// Prefers the manifest written at compile time; falls back to the historical
/// silent default of max_seq=512 when the manifest has empty shapes.
pub fn trt_runtime_profile_bounds(
    cache_dir: &str,
    config: &ModelConfig,
) -> crate::engine::provider_chain::TrtProfileBounds {
    if let Some(stored) = read_trt_manifest(cache_dir) {
        if let Some(bounds) =
            crate::engine::provider_chain::trt_profile_bounds_from_signature(&stored.profile_shapes)
        {
            return bounds;
        }
    }
    // Legacy caches / empty signatures were built with pad_seq_len.unwrap_or(512).
    let max_batch = config.max_batch_size.clamp(1, 32);
    crate::engine::provider_chain::TrtProfileBounds {
        min_batch: 1,
        max_batch,
        min_seq: 1,
        max_seq: config.pad_seq_len.unwrap_or(512).clamp(64, 4096),
        opt_batch: 1,
        opt_seq: 256.min(config.pad_seq_len.unwrap_or(512)),
    }
}

/// Engine bytes plus a matching manifest. A missing or different manifest recompiles.
pub fn trt_cache_is_current(cache_dir: &str, current: &TrtCacheManifest) -> bool {
    has_cached_trt_engine(cache_dir)
        && read_trt_manifest(cache_dir).is_some_and(|stored| manifest_matches(&stored, current))
}

pub fn quarantine_trt_cache(cache_dir: &str) -> std::io::Result<std::path::PathBuf> {
    let src = std::path::Path::new(cache_dir);
    if !src.exists() {
        return Ok(src.to_path_buf());
    }
    let dest = std::path::PathBuf::from(format!("{cache_dir}.quarantine"));
    if dest.exists() {
        std::fs::remove_dir_all(&dest)?;
    }
    std::fs::rename(src, &dest)?;
    Ok(dest)
}

fn probe_tensorrt_version() -> String {
    if let Ok(version) = std::env::var("TENSORRT_VERSION") {
        if !version.trim().is_empty() {
            return version.trim().to_string();
        }
    }
    "unknown".to_string()
}

fn probe_nvidia_versions() -> (String, String, String) {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=driver_version,compute_cap",
            "--format=csv,noheader",
        ])
        .output();
    let Ok(output) = output else {
        return (
            "unknown".to_string(),
            "unknown".to_string(),
            "unknown".to_string(),
        );
    };
    if !output.status.success() {
        return (
            "unknown".to_string(),
            "unknown".to_string(),
            "unknown".to_string(),
        );
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut parts = text.lines().next().unwrap_or("").split(',');
    let driver = parts.next().unwrap_or("unknown").trim();
    let cap = parts.next().unwrap_or("unknown").trim();
    let cuda = std::env::var("CUDA_VERSION").unwrap_or_else(|_| "unknown".to_string());
    (
        if cuda.is_empty() {
            "unknown".to_string()
        } else {
            cuda
        },
        if driver.is_empty() {
            "unknown".to_string()
        } else {
            driver.to_string()
        },
        if cap.is_empty() {
            "unknown".to_string()
        } else {
            cap.to_string()
        },
    )
}

/// Shared TensorRT builder knobs applied by every compile/load path.
/// Jetson uses a tight policy so Myelin autotune cannot exhaust unified memory;
/// discrete-GPU targets keep ORT's higher default optimization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrtBuilderPolicy {
    /// ORT provider option `trt_builder_optimization_level` (0–5; ORT default 3).
    pub optimization_level: u8,
    /// Hard cap for `trt_max_workspace_size` in MB (never multi-GB tactic scratch).
    pub max_workspace_mb: usize,
    /// ORT `trt_force_sequential_engine_build` — lowers peak UMA during build.
    pub force_sequential_engine_build: bool,
    /// ORT `trt_build_heuristics_enable` — prefer heuristics over heavy timing.
    pub build_heuristics: bool,
}

/// Jetson / tight unified-memory builder policy.
///
/// Level 0 (heuristic-only): ORT docs say levels below 3 trade engine quality for build
/// time; level 0 disables timing-based tactic selection. Level 1 still exercised Myelin
/// autotune hard enough to hit NvMap ENOMEM on 16GB Orin, so Jetson stays at 0.
/// There is no supported ORT `trt_tactic_sources` token that disables Myelin
/// (TRT TacticSource has no MYELIN flag), so we do not invent one.
pub fn jetson_trt_builder_policy() -> TrtBuilderPolicy {
    TrtBuilderPolicy {
        optimization_level: 0,
        max_workspace_mb: 2048,
        force_sequential_engine_build: true,
        // ORT warns this is deprecated on TRT ≥8.6 and only meaningful at level 2.
        build_heuristics: false,
    }
}

/// Discrete-GPU / non-Tegra builder policy (ORT-like defaults, still workspace-capped).
pub fn discrete_gpu_trt_builder_policy() -> TrtBuilderPolicy {
    TrtBuilderPolicy {
        optimization_level: 3,
        max_workspace_mb: 4096,
        force_sequential_engine_build: false,
        build_heuristics: false,
    }
}

/// Select the builder policy for this compile target.
pub fn trt_builder_policy() -> TrtBuilderPolicy {
    if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        jetson_trt_builder_policy()
    } else {
        discrete_gpu_trt_builder_policy()
    }
}

fn trt_workspace_mb(config: &ModelConfig, available_mb: f64) -> usize {
    let policy = trt_builder_policy();
    if let Some(user_req) = config.trt_max_workspace_mb {
        // Honor user-specified workspace, bounded only by safe available headroom
        let safe_ceiling = if available_mb > 2000.0 {
            ((available_mb - 1500.0).max(512.0) as usize).min(policy.max_workspace_mb)
        } else {
            512
        };
        return user_req.min(safe_ceiling).max(64);
    }

    // Dynamic scaling based on available system memory
    let dynamic_target = if available_mb >= 4000.0 {
        1024
    } else if available_mb >= 2000.0 {
        768
    } else {
        512
    };
    dynamic_target.min(policy.max_workspace_mb).max(64)
}

fn estimated_trt_build_memory_mb(model_size_mb: usize, workspace_mb: usize) -> usize {
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        // TensorRT duplicates weights during graph conversion and tactic selection. Jetson also
        // needs substantial contiguous UMA headroom or NvMap allocations can fail before OOM.
        model_size_mb
            .saturating_mul(4)
            .saturating_add(workspace_mb)
            .saturating_add(2048)
    }
    #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
    {
        (model_size_mb.saturating_mul(5) / 2)
            .saturating_add(workspace_mb)
            .saturating_add(512)
    }
}

/// Dynamically evaluates whether cold compilation of a TensorRT engine is physically
/// feasible without exhausting host/device memory or failing contiguous kernel allocations.
pub fn is_trt_compilation_feasible(
    config: &ModelConfig,
    _model_path: Option<&std::path::Path>,
    file_len: u64,
) -> (bool, String) {
    let base_cache_dir = trt_cache_dir(config);

    // Check base dir and precision-scoped subdirectories
    let has_cache = has_cached_trt_engine(&base_cache_dir)
        || has_cached_trt_engine(&format!("{}/int8", base_cache_dir))
        || has_cached_trt_engine(&format!("{}/fp16", base_cache_dir))
        || has_cached_trt_engine(&format!("{}/fp32", base_cache_dir));

    if has_cache {
        return (
            true,
            format!(
                "Found pre-compiled TensorRT engine cache for '{}'",
                config.name
            ),
        );
    }

    let snap = crate::engine::memory::get_memory_snapshot();
    let model_size_mb = (file_len / (1024 * 1024)) as usize;

    let workspace_mb = trt_workspace_mb(config, snap.system_available_mb);
    let required_mb = estimated_trt_build_memory_mb(model_size_mb, workspace_mb);

    // Check 1: General available system memory headroom
    if snap.system_available_mb > 0.0 && snap.system_available_mb < required_mb as f64 {
        return (
            false,
            format!(
                "Insufficient system memory for cold compilation (need ~{} MB, available: {:.0} MB)",
                required_mb, snap.system_available_mb
            ),
        );
    }

    // Check 2: NVIDIA Jetson / Tegra UMA hard watermark for cold compilation
    // Cold compiling any INT8/FP16 TensorRT engine requires a minimum of 2.5 GB free RAM on Jetson
    // to perform tactic search without risking OOM Killer panic.
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        if snap.system_available_mb > 0.0 && snap.system_available_mb < 2500.0 {
            return (
                false,
                format!(
                    "System available memory ({:.0} MB) is below the 2500 MB safe watermark for TensorRT cold compilation on Tegra unified memory",
                    snap.system_available_mb
                ),
            );
        }
    }

    (
        true,
        format!(
            "Memory pre-flight passed (need ~{} MB, available: {:.0} MB)",
            required_mb, snap.system_available_mb
        ),
    )
}

/// Scans up to `max_bytes` of an ONNX file in 64KB rolling windows to detect whether
/// it contains Einsum operator nodes or equations.
pub fn detect_einsum_in_file(path: &std::path::Path, max_bytes: usize) -> bool {
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };

    const CHUNK_SIZE: usize = 64 * 1024;
    const OVERLAP: usize = 32;
    let mut buf = vec![0u8; CHUNK_SIZE + OVERLAP];
    let mut total_read = 0;
    let mut overlap_len = 0;

    while total_read < max_bytes {
        let to_read = (max_bytes - total_read).min(CHUNK_SIZE);
        if to_read == 0 {
            break;
        }
        let n = match file.read(&mut buf[overlap_len..overlap_len + to_read]) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        let valid_len = overlap_len + n;
        let window = &buf[..valid_len];

        if window.windows(b"equation".len()).any(|w| w == b"equation")
            || window.windows(b"Einsum".len()).any(|w| w == b"Einsum")
        {
            return true;
        }

        total_read += n;
        if valid_len >= OVERLAP {
            buf.copy_within(valid_len - OVERLAP..valid_len, 0);
            overlap_len = OVERLAP;
        } else {
            overlap_len = 0;
        }
    }
    false
}

/// Checks if raw bytes contain QDQ (QuantizeLinear and DequantizeLinear) nodes.
pub fn detect_qdq_nodes(bytes: &[u8]) -> bool {
    let has_quantize = bytes
        .windows(b"QuantizeLinear".len())
        .any(|w| w == b"QuantizeLinear");
    let has_dequantize = bytes
        .windows(b"DequantizeLinear".len())
        .any(|w| w == b"DequantizeLinear");
    has_quantize && has_dequantize
}

/// Zero-allocation streaming scanner that searches up to `max_bytes` of an ONNX file
/// for QDQ nodes using a tiny 64 KB rolling buffer. Avoids allocating multi-megabyte vectors.
pub fn detect_qdq_in_file(path: &std::path::Path, max_bytes: usize) -> bool {
    use std::io::Read;
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };

    const BUF_SIZE: usize = 64 * 1024;
    const OVERLAP: usize = 32; // > "DequantizeLinear".len() (16)
    let mut buf = [0u8; BUF_SIZE];
    let mut total_read = 0;
    let mut has_quantize = false;
    let mut has_dequantize = false;

    let mut overlap_len = 0;
    while total_read < max_bytes {
        let to_read = (max_bytes - total_read).min(BUF_SIZE - overlap_len);
        if to_read == 0 {
            break;
        }
        let n = match file.read(&mut buf[overlap_len..overlap_len + to_read]) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        let valid_len = overlap_len + n;
        let window = &buf[..valid_len];

        if !has_quantize
            && window
                .windows(b"QuantizeLinear".len())
                .any(|w| w == b"QuantizeLinear")
        {
            has_quantize = true;
        }
        if !has_dequantize
            && window
                .windows(b"DequantizeLinear".len())
                .any(|w| w == b"DequantizeLinear")
        {
            has_dequantize = true;
        }
        if has_quantize && has_dequantize {
            return true;
        }

        total_read += n;
        // Keep the last OVERLAP bytes for cross-boundary matching
        if valid_len >= OVERLAP {
            buf.copy_within(valid_len - OVERLAP..valid_len, 0);
            overlap_len = OVERLAP;
        } else {
            overlap_len = 0;
        }
    }
    has_quantize && has_dequantize
}

/// Scans up to `max_bytes` of an ONNX file for operators characteristic of CPU dynamic quantization
/// (MatMulInteger, DynamicQuantizeLinear, QLinearMatMul, QLinearConv).
/// TensorRT cannot parse these operators, causing catastrophic graph partitioning into hundreds of micro-engines.
pub fn detect_dynamic_quantization_in_file(path: &std::path::Path, max_bytes: usize) -> bool {
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };

    const BUF_SIZE: usize = 64 * 1024;
    const OVERLAP: usize = 32;
    let mut buf = [0u8; BUF_SIZE];
    let mut total_read = 0;
    let mut overlap_len = 0;

    while total_read < max_bytes {
        let to_read = (max_bytes - total_read).min(BUF_SIZE - overlap_len);
        if to_read == 0 {
            break;
        }
        let n = match file.read(&mut buf[overlap_len..overlap_len + to_read]) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        let valid_len = overlap_len + n;
        let window = &buf[..valid_len];

        if window
            .windows(b"MatMulInteger".len())
            .any(|w| w == b"MatMulInteger")
            || window
                .windows(b"DynamicQuantizeLinear".len())
                .any(|w| w == b"DynamicQuantizeLinear")
            || window
                .windows(b"QLinearMatMul".len())
                .any(|w| w == b"QLinearMatMul")
            || window
                .windows(b"QLinearConv".len())
                .any(|w| w == b"QLinearConv")
        {
            return true;
        }

        total_read += n;
        if valid_len >= OVERLAP {
            buf.copy_within(valid_len - OVERLAP..valid_len, 0);
            overlap_len = OVERLAP;
        } else {
            overlap_len = 0;
        }
    }
    false
}

/// Detects the model precision using a robust two-layer strategy:
/// 1. File name / path heuristic check (fastest, zero I/O):
///    - Contains "_int8", "-int8", ".int8", "model_int8" → INT8
///    - Contains "_fp16", "-fp16", ".fp16", "half" → FP16
/// 2. If name doesn't specify (e.g. `model.onnx`), stream-scan the file header (max 4 MB) for QDQ nodes.
/// 3. Default fallback → FP32
pub fn detect_model_precision(path: &std::path::Path) -> ModelPrecision {
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_lowercase();

    if detect_qdq_in_file(path, 8 * 1024 * 1024) {
        return ModelPrecision::Int8;
    }

    if file_name.contains("_int8")
        || file_name.contains("-int8")
        || file_name.contains(".int8")
        || file_name.contains("model_int8")
    {
        return ModelPrecision::Int8;
    }

    if file_name.contains("_fp16")
        || file_name.contains("-fp16")
        || file_name.contains(".fp16")
        || file_name.contains("half")
    {
        return ModelPrecision::Fp16;
    }

    if file_name.contains("_fp32") || file_name.contains("-fp32") || file_name.contains(".fp32") {
        return ModelPrecision::Fp32;
    }

    ModelPrecision::Fp32
}

#[cfg(feature = "tensorrt")]
pub fn build_tensorrt_ep(
    config: &ModelConfig,
    precision: ModelPrecision,
    model_path: Option<&std::path::Path>,
) -> ort::ep::TensorRT {
    let mut trt = ort::ep::TensorRT::default()
        .with_context_memory_sharing(config.context_memory_sharing)
        .with_layer_norm_fp32_fallback(true);

    match precision {
        ModelPrecision::Int8 => {
            info!(
                model = %config.name,
                "Enabling TensorRT INT8 precision (with FP16 mixed-precision fallback for non-quantized ops)"
            );
            trt = trt.with_int8(true).with_fp16(true);
        }
        ModelPrecision::Fp16 => {
            info!(
                model = %config.name,
                "Enabling TensorRT FP16 precision"
            );
            trt = trt.with_int8(false).with_fp16(true);
        }
        ModelPrecision::Fp32 => {
            info!(
                model = %config.name,
                "Enabling TensorRT FP32 precision (with FP16 mixed-precision acceleration)"
            );
            trt = trt.with_int8(false).with_fp16(true);
        }
    }

    let snap = crate::engine::memory::get_memory_snapshot();
    let policy = trt_builder_policy();
    let workspace_mb = trt_workspace_mb(config, snap.system_available_mb);
    info!(
        model = %config.name,
        workspace_mb = workspace_mb,
        builder_optimization_level = policy.optimization_level,
        force_sequential_engine_build = policy.force_sequential_engine_build,
        build_heuristics = policy.build_heuristics,
        context_memory_sharing = config.context_memory_sharing,
        "Configuring TensorRT execution provider builder policy"
    );
    trt = trt
        .with_max_workspace_size(workspace_mb * 1024 * 1024)
        .with_builder_optimization_level(policy.optimization_level)
        .with_force_sequential_engine_build(policy.force_sequential_engine_build)
        .with_build_heuristics(policy.build_heuristics);

    let cache_dir = trt_cache_dir(config);

    if let Err(e) = std::fs::create_dir_all(&cache_dir) {
        warn!(
            model = %config.name,
            precision = %precision,
            "Failed to create TensorRT cache directory '{}': {}",
            cache_dir, e
        );
    } else {
        info!(
            model = %config.name,
            precision = %precision,
            cache_dir = %cache_dir,
            "Configuring TensorRT engine and timing cache"
        );
        let timing_cache_file = format!("{}/timing.cache", cache_dir);
        trt = trt
            .with_engine_cache(true)
            .with_engine_cache_path(&cache_dir)
            .with_timing_cache(true)
            .with_timing_cache_path(timing_cache_file);
    }

    let path_to_inspect = model_path.or_else(|| config.path.as_deref().map(std::path::Path::new));
    let profiles = applied_trt_profiles(config, &cache_dir, path_to_inspect);
    info!(
        model = %config.name,
        min = %profiles.min_shapes,
        opt = %profiles.opt_shapes,
        max = %profiles.max_shapes,
        "Applying TensorRT dynamic profiles"
    );
    trt = trt
        .with_profile_min_shapes(&profiles.min_shapes)
        .with_profile_opt_shapes(&profiles.opt_shapes)
        .with_profile_max_shapes(&profiles.max_shapes);

    trt
}

/// Profiles actually fed to the TensorRT EP for this cache/config pair.
///
/// Compile and load share this builder so they cannot drift. When a cache already
/// exists, batch/seq bounds come from the manifest (or the historical [1,1]..[4,512]
/// default); shape *names* always come from ONNX introspection so every dynamic
/// input (including `position_ids`) gets a profile.
pub fn applied_trt_profiles(
    config: &ModelConfig,
    cache_dir: &str,
    model_path: Option<&std::path::Path>,
) -> crate::engine::onnx_introspect::TrtDynamicProfiles {
    let path_to_inspect = model_path.or_else(|| config.path.as_deref().map(std::path::Path::new));

    // Explicit full overrides win.
    if let (Some(min), Some(opt), Some(max)) = (
        config.trt_profile_min_shapes.as_deref(),
        config.trt_profile_opt_shapes.as_deref(),
        config.trt_profile_max_shapes.as_deref(),
    ) {
        return crate::engine::onnx_introspect::TrtDynamicProfiles {
            min_shapes: min.to_string(),
            opt_shapes: opt.to_string(),
            max_shapes: max.to_string(),
        };
    }

    if has_cached_trt_engine(cache_dir) {
        let bounds = trt_runtime_profile_bounds(cache_dir, config);
        if let Some(p) = path_to_inspect {
            if let Ok(inputs) = crate::engine::onnx_introspect::introspect_onnx_inputs(p) {
                if !inputs.is_empty() {
                    if let Some(auto) = crate::engine::onnx_introspect::synthesize_trt_profiles(
                        &inputs,
                        bounds.max_batch,
                        bounds.max_seq,
                    ) {
                        info!(
                            model = %config.name,
                            max_batch = bounds.max_batch,
                            max_seq = bounds.max_seq,
                            inputs = inputs.len(),
                            "Aligning TensorRT EP profiles to cached engine bounds for all ONNX inputs"
                        );
                        return auto;
                    }
                }
            }
        }
        // Introspection failed: still emit at least the common transformer trio so
        // position_ids is not silently dropped the way a two-name legacy string was.
        return profiles_for_bounds(bounds);
    }

    resolve_trt_dynamic_profiles(config, path_to_inspect)
}

/// Universal NLP/transformer profile covering the inputs Qwen-style embedding graphs use.
fn profiles_for_bounds(
    bounds: crate::engine::provider_chain::TrtProfileBounds,
) -> crate::engine::onnx_introspect::TrtDynamicProfiles {
    let names = ["input_ids", "attention_mask", "position_ids"];
    let min = names
        .iter()
        .map(|n| format!("{n}:{}x{}", bounds.min_batch, bounds.min_seq))
        .collect::<Vec<_>>()
        .join(",");
    let opt = names
        .iter()
        .map(|n| format!("{n}:{}x{}", bounds.opt_batch, bounds.opt_seq))
        .collect::<Vec<_>>()
        .join(",");
    let max = names
        .iter()
        .map(|n| format!("{n}:{}x{}", bounds.max_batch, bounds.max_seq))
        .collect::<Vec<_>>()
        .join(",");
    crate::engine::onnx_introspect::TrtDynamicProfiles {
        min_shapes: min,
        opt_shapes: opt,
        max_shapes: max,
    }
}

/// True when ORT rejected the session because EP options omitted a dynamic input's profile.
/// This is a session-config bug, not a corrupt `.engine` file.
pub fn is_missing_trt_shape_profile_error(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    lower.contains("no associated shape profiles")
        || lower.contains("has no associated shape profiles")
        || lower.contains("provide all the dynamic shape inputs")
        || (lower.contains("dynamic shape") && lower.contains("profile"))
}

/// True when the engine bytes themselves look unusable (corruption / TRT mismatch).
pub fn is_corrupt_trt_engine_error(err: &str) -> bool {
    if is_missing_trt_shape_profile_error(err) {
        return false;
    }
    let lower = err.to_ascii_lowercase();
    lower.contains("engine")
        && (lower.contains("deserialize")
            || lower.contains("corrupt")
            || lower.contains("incompatible")
            || lower.contains("version")
            || lower.contains("truncated"))
}

pub fn configure_builder_for_provider_with_precision(
    builder: SessionBuilder,
    config: &ModelConfig,
    provider_name: &str,
    _trt_precision: ModelPrecision,
    _model_path: Option<&std::path::Path>,
) -> Result<SessionBuilder, ort::Error> {
    let normalized = provider_name.to_lowercase();
    match normalized.as_str() {
        "tensorrt" | "trt" => {
            #[cfg(feature = "tensorrt")]
            {
                info!(
                    model = %config.name,
                    precision = %_trt_precision,
                    "Configuring TensorRT Execution Provider with precision target..."
                );
                let trt = build_tensorrt_ep(config, _trt_precision, _model_path);

                // Register TensorRT standalone (do NOT attach CUDA EP inside the same session;
                // fallback is handled cleanly by the candidate providers loop in ModelSession::load)
                Ok(builder.with_execution_providers([trt.build()])?)
            }
            #[cfg(not(feature = "tensorrt"))]
            {
                return Err(ort::Error::new(
                    "TensorRT EP requested but crate feature 'tensorrt' was not compiled",
                ));
            }
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                info!(model = %config.name, "Configuring CUDA Execution Provider...");
                let cuda = build_cuda_ep(config);
                Ok(builder.with_execution_providers([cuda.build()])?)
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(ort::Error::new(
                    "CUDA EP requested but crate feature 'cuda' was not compiled",
                ));
            }
        }
        "coreml" => {
            #[cfg(feature = "coreml")]
            {
                info!(model = %config.name, "Configuring CoreML Execution Provider...");
                let coreml = ort::ep::CoreML::default();
                return Ok(builder.with_execution_providers([coreml.build()])?);
            }
            #[cfg(not(feature = "coreml"))]
            {
                Err(ort::Error::new(
                    "CoreML EP requested but crate feature 'coreml' was not compiled",
                ))
            }
        }
        "directml" | "dml" => {
            #[cfg(feature = "directml")]
            {
                info!(model = %config.name, "Configuring DirectML Execution Provider...");
                let dml = ort::ep::DirectML::default();
                return Ok(builder.with_execution_providers([dml.build()])?);
            }
            #[cfg(not(feature = "directml"))]
            {
                Err(ort::Error::new(
                    "DirectML EP requested but crate feature 'directml' was not compiled",
                ))
            }
        }
        "cpu" => {
            info!(model = %config.name, "Configuring CPU Execution Provider");
            Ok(builder)
        }
        unknown => Err(ort::Error::new(format!(
            "Unrecognized provider name '{}'",
            unknown
        ))),
    }
}

pub fn configure_builder_for_provider(
    builder: SessionBuilder,
    config: &ModelConfig,
    provider_name: &str,
) -> Result<SessionBuilder, ort::Error> {
    configure_builder_for_provider_with_precision(
        builder,
        config,
        provider_name,
        ModelPrecision::Fp16,
        None,
    )
}

pub fn configure_execution_providers(
    builder: SessionBuilder,
    config: &ModelConfig,
) -> Result<(SessionBuilder, String), ort::Error> {
    let candidates = get_candidate_providers(config);
    for p in &candidates {
        if let Ok(b) = configure_builder_for_provider(builder.clone(), config, p) {
            return Ok((b, p.clone()));
        }
    }
    Ok((builder, "CPU".to_string()))
}

#[cfg(test)]
mod tests {
    use super::estimated_trt_build_memory_mb;
    use super::{
        is_corrupt_trt_engine_error, is_missing_trt_shape_profile_error, jetson_trt_builder_policy,
        manifest_matches, profiles_for_bounds, trt_builder_policy, trt_cache_is_current,
        TrtCacheManifest,
    };
    use crate::engine::provider_chain::TrtProfileBounds;

    #[test]
    fn trt_build_estimate_includes_weights_workspace_and_headroom() {
        let estimated = estimated_trt_build_memory_mb(1024, 512);

        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        assert_eq!(estimated, 6656);

        #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
        assert_eq!(estimated, 3584);
    }

    #[test]
    fn jetson_builder_policy_sets_level_and_workspace() {
        let jetson = jetson_trt_builder_policy();
        assert_eq!(jetson.optimization_level, 0);
        assert_eq!(jetson.max_workspace_mb, 2048);
        assert!(jetson.force_sequential_engine_build);
        assert!(!jetson.build_heuristics);

        let selected = trt_builder_policy();
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        {
            assert_eq!(selected, jetson);
        }
        #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
        {
            assert_eq!(selected.optimization_level, 3);
            assert_eq!(selected.max_workspace_mb, 4096);
        }
    }

    #[test]
    fn arch_manifest_match_skips_recompile() {
        let current = TrtCacheManifest {
            onnx_sha256: "abc".to_string(),
            tensorrt_version: "10.0".to_string(),
            cuda_version: "12.4".to_string(),
            driver_version: "550".to_string(),
            compute_capability: "8.7".to_string(),
            precision: "FP16".to_string(),
            profile_shapes:
                "min=input_ids:1x1;opt=input_ids:1x256;max=input_ids:4x512;buckets=128,256,512,1024"
                    .to_string(),
        };
        assert!(manifest_matches(&current, &current));
        let mut changed = current.clone();
        changed.onnx_sha256 = "def".to_string();
        assert!(!manifest_matches(&current, &changed));
        changed.onnx_sha256 = current.onnx_sha256.clone();
        changed.compute_capability = "8.9".to_string();
        assert!(!manifest_matches(&current, &changed));
        // Widening intended compile profile must not invalidate an existing cache.
        changed.compute_capability = current.compute_capability.clone();
        changed.profile_shapes = "min=input_ids:1x1;max=input_ids:4x2048".to_string();
        assert!(manifest_matches(&current, &changed));
    }

    #[test]
    fn matching_manifest_does_not_request_recompile() {
        let dir = std::env::temp_dir().join(format!(
            "jetort_trt_reuse_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        // Minimal non-empty .engine so has_cached_trt_engine succeeds (size > 1KB).
        std::fs::write(dir.join("model.engine"), vec![0u8; 2048]).unwrap();
        let stored = TrtCacheManifest {
            onnx_sha256: "abc".to_string(),
            tensorrt_version: "10.0".to_string(),
            cuda_version: "12.4".to_string(),
            driver_version: "550".to_string(),
            compute_capability: "8.7".to_string(),
            precision: "FP16".to_string(),
            profile_shapes: "min=input_ids:1x1;max=input_ids:4x512".to_string(),
        };
        super::write_trt_manifest(dir.to_str().unwrap(), &stored).unwrap();
        let mut current = stored.clone();
        // Default formula now prefers 4x2048 — must still reuse the on-disk engine.
        current.profile_shapes = "min=input_ids:1x1;max=input_ids:4x2048".to_string();
        assert!(trt_cache_is_current(dir.to_str().unwrap(), &current));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_position_ids_profile_is_config_bug_not_corrupt_engine() {
        let err = "TensorRT cache deserialize failed (User needs to provide all the dynamic shape inputs with associated profiles ... Following input(s) has no associated shape profiles provided: position_ids)";
        assert!(is_missing_trt_shape_profile_error(err));
        assert!(!is_corrupt_trt_engine_error(err));
        assert!(is_corrupt_trt_engine_error(
            "failed to deserialize cuda engine: incompatible TensorRT version"
        ));
    }

    #[test]
    fn cached_profile_builder_covers_position_ids() {
        let bounds = TrtProfileBounds {
            min_batch: 1,
            max_batch: 4,
            min_seq: 1,
            max_seq: 512,
            opt_batch: 1,
            opt_seq: 256,
        };
        let profiles = profiles_for_bounds(bounds);
        assert!(profiles.max_shapes.contains("input_ids:4x512"));
        assert!(profiles.max_shapes.contains("attention_mask:4x512"));
        assert!(profiles.max_shapes.contains("position_ids:4x512"));
        assert!(profiles.min_shapes.contains("position_ids:1x1"));
    }
}
