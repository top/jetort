use crate::config::ModelConfig;
use ort::session::builder::SessionBuilder;
use std::io::Read;
use tracing::info;
#[cfg(feature = "tensorrt")]
use tracing::{debug, warn};

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
    let use_auto = config.providers.is_empty()
        || config
            .providers
            .iter()
            .any(|p| p.eq_ignore_ascii_case("auto"));

    let candidate_providers = if use_auto {
        let mut list = Vec::new();

        #[cfg(target_os = "macos")]
        {
            list.push("coreml".to_string());
            list.push("cpu".to_string());
        }

        #[cfg(target_os = "windows")]
        {
            list.push("tensorrt".to_string());
            list.push("cuda".to_string());
            list.push("directml".to_string());
            list.push("cpu".to_string());
        }

        #[cfg(all(
            target_os = "linux",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ))]
        {
            list.push("tensorrt".to_string());
            list.push("cuda".to_string());
            list.push("cpu".to_string());
        }

        #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
        {
            list.push("cpu".to_string());
        }
        list
    } else {
        let mut list = config.providers.clone();
        #[cfg(feature = "cuda")]
        if !list.iter().any(|p| p.eq_ignore_ascii_case("cuda")) {
            list.push("cuda".to_string());
        }
        if !list.iter().any(|p| p.eq_ignore_ascii_case("cpu")) {
            list.push("cpu".to_string());
        }
        list
    };

    candidate_providers
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

fn trt_workspace_mb(config: &ModelConfig, available_mb: f64) -> usize {
    config.trt_max_workspace_mb.unwrap_or_else(|| {
        if available_mb >= 4000.0 {
            512
        } else if available_mb >= 2000.0 {
            384
        } else {
            256
        }
    })
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
                "Enabling TensorRT FP32 precision (INT8 disabled, FP16 disabled)"
            );
            trt = trt.with_int8(false).with_fp16(false);
        }
    }

    let snap = crate::engine::memory::get_memory_snapshot();
    let workspace_mb = trt_workspace_mb(config, snap.system_available_mb);
    info!(
        model = %config.name,
        workspace_mb = workspace_mb,
        context_memory_sharing = config.context_memory_sharing,
        "Configuring TensorRT execution provider memory controls"
    );
    trt = trt.with_max_workspace_size(workspace_mb * 1024 * 1024);

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

    // Dynamic profile shapes configuration:
    // 1. If explicit profiles provided in ModelConfig, prioritize user configuration
    // 2. Otherwise, perform automated ONNX graph introspection to semantically deduce profiles
    // 3. If introspection unavailable, fall back to safe universal transformer profiles
    let path_to_inspect = model_path.or_else(|| config.path.as_deref().map(std::path::Path::new));
    let len = config.pad_seq_len.unwrap_or(512);
    let max_batch = config.max_batch_size.clamp(1, 8);

    let has_cache = has_cached_trt_engine(&cache_dir);
    if has_cache {
        info!(
            model = %config.name,
            cache_dir = %cache_dir,
            "⚡ Pre-compiled TensorRT engine cache detected; preserving engine cache profile alignment"
        );
    }

    let mut auto_profiles = None;
    // Always introspect the ONNX graph to synthesize profiles covering every dynamic input
    // (e.g. input_ids/attention_mask/position_ids), unless the user specified explicit profiles.
    // Introspection is cheap (just reads the ONNX header) and must run even when an engine
    // cache already exists on disk: TensorRT still requires an EP profile option string that
    // matches the graph's full input set at session-creation time, regardless of caching, and
    // an incomplete fallback profile (missing e.g. position_ids) causes a build failure and a
    // silent fallback to CUDA even though a valid cached engine could otherwise be reused.
    if config.trt_profile_min_shapes.is_none() {
        if let Some(p) = path_to_inspect {
            match crate::engine::onnx_introspect::introspect_onnx_inputs(p) {
                Ok(inputs) if !inputs.is_empty() => {
                    info!(
                        model = %config.name,
                        num_inputs = inputs.len(),
                        "Auto-introspected ONNX graph input tensors: {:?}",
                        inputs.iter().map(|i| (&i.name, &i.dims)).collect::<Vec<_>>()
                    );
                    auto_profiles = crate::engine::onnx_introspect::synthesize_trt_profiles(
                        &inputs, max_batch, len,
                    );
                }
                Ok(_) => {
                    debug!(model = %config.name, "No graph inputs found during ONNX introspection; using universal fallback");
                }
                Err(e) => {
                    debug!(model = %config.name, error = %e, "ONNX graph introspection skipped; using universal fallback");
                }
            }
        }
    }

    if let Some(ref min_s) = config.trt_profile_min_shapes {
        trt = trt.with_profile_min_shapes(min_s);
    } else if let Some(ref auto) = auto_profiles {
        info!(
            model = %config.name,
            min = %auto.min_shapes,
            opt = %auto.opt_shapes,
            max = %auto.max_shapes,
            "Applying auto-synthesized dynamic TensorRT profiles from ONNX graph introspection"
        );
        trt = trt
            .with_profile_min_shapes(&auto.min_shapes)
            .with_profile_opt_shapes(&auto.opt_shapes)
            .with_profile_max_shapes(&auto.max_shapes);
    } else {
        // Universal default dynamic profile shapes for NLP/transformer/reranker models.
        // This branch only runs if graph introspection above genuinely failed (e.g. unreadable
        // ONNX file), in which case the model is unlikely to build successfully regardless;
        // kept intentionally minimal/safe since profile entries for input names absent from
        // the actual graph can cause a hard TensorRT build error rather than a clean fallback.
        let min_shapes = "input_ids:1x1,attention_mask:1x1";
        let opt_shapes = format!("input_ids:1x{len},attention_mask:1x{len}");
        let max_shapes = format!("input_ids:{max_batch}x{len},attention_mask:{max_batch}x{len}");
        trt = trt
            .with_profile_min_shapes(min_shapes)
            .with_profile_opt_shapes(opt_shapes)
            .with_profile_max_shapes(max_shapes);
    }

    if let Some(ref opt_s) = config.trt_profile_opt_shapes {
        trt = trt.with_profile_opt_shapes(opt_s);
    }
    if let Some(ref max_s) = config.trt_profile_max_shapes {
        trt = trt.with_profile_max_shapes(max_s);
    }

    trt
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

    #[test]
    fn trt_build_estimate_includes_weights_workspace_and_headroom() {
        let estimated = estimated_trt_build_memory_mb(1024, 512);

        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        assert_eq!(estimated, 6656);

        #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
        assert_eq!(estimated, 3584);
    }
}
