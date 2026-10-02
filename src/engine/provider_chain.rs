//! One execution-provider chain for single-pass ONNX sessions.
//!
//! The name returned by [`Backend::canonical_name`] is the only provider string
//! logs, `/v1/models`, KServe metadata, and the session pool may use.

use crate::config::ModelConfig;

/// Backend this binary can actually execute on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    TensorRt,
    Cuda,
    CoreMl,
    DirectMl,
    Cpu,
}

impl Backend {
    /// Display name. This is also the value stored as `active_provider`.
    pub fn canonical_name(self) -> &'static str {
        match self {
            Backend::TensorRt => "TensorRT",
            Backend::Cuda => "CUDA",
            Backend::CoreMl => "CoreML",
            Backend::DirectMl => "DirectML",
            Backend::Cpu => "CPU",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        let trimmed = name.trim();
        let lower = trimmed.to_ascii_lowercase();
        let head = lower
            .split('(')
            .next()
            .unwrap_or(lower.as_str())
            .trim()
            .trim_end_matches(|c: char| c == ' ' || c == '-');
        match head {
            "tensorrt" | "trt" => Some(Backend::TensorRt),
            "cuda" => Some(Backend::Cuda),
            "coreml" => Some(Backend::CoreMl),
            "directml" | "dml" => Some(Backend::DirectMl),
            "cpu" => Some(Backend::Cpu),
            _ => None,
        }
    }

    pub fn is_compiled_in(self) -> bool {
        match self {
            Backend::Cpu => true,
            Backend::Cuda => cfg!(feature = "cuda"),
            Backend::TensorRt => cfg!(feature = "tensorrt"),
            Backend::CoreMl => cfg!(feature = "coreml"),
            Backend::DirectMl => cfg!(feature = "directml"),
        }
    }
}

/// Build target. Runtime walks only the backends compiled into this binary,
/// in the order of the target the binary was built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeployTarget {
    Jetson,
    LinuxNvidia,
    Macos,
    WindowsNvidia,
    WindowsOther,
    LinuxAmd,
    Cpu,
}

pub fn chain_for_target(target: DeployTarget) -> &'static [Backend] {
    match target {
        DeployTarget::Jetson | DeployTarget::LinuxNvidia | DeployTarget::WindowsNvidia => {
            &[Backend::TensorRt, Backend::Cuda, Backend::Cpu]
        }
        DeployTarget::Macos => &[Backend::CoreMl, Backend::Cpu],
        DeployTarget::WindowsOther => &[Backend::DirectMl, Backend::Cpu],
        DeployTarget::LinuxAmd | DeployTarget::Cpu => &[Backend::Cpu],
    }
}

pub fn compiled_runtime_target() -> DeployTarget {
    if cfg!(target_os = "macos") {
        DeployTarget::Macos
    } else if cfg!(target_os = "windows") {
        if cfg!(feature = "cuda") || cfg!(feature = "tensorrt") {
            DeployTarget::WindowsNvidia
        } else {
            DeployTarget::WindowsOther
        }
    } else if cfg!(target_os = "linux")
        && cfg!(target_arch = "aarch64")
        && (cfg!(feature = "cuda") || cfg!(feature = "tensorrt"))
    {
        DeployTarget::Jetson
    } else if cfg!(target_os = "linux") && (cfg!(feature = "cuda") || cfg!(feature = "tensorrt")) {
        DeployTarget::LinuxNvidia
    } else {
        DeployTarget::Cpu
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FallbackReason {
    LibraryMissing { detail: String },
    DeviceMissing { detail: String },
    EpNotCompiled { detail: String },
    SessionCreateFailed { detail: String },
}

impl FallbackReason {
    pub fn kind_label(&self) -> &'static str {
        match self {
            FallbackReason::LibraryMissing { .. } => "library missing",
            FallbackReason::DeviceMissing { .. } => "device missing",
            FallbackReason::EpNotCompiled { .. } => "EP not compiled",
            FallbackReason::SessionCreateFailed { .. } => "session create failed",
        }
    }

    fn detail(&self) -> &str {
        match self {
            FallbackReason::LibraryMissing { detail }
            | FallbackReason::DeviceMissing { detail }
            | FallbackReason::EpNotCompiled { detail }
            | FallbackReason::SessionCreateFailed { detail } => detail,
        }
    }

    /// Warning text. Callers log this with `tracing::warn`.
    pub fn warning_line(&self, fallen_back_to: &str) -> String {
        format!(
            "⚠️ Falling back to {}: {} ({})",
            fallen_back_to,
            self.kind_label(),
            self.detail()
        )
    }
}

pub fn classify_provider_failure(err: &str) -> FallbackReason {
    let lower = err.to_ascii_lowercase();
    if lower.contains("not compiled") || lower.contains("was not compiled") {
        FallbackReason::EpNotCompiled {
            detail: err.to_string(),
        }
    } else if lower.contains("libnvinfer")
        || lower.contains("library")
        || lower.contains("dlopen")
        || lower.contains("cannot open shared object")
        || lower.contains("image not found")
    {
        FallbackReason::LibraryMissing {
            detail: err.to_string(),
        }
    } else if lower.contains("no cuda")
        || lower.contains("device")
        || lower.contains("driver")
        || lower.contains("insufficient")
        || lower.contains("memory")
    {
        FallbackReason::DeviceMissing {
            detail: err.to_string(),
        }
    } else {
        FallbackReason::SessionCreateFailed {
            detail: err.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SelectionPlan {
    pub requested: Backend,
    pub walk: Vec<Backend>,
    pub fallback_warnings: Vec<String>,
}

/// Preferred order for this binary, then drop backends that were not compiled in.
/// Each dropped backend produces a fallback warning whose target is the next
/// backend that will actually be tried.
pub fn selection_plan(configured: &[String]) -> SelectionPlan {
    let auto = configured.is_empty()
        || configured
            .iter()
            .any(|provider| provider.eq_ignore_ascii_case("auto"));
    let preferred: Vec<Backend> = if auto {
        chain_for_target(compiled_runtime_target()).to_vec()
    } else {
        let mut list = Vec::new();
        for name in configured {
            if name.eq_ignore_ascii_case("auto") {
                continue;
            }
            if let Some(backend) = Backend::parse(name) {
                if !list.contains(&backend) {
                    list.push(backend);
                }
            }
        }
        if !list.contains(&Backend::Cpu) {
            list.push(Backend::Cpu);
        }
        list
    };

    let walk: Vec<Backend> = preferred
        .iter()
        .copied()
        .filter(|backend| backend.is_compiled_in())
        .collect();
    let mut fallback_warnings = Vec::new();
    for (index, backend) in preferred.iter().enumerate() {
        if backend.is_compiled_in() {
            continue;
        }
        let target = preferred[index + 1..]
            .iter()
            .find(|next| next.is_compiled_in())
            .or_else(|| walk.last());
        if let Some(target) = target {
            let reason = FallbackReason::EpNotCompiled {
                detail: format!(
                    "{} is not compiled into this binary",
                    backend.canonical_name()
                ),
            };
            fallback_warnings.push(reason.warning_line(target.canonical_name()));
        }
    }
    let requested = walk.first().copied().unwrap_or(Backend::Cpu);
    SelectionPlan {
        requested,
        walk,
        fallback_warnings,
    }
}

pub fn candidate_provider_names(config: &ModelConfig) -> Vec<String> {
    selection_plan(&config.providers)
        .walk
        .into_iter()
        .map(|backend| backend.canonical_name().to_string())
        .collect()
}

pub fn is_tensorrt_name(name: &str) -> bool {
    Backend::parse(name) == Some(Backend::TensorRt)
}

/// Canonical provider string. Unknown labels stay unchanged so a parse miss
/// is visible instead of being rewritten into a GPU name.
pub fn canonical_provider_name(name: &str) -> String {
    Backend::parse(name)
        .map(|backend| backend.canonical_name().to_string())
        .unwrap_or_else(|| name.trim().to_string())
}

pub const TRT_PAD_BUCKETS: [usize; 4] = [128, 256, 512, 1024];

/// Hard cap when synthesizing a new TensorRT profile max sequence.
/// On Jetson UMA architectures, multi-layer transformers (like Qwen3 28L) require
/// quadratic tactic workspace; capping default auto-profile sequence to 512 prevents NvMap ENOMEM.
/// Longer inputs seamlessly fall back to CUDA/CPU at runtime.
#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
pub const TRT_PROFILE_SEQ_CAP: usize = 512;

#[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
pub const TRT_PROFILE_SEQ_CAP: usize = 2048;


/// Active TensorRT optimization-profile bounds for one loaded engine.
/// Runtime must never call setInputShape outside these bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrtProfileBounds {
    pub min_batch: usize,
    pub max_batch: usize,
    pub min_seq: usize,
    pub max_seq: usize,
    pub opt_batch: usize,
    pub opt_seq: usize,
}

impl TrtProfileBounds {
    pub fn default_safe() -> Self {
        Self {
            min_batch: 1,
            max_batch: 4,
            min_seq: 1,
            max_seq: 512,
            opt_batch: 1,
            opt_seq: 256,
        }
    }

    pub fn contains(&self, batch: usize, seq: usize) -> bool {
        batch >= self.min_batch
            && batch <= self.max_batch
            && seq >= self.min_seq
            && seq <= self.max_seq
    }

    pub fn range_label(&self) -> String {
        format!(
            "[{},{}]..[{},{}]",
            self.min_batch, self.min_seq, self.max_batch, self.max_seq
        )
    }
}

/// How a request should hit the provider chain relative to the loaded TRT profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeRoute {
    /// Run on TensorRT. `microbatch` is the per-forward batch size (<= profile max).
    TensorRt { seq_pad: usize, microbatch: usize },
    /// Leave TensorRT; next backend in the chain handles the full request.
    Fallback {
        reason: String,
        actual_batch: usize,
        actual_seq: usize,
    },
}

/// Parse `NxM` from a TensorRT profile fragment like `input_ids:4x512`.
fn parse_batch_seq_token(token: &str) -> Option<(usize, usize)> {
    let dims = token
        .rsplit_once(':')
        .map(|(_, dims)| dims)
        .unwrap_or(token);
    let mut parts = dims.split('x');
    let batch = parts.next()?.parse::<usize>().ok()?;
    let seq = parts.next()?.parse::<usize>().ok()?;
    Some((batch, seq))
}

/// Prefer `input_ids` when present; otherwise first `BxS` pair in the shape string.
fn first_batch_seq(shapes: &str) -> Option<(usize, usize)> {
    let mut fallback = None;
    for part in shapes.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let Some((batch, seq)) = parse_batch_seq_token(part) else {
            continue;
        };
        let name = part.split(':').next().unwrap_or("").to_ascii_lowercase();
        if name.contains("input_ids") || name == "input_ids" {
            return Some((batch, seq));
        }
        if fallback.is_none() {
            fallback = Some((batch, seq));
        }
    }
    fallback
}

/// Build bounds from the min/opt/max shape strings written into the TRT manifest.
pub fn trt_profile_bounds_from_shapes(
    min_shapes: &str,
    opt_shapes: &str,
    max_shapes: &str,
) -> Option<TrtProfileBounds> {
    let (min_batch, min_seq) = first_batch_seq(min_shapes)?;
    let (opt_batch, opt_seq) = first_batch_seq(opt_shapes).unwrap_or((min_batch, min_seq));
    let (max_batch, max_seq) = first_batch_seq(max_shapes)?;
    if max_batch == 0 || max_seq == 0 {
        return None;
    }
    Some(TrtProfileBounds {
        min_batch: min_batch.max(1),
        max_batch,
        min_seq: min_seq.max(1),
        max_seq,
        opt_batch: opt_batch.max(1),
        opt_seq: opt_seq.max(1),
    })
}

/// Parse bounds from a compact manifest signature:
/// `min=...;opt=...;max=...;buckets=...` or a bare `max=` string.
pub fn trt_profile_bounds_from_signature(signature: &str) -> Option<TrtProfileBounds> {
    let mut min_s = String::new();
    let mut opt_s = String::new();
    let mut max_s = String::new();
    for part in signature.split(';') {
        let part = part.trim();
        if let Some(rest) = part.strip_prefix("min=") {
            min_s = rest.to_string();
        } else if let Some(rest) = part.strip_prefix("opt=") {
            opt_s = rest.to_string();
        } else if let Some(rest) = part.strip_prefix("max=") {
            max_s = rest.to_string();
        }
    }
    if max_s.is_empty() {
        return None;
    }
    if min_s.is_empty() {
        min_s = "input_ids:1x1".to_string();
    }
    if opt_s.is_empty() {
        opt_s = max_s.clone();
    }
    trt_profile_bounds_from_shapes(&min_s, &opt_s, &max_s)
}

/// Profile max sequence used when *compiling* a new engine.
/// Explicit `pad_seq_len` wins when set; otherwise min(model context, [`TRT_PROFILE_SEQ_CAP`]).
/// Never silently stick at 512 when the model advertises a larger context, and never build
/// a 131072-seq profile on Jetson.
pub fn trt_compile_max_seq(n_ctx: Option<usize>, pad_seq_len: Option<usize>) -> usize {
    let from_pad = pad_seq_len.filter(|v| *v > 0);
    if let Some(pad) = from_pad {
        return pad.clamp(64, 4096);
    }
    let from_ctx = n_ctx.filter(|v| *v > 0).unwrap_or(512);
    from_ctx.clamp(64, TRT_PROFILE_SEQ_CAP)
}

/// Smallest pad bucket that can hold `seq_len` and still fits inside the loaded profile.
pub fn trt_pad_bucket(seq_len: usize, profile_max_seq: usize) -> Option<usize> {
    if profile_max_seq == 0 {
        return None;
    }
    TRT_PAD_BUCKETS
        .into_iter()
        .find(|bucket| *bucket >= seq_len && *bucket <= profile_max_seq)
        .or_else(|| {
            // No named bucket fits; if the real length itself is inside the profile, pad to it.
            if seq_len <= profile_max_seq {
                Some(seq_len.max(1))
            } else {
                None
            }
        })
}

/// Decide TensorRT vs next-backend for one forward shape.
///
/// - Sequence longer than profile max → fallback (never pad to an unbuilt bucket).
/// - Batch larger than profile max but sequence fits → stay on TensorRT and microbatch.
/// - Both fit → pad inside the profile only.
pub fn plan_shape_route(batch: usize, seq_len: usize, profile: &TrtProfileBounds) -> ShapeRoute {
    let batch = batch.max(1);
    let seq_len = seq_len.max(1);
    if seq_len > profile.max_seq {
        return ShapeRoute::Fallback {
            reason: format!(
                "shape outside TensorRT profile: actual=[{batch},{seq_len}] profile={}",
                profile.range_label()
            ),
            actual_batch: batch,
            actual_seq: seq_len,
        };
    }
    let Some(seq_pad) = trt_pad_bucket(seq_len, profile.max_seq) else {
        return ShapeRoute::Fallback {
            reason: format!(
                "shape outside TensorRT profile: actual=[{batch},{seq_len}] profile={}",
                profile.range_label()
            ),
            actual_batch: batch,
            actual_seq: seq_len,
        };
    };
    let microbatch = batch.min(profile.max_batch.max(1));
    if microbatch == 0 {
        return ShapeRoute::Fallback {
            reason: format!(
                "shape outside TensorRT profile: actual=[{batch},{seq_len}] profile={}",
                profile.range_label()
            ),
            actual_batch: batch,
            actual_seq: seq_len,
        };
    }
    ShapeRoute::TensorRt {
        seq_pad: seq_pad.max(8),
        microbatch,
    }
}

/// Warning line when a request leaves TensorRT because of profile bounds.
pub fn shape_fallback_warning(fallen_back_to: &str, reason: &str) -> String {
    format!(
        "⚠️ Falling back to {}: shape outside TensorRT profile ({})",
        fallen_back_to, reason
    )
}

/// Sequence length used for a forward pass.
/// Explicit `pad_seq_len` wins. TensorRT pads only inside `profile` (never above profile max).
/// CUDA/CPU stay dynamic up to `dynamic_cap`.
pub fn resolve_seq_pad(
    explicit_pad: Option<usize>,
    active_provider: &str,
    seq_len: usize,
    n_ctx: usize,
    dynamic_cap: usize,
    profile: Option<&TrtProfileBounds>,
) -> usize {
    if let Some(cap) = explicit_pad {
        let capped = match profile {
            Some(p) if is_tensorrt_name(active_provider) => cap.min(p.max_seq),
            _ => cap,
        };
        return seq_len.min(capped).max(8);
    }
    if is_tensorrt_name(active_provider) {
        let max_seq = profile.map(|p| p.max_seq).unwrap_or(n_ctx.max(1));
        if let Some(bucket) = trt_pad_bucket(seq_len.max(1), max_seq) {
            return bucket.max(8);
        }
        // Outside profile: caller should have routed to CUDA already; keep a safe dynamic length.
        return seq_len.min(dynamic_cap).min(max_seq.max(8)).max(8);
    }
    seq_len.min(dynamic_cap).max(8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_canonical_provider_name_strips_precision_decoration() {
        assert_eq!(canonical_provider_name("TensorRT (INT8)"), "TensorRT");
        assert_eq!(canonical_provider_name("trt"), "TensorRT");
        assert_eq!(canonical_provider_name("CUDA"), "CUDA");
        assert_eq!(canonical_provider_name("cpu"), "CPU");
        assert_eq!(canonical_provider_name("DirectML"), "DirectML");
        assert!(is_tensorrt_name("TensorRT (INT8)"));
        assert!(!is_tensorrt_name("CUDA"));
    }

    #[test]
    fn arch_chain_for_each_target() {
        assert_eq!(
            chain_for_target(DeployTarget::Jetson),
            &[Backend::TensorRt, Backend::Cuda, Backend::Cpu]
        );
        assert_eq!(
            chain_for_target(DeployTarget::LinuxNvidia),
            chain_for_target(DeployTarget::Jetson)
        );
        assert_eq!(
            chain_for_target(DeployTarget::WindowsNvidia),
            chain_for_target(DeployTarget::Jetson)
        );
        assert_eq!(
            chain_for_target(DeployTarget::Macos),
            &[Backend::CoreMl, Backend::Cpu]
        );
        assert_eq!(
            chain_for_target(DeployTarget::WindowsOther),
            &[Backend::DirectMl, Backend::Cpu]
        );
        assert_eq!(chain_for_target(DeployTarget::LinuxAmd), &[Backend::Cpu]);
        assert_eq!(chain_for_target(DeployTarget::Cpu), &[Backend::Cpu]);
        assert!(chain_for_target(DeployTarget::LinuxAmd)
            .iter()
            .all(|backend| *backend == Backend::Cpu));
    }

    #[test]
    fn arch_selection_walks_only_compiled_backends() {
        let plan = selection_plan(&["auto".to_string()]);
        assert!(plan.walk.iter().all(|backend| backend.is_compiled_in()));
        assert_eq!(plan.walk.last().copied(), Some(Backend::Cpu));
        assert_eq!(plan.requested, plan.walk[0]);
    }

    #[test]
    fn arch_fallback_warning_names_target_and_reason() {
        let line = FallbackReason::LibraryMissing {
            detail: "libnvinfer.so".to_string(),
        }
        .warning_line("CUDA");
        assert!(line.contains('⚠'));
        assert!(line.contains("Falling back to CUDA"));
        assert!(line.contains("library missing"));
        assert!(line.contains("libnvinfer.so"));

        let device = classify_provider_failure("insufficient system memory");
        assert!(device.warning_line("CPU").contains("device missing"));
        let missing = classify_provider_failure("libnvinfer.so: cannot open shared object");
        assert!(missing.warning_line("CUDA").contains("library missing"));
        let compiled = classify_provider_failure("feature 'tensorrt' was not compiled");
        assert!(compiled.warning_line("CUDA").contains("EP not compiled"));
        let session = classify_provider_failure("IBuilder::buildSerializedNetwork failed");
        assert!(session
            .warning_line("CUDA")
            .contains("session create failed"));
    }

    #[test]
    fn arch_trt_pad_buckets() {
        // Buckets are constrained by the loaded profile max, not by n_ctx alone.
        assert_eq!(trt_pad_bucket(1, 512), Some(128));
        assert_eq!(trt_pad_bucket(128, 512), Some(128));
        assert_eq!(trt_pad_bucket(129, 512), Some(256));
        assert_eq!(trt_pad_bucket(512, 512), Some(512));
        assert_eq!(trt_pad_bucket(513, 512), None);
        assert_eq!(trt_pad_bucket(200, 256), Some(256));
        assert_eq!(trt_pad_bucket(200, 128), None);
        // Length 1024 vs profile max 512 must not invent a 1024 pad.
        assert_eq!(trt_pad_bucket(1024, 512), None);
        assert_eq!(trt_pad_bucket(600, 512), None);
        assert_eq!(trt_pad_bucket(600, 1024), Some(1024));

        let profile = TrtProfileBounds {
            min_batch: 1,
            max_batch: 4,
            min_seq: 1,
            max_seq: 512,
            opt_batch: 1,
            opt_seq: 256,
        };
        assert_eq!(resolve_seq_pad(None, "CUDA", 100, 4096, 512, None), 100);
        assert_eq!(
            resolve_seq_pad(None, "TensorRT", 200, 4096, 512, Some(&profile)),
            256
        );
        assert_eq!(
            resolve_seq_pad(Some(64), "TensorRT", 100, 4096, 512, Some(&profile)),
            64
        );
        assert_ne!(resolve_seq_pad(None, "CUDA", 32, 4096, 512, None), 4096);
        // Never pad TensorRT to 1024 when the engine profile max is 512.
        assert_ne!(
            resolve_seq_pad(None, "TensorRT", 600, 8192, 1024, Some(&profile)),
            1024
        );
    }

    #[test]
    fn arch_profile_route_1024_vs_512_falls_to_cuda() {
        let profile = TrtProfileBounds {
            min_batch: 1,
            max_batch: 4,
            min_seq: 1,
            max_seq: 512,
            opt_batch: 1,
            opt_seq: 256,
        };
        match plan_shape_route(1, 1024, &profile) {
            ShapeRoute::Fallback {
                actual_batch,
                actual_seq,
                reason,
            } => {
                assert_eq!(actual_batch, 1);
                assert_eq!(actual_seq, 1024);
                assert!(reason.contains("shape outside TensorRT profile"));
                assert!(reason.contains("[1,1024]"));
                assert!(reason.contains("[1,1]..[4,512]"));
            }
            ShapeRoute::TensorRt { seq_pad, .. } => {
                panic!("must not pad to TensorRT bucket {seq_pad} outside profile")
            }
        }
        let warn = shape_fallback_warning(
            "CUDA",
            "shape outside TensorRT profile: actual=[1,1024] profile=[1,1]..[4,512]",
        );
        assert!(warn.contains('⚠'));
        assert!(warn.contains("CUDA"));
        assert!(warn.contains("shape outside TensorRT profile"));
    }

    #[test]
    fn arch_microbatch_when_batch_exceeds_profile() {
        let profile = TrtProfileBounds {
            min_batch: 1,
            max_batch: 4,
            min_seq: 1,
            max_seq: 512,
            opt_batch: 1,
            opt_seq: 256,
        };
        // 10 items, each sequence fits → stay on TensorRT with microbatch 4.
        match plan_shape_route(10, 200, &profile) {
            ShapeRoute::TensorRt {
                seq_pad,
                microbatch,
            } => {
                assert_eq!(seq_pad, 256);
                assert_eq!(microbatch, 4);
            }
            ShapeRoute::Fallback { reason, .. } => {
                panic!("batch overflow should microbatch, not fallback: {reason}")
            }
        }
        // Single item over profile max → leave TensorRT (microbatch cannot help).
        assert!(matches!(
            plan_shape_route(10, 800, &profile),
            ShapeRoute::Fallback { .. }
        ));
        // In-profile stays on TensorRT without microbatch shrink.
        match plan_shape_route(2, 100, &profile) {
            ShapeRoute::TensorRt {
                seq_pad,
                microbatch,
            } => {
                assert_eq!(seq_pad, 128);
                assert_eq!(microbatch, 2);
            }
            ShapeRoute::Fallback { .. } => panic!("in-profile must stay on TensorRT"),
        }
    }

    #[test]
    fn arch_parse_profile_bounds_from_manifest_shapes() {
        let bounds = trt_profile_bounds_from_shapes(
            "input_ids:1x1,attention_mask:1x1",
            "input_ids:1x256,attention_mask:1x256",
            "input_ids:4x512,attention_mask:4x512",
        )
        .expect("parse");
        assert_eq!(bounds.max_batch, 4);
        assert_eq!(bounds.max_seq, 512);
        assert_eq!(bounds.min_seq, 1);
        assert_eq!(bounds.opt_seq, 256);

        let from_sig = trt_profile_bounds_from_signature(
            "min=input_ids:1x1,attention_mask:1x1;opt=input_ids:1x256,attention_mask:1x256;max=input_ids:4x512,attention_mask:4x512;buckets=128,256,512,1024",
        )
        .expect("signature");
        assert_eq!(from_sig, bounds);

        assert_eq!(trt_compile_max_seq(Some(8192), None), TRT_PROFILE_SEQ_CAP);
        #[cfg(not(all(target_os = "linux", target_arch = "aarch64")))]
        assert_eq!(trt_compile_max_seq(Some(1024), None), 1024);
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        assert_eq!(trt_compile_max_seq(Some(1024), None), TRT_PROFILE_SEQ_CAP);
        assert_eq!(trt_compile_max_seq(Some(256), None), 256);
        assert_eq!(trt_compile_max_seq(None, Some(512)), 512);
        // Large model context must cap at TRT_PROFILE_SEQ_CAP, not silent 512 and not raw 131072.
        assert_eq!(trt_compile_max_seq(Some(131072), None), TRT_PROFILE_SEQ_CAP);
        assert_eq!(trt_compile_max_seq(Some(131072), Some(768)), 768);
    }
}
