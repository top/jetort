use clap::Parser;
use jetort::{config::AppConfig, engine::ModelRegistry, server::create_router};
use std::env;
use std::net::SocketAddr;
use std::time::Instant;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
#[command(
    name = "jetort",
    about = "Ultra-lightweight, zero-overhead one-shot tensor inference engine & server"
)]
struct Args {
    #[arg(short, long, default_value = "config.yaml")]
    config: String,

    #[arg(long, hide = true)]
    compile_model: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut config = AppConfig::load_from_file(&args.config)?;
    init_tracing(&config);
    let startup = Instant::now();
    info!(
        config = %args.config,
        models = config.models.len(),
        "server startup begin"
    );
    info!(
        cuda = cfg!(feature = "cuda"),
        tensorrt = cfg!(feature = "tensorrt"),
        coreml = cfg!(feature = "coreml"),
        directml = cfg!(feature = "directml"),
        cuda_arch = option_env!("JETORT_CUDA_ARCH").unwrap_or(""),
        "compiled execution providers"
    );

    // Enable CUDA lazy loading to reduce initialization device memory footprint and speed up TensorRT startup
    if env::var("CUDA_MODULE_LOADING").is_err() {
        env::set_var("CUDA_MODULE_LOADING", "LAZY");
    }

    // Initialize ONNX Runtime library from explicit path or system default
    init_onnxruntime(&config)?;

    if let Some(model_name) = args.compile_model.as_deref() {
        compile_model_cache(&config, model_name)?;
        return Ok(());
    }

    // Cold TensorRT builds happen before the registry starts. A failed build
    // removes TensorRT from that model's provider list for this process, so
    // request-time loading cannot accidentally start the same build again.
    let failed_compiles = precompile_missing_caches(&config, &args.config)?;
    for model in &mut config.models {
        if failed_compiles.iter().any(|name| name == &model.name) {
            model.providers = providers_without_tensorrt(&model.providers);
            warn!(
                model = %model.name,
                providers = ?model.providers,
                "TensorRT compilation failed; this process will serve on the fallback provider"
            );
        }
    }

    // Initialize Model Registry
    let registry = ModelRegistry::new();
    registry.init_from_config(&config).await;
    registry.start_memory_monitor();

    // Create Axum HTTP Router
    let app = create_router(registry);

    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port).parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!(
        address = %addr,
        startup_ms = startup.elapsed().as_millis() as u64,
        "server listening"
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Service shut down gracefully");
    Ok(())
}

fn compile_model_cache(
    config: &AppConfig,
    model_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_config = config
        .models
        .iter()
        .find(|model| model.name == model_name)
        .ok_or_else(|| format!("Model '{model_name}' is not configured"))?;
    let cache_dir = jetort::engine::provider::trt_cache_dir(model_config);
    if let Some(manifest) = trt_manifest_for(model_config) {
        if jetort::engine::provider::trt_cache_is_current(&cache_dir, &manifest) {
            info!(
                model = model_name,
                cache_dir, "TensorRT cache manifest matches; skipping compile"
            );
            return Ok(());
        }
    } else if jetort::engine::provider::has_cached_trt_engine(&cache_dir) {
        info!(
            model = model_name,
            cache_dir, "TensorRT cache already available"
        );
        return Ok(());
    }

    // This command runs in a short-lived child process. Dropping the session
    // and exiting releases Jetson's TensorRT builder allocations before the
    // parent starts the next model.
    info!(
        model = model_name,
        "isolated TensorRT cache compilation started"
    );
    let session = jetort::engine::session::ModelSession::load(model_config)?;
    drop(session);
    if !jetort::engine::provider::has_cached_trt_engine(&cache_dir) {
        return Err(format!(
            "Model '{model_name}' did not produce a TensorRT cache; inspect provider fallback logs"
        )
        .into());
    }
    info!(
        model = model_name,
        cache_dir, "isolated TensorRT cache compilation completed"
    );
    Ok(())
}

fn precompile_missing_caches(
    config: &AppConfig,
    config_path: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    if !cfg!(feature = "tensorrt") {
        return Ok(Vec::new());
    }

    let executable = std::env::current_exe()?;
    let mut pending = Vec::new();
    for model in &config.models {
        if !jetort::engine::provider::get_candidate_providers(model)
            .iter()
            .any(|provider| provider.eq_ignore_ascii_case("tensorrt"))
        {
            continue;
        }
        let Ok((model_path, _)) = jetort::engine::session::resolve_model_and_tokenizer_path(model)
        else {
            continue;
        };
        let inspection = jetort::engine::session::inspect_or_fallback(&model_path, model);
        let probe = jetort::engine::header_prober::probe_model_bundle(
            &model_path,
            Some(&inspection),
            model.engine_kind,
        );
        if probe.kind != jetort::config::EngineKind::OneShot {
            continue;
        }
        let cache_dir = jetort::engine::provider::trt_cache_dir(model);
        let current = manifest_for_path(model, &model_path, &inspection);
        let reusable = current.as_ref().is_some_and(|manifest| {
            jetort::engine::provider::trt_cache_is_current(&cache_dir, manifest)
        });
        if !reusable {
            pending.push(model.name.clone());
        }
    }

    info!(
        models_to_compile = pending.len(),
        "TensorRT cache discovery completed"
    );
    let mut failed = Vec::new();
    for (index, model_name) in pending.iter().enumerate() {
        info!(
            model = model_name,
            progress = index + 1,
            total = pending.len(),
            "starting isolated TensorRT cache compiler"
        );
        let status = std::process::Command::new(&executable)
            .arg("--config")
            .arg(config_path)
            .arg("--compile-model")
            .arg(model_name)
            .status()?;
        if !status.success() {
            failed.push(model_name.clone());
            warn!(
                model = model_name,
                status = %status,
                "TensorRT cache compilation failed; runtime loading will use the configured fallback"
            );
        }
    }
    Ok(failed)
}

fn providers_without_tensorrt(providers: &[String]) -> Vec<String> {
    use jetort::engine::provider_chain::{chain_for_target, compiled_runtime_target, Backend};
    let auto = providers.is_empty()
        || providers
            .iter()
            .any(|provider| provider.eq_ignore_ascii_case("auto"));
    if auto {
        return chain_for_target(compiled_runtime_target())
            .iter()
            .copied()
            .filter(|backend| *backend != Backend::TensorRt && backend.is_compiled_in())
            .map(|backend| backend.canonical_name().to_string())
            .collect();
    }
    let mut kept: Vec<String> = providers
        .iter()
        .filter(|provider| {
            !provider.eq_ignore_ascii_case("tensorrt") && !provider.eq_ignore_ascii_case("trt")
        })
        .cloned()
        .collect();
    if kept.is_empty() {
        kept.push(Backend::Cpu.canonical_name().to_string());
    }
    kept
}

fn trt_manifest_for(
    model: &jetort::config::ModelConfig,
) -> Option<jetort::engine::provider::TrtCacheManifest> {
    let (model_path, _) = jetort::engine::session::resolve_model_and_tokenizer_path(model).ok()?;
    let inspection = jetort::engine::session::inspect_or_fallback(&model_path, model);
    manifest_for_path(model, &model_path, &inspection)
}

fn manifest_for_path(
    model: &jetort::config::ModelConfig,
    model_path: &std::path::Path,
    inspection: &jetort::engine::onnx_introspect::OnnxModelInspection,
) -> Option<jetort::engine::provider::TrtCacheManifest> {
    let precision = match inspection.quantization_kind {
        jetort::engine::onnx_introspect::QuantizationKind::ExplicitQdqInt8 => "INT8",
        jetort::engine::onnx_introspect::QuantizationKind::Fp16 => "FP16",
        _ => "FP16",
    };
    jetort::engine::provider::current_trt_manifest(
        model_path,
        precision,
        &jetort::engine::provider::trt_profile_signature(model),
    )
}

fn init_tracing(config: &AppConfig) {
    let default_filter = format!(
        "jetort={},ort=warn,tower_http=warn",
        config.server.log_level
    );
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| default_filter.into());

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().compact())
        .init();

    if env::var("RUST_LOG").is_ok() {
        info!("log filter sourced from RUST_LOG");
    } else {
        info!(level = %config.server.log_level, "log filter sourced from configuration");
    }
}

fn init_onnxruntime(config: &AppConfig) -> Result<(), Box<dyn std::error::Error>> {
    let setup = |b: ort::environment::EnvironmentBuilder| {
        b.with_logger(std::sync::Arc::new(
            |level, _category, id, location, message| {
                // Suppress verbose TRT internal importer probes, tactic evaluations, and allocator noise
                let is_noisy_internal = location.contains("ModelImporter")
                    || location.contains("onnxOpCheckers")
                    || location.contains("checkFallback")
                    || location.contains("bfc_arena")
                    || location.contains("BFCArena")
                    || location.contains("defaultAllocator")
                    || location.contains("optimizer.cpp")
                    || message.contains("ModelImporter")
                    || message.contains("onnxOpCheckers")
                    || message.contains("checkFallback")
                    || message.contains("Plugin not found")
                    || message.contains("creator &&")
                    || message.contains("Tactic Device request")
                    || message.contains("Skipping tactic")
                    || message.contains("UNSUPPORTED_STATE")
                    || message.contains("End node")
                    || message.contains("--- Node of")
                    || message.contains("In node:")
                    || message.contains("BFCArena::AllocateRawInternal")
                    || message.contains("Failed to allocate memory for requested buffer")
                    || message.contains("Failed to parse ONNX model")
                    || message.contains("Builder failed to build")
                    || message.contains("defaultAllocator")
                    || message.contains("Requested amount of GPU memory")
                    || message.contains("could not be allocated")
                    || message.contains("IBuilder::buildSerializedNetwork")
                    || message.contains("Could not find any implementation for node")
                    || message.contains("Could not find a CPU kernel")
                    || message.contains("insufficient workspace")
                    || message.contains("computeCosts")
                    || message.contains("ForeignNode");

                if is_noisy_internal {
                    tracing::debug!(target: "ort::internal", id = %id, location = %location, "{message}");
                    return;
                }

                match level {
                    ort::logging::LogLevel::Verbose => tracing::trace!(target: "ort", id = %id, "{message}"),
                    ort::logging::LogLevel::Info => tracing::info!(target: "ort", id = %id, "{message}"),
                    ort::logging::LogLevel::Warning => tracing::warn!(target: "ort", id = %id, "{message}"),
                    ort::logging::LogLevel::Error | ort::logging::LogLevel::Fatal => {
                        tracing::error!(target: "ort", id = %id, "{message}");
                    }
                }
            },
        ))
    };

    // 1. Explicit path in config.yaml
    if let Some(ref path) = config.server.ort_dylib_path {
        info!("Loading ONNX Runtime from config: {}", path);
        setup(ort::init_from(path)?).commit();
        return Ok(());
    }

    // 2. Standard environment variable ORT_DYLIB_PATH
    if let Ok(path) = env::var("ORT_DYLIB_PATH") {
        info!("Loading ONNX Runtime from ORT_DYLIB_PATH: {}", path);
        setup(ort::init_from(&path)?).commit();
        return Ok(());
    }

    // 3. System installed library path (/usr/local/lib)
    if std::path::Path::new("/usr/local/lib/libonnxruntime.so").exists() {
        if let Ok(builder) = ort::init_from("/usr/local/lib/libonnxruntime.so") {
            info!("Loading ONNX Runtime from /usr/local/lib/libonnxruntime.so");
            setup(builder).commit();
            return Ok(());
        }
    }

    // 4. Local directory check (e.g. adjacent to binary or in ./lib)
    let local_paths = [
        "libonnxruntime.so",
        "lib/libonnxruntime.so",
        "target/release/libonnxruntime.so",
    ];
    for p in local_paths {
        if std::path::Path::new(p).exists() {
            if let Ok(builder) = ort::init_from(p) {
                info!("Loading ONNX Runtime from local path: {}", p);
                setup(builder).commit();
                return Ok(());
            }
        }
    }

    // 5. Default: load from system dynamic library paths (/usr/local/lib, /usr/lib, ldconfig cache)
    setup(ort::init()).commit();
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C signal handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C interrupt signal");
        },
        _ = terminate => {
            info!("Received SIGTERM terminate signal");
        },
    }
}
