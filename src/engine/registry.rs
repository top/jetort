use crate::config::{AppConfig, ModelConfig};
use crate::engine::session::{resolve_model_and_tokenizer_path, ModelSession};
use crate::error::AppError;
use serde::Serialize;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex as AsyncMutex, RwLock};
use tracing::{error, info, warn};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    Discovered,
    Loading,
    Loaded,
    Unloading,
    Unloaded,
    Failed,
}

impl std::fmt::Display for ModelState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ModelState::Discovered => "discovered",
            ModelState::Loading => "loading",
            ModelState::Loaded => "loaded",
            ModelState::Unloading => "unloading",
            ModelState::Unloaded => "unloaded",
            ModelState::Failed => "failed",
        })
    }
}

struct ModelRuntime {
    state: ModelState,
    session: Option<Arc<ModelSession>>,
    load_error: Option<String>,
}

struct ModelEntry {
    config: ModelConfig,
    runtime: Mutex<ModelRuntime>,
    active_requests: AtomicUsize,
    last_access: AtomicU64,
    load_generation: AtomicU64,
    estimated_memory_mb: usize,
}

impl ModelEntry {
    fn new(config: ModelConfig) -> Self {
        // ponytail: model_path may be just the ONNX graph (a few hundred KB) with the
        // actual weights split into a sibling `<file>.data` blob (ONNX external-data
        // format, used once a model exceeds the 2GB protobuf limit, e.g. ORT-GenAI's
        // builder output). Summing the whole bundle directory instead of the single
        // file avoids drastically undercounting memory needs for such models.
        let estimated_memory_mb = resolve_model_and_tokenizer_path(&config)
            .ok()
            .and_then(|(path, _)| path.parent().map(|dir| dir.to_path_buf()))
            .map(|dir| {
                let total_bytes: u64 = std::fs::read_dir(&dir)
                    .into_iter()
                    .flatten()
                    .flatten()
                    .filter_map(|entry| entry.metadata().ok())
                    .filter(|meta| meta.is_file())
                    .map(|meta| meta.len())
                    .sum();
                let file_mb = total_bytes.div_ceil(1024 * 1024) as usize;
                file_mb.saturating_mul(3) / 2 + 256
            })
            .unwrap_or(512);
        Self {
            config,
            runtime: Mutex::new(ModelRuntime {
                state: ModelState::Discovered,
                session: None,
                load_error: None,
            }),
            active_requests: AtomicUsize::new(0),
            last_access: AtomicU64::new(now_millis()),
            load_generation: AtomicU64::new(0),
            estimated_memory_mb,
        }
    }
}

pub struct ModelLease {
    session: Arc<ModelSession>,
    entry: Arc<ModelEntry>,
}

impl Deref for ModelLease {
    type Target = ModelSession;

    fn deref(&self) -> &Self::Target {
        &self.session
    }
}

impl Drop for ModelLease {
    fn drop(&mut self) {
        let previous = self.entry.active_requests.fetch_sub(1, Ordering::AcqRel);
        if previous == 1 {
            let mut runtime = self.entry.runtime.lock().expect("model runtime poisoned");
            if runtime.state == ModelState::Unloading {
                runtime.session.take();
                runtime.state = ModelState::Unloaded;
                drop(runtime);
                crate::engine::memory::trim_allocator_heap();
                info!(model = %self.entry.config.name, "deferred model unload completed");
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelStatus {
    pub name: String,
    pub alias: Option<String>,
    pub state: ModelState,
    pub active_requests: usize,
    pub load_on_startup: bool,
    pub estimated_memory_mb: usize,
    pub provider: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone)]
pub struct ModelRegistry {
    entries: Arc<RwLock<HashMap<String, Arc<ModelEntry>>>>,
    load_lock: Arc<AsyncMutex<()>>,
    initialized: Arc<AtomicBool>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            load_lock: Arc::new(AsyncMutex::new(())),
            initialized: Arc::new(AtomicBool::new(false)),
        }
    }

    pub async fn init_from_config(&self, config: &AppConfig) {
        let started = Instant::now();
        info!(
            models_total = config.models.len(),
            "model registry discovery started"
        );

        {
            let mut entries = self.entries.write().await;
            for model_config in &config.models {
                entries.insert(
                    model_config.name.clone(),
                    Arc::new(ModelEntry::new(model_config.clone())),
                );
            }
        }

        let mut loaded = 0usize;
        let mut failed = 0usize;
        for (index, model_config) in config.models.iter().enumerate() {
            if !model_config.load_on_startup {
                let entry = self
                    .find_entry(&model_config.name)
                    .await
                    .expect("registered");
                entry.runtime.lock().expect("model runtime poisoned").state = ModelState::Unloaded;
                info!(
                    model = %model_config.name,
                    progress = index + 1,
                    models_total = config.models.len(),
                    "model startup loading disabled"
                );
                continue;
            }

            match self.load_entry(model_config.name.clone()).await {
                Ok(()) => loaded += 1,
                Err(error) => {
                    failed += 1;
                    error!(
                        model = %model_config.name,
                        error = %error,
                        "model startup load failed; model remains unavailable"
                    );
                }
            }
        }

        info!(
            loaded,
            failed,
            models_total = config.models.len(),
            initialization_ms = started.elapsed().as_secs_f64() * 1000.0,
            "model registry initialization completed"
        );
        self.initialized.store(true, Ordering::Release);
    }

    async fn find_entry(&self, name: &str) -> Option<Arc<ModelEntry>> {
        let entries = self.entries.read().await;
        if let Some(entry) = entries.get(name) {
            return Some(entry.clone());
        }
        if let Some(entry) = entries
            .values()
            .find(|entry| {
                entry.config.name.eq_ignore_ascii_case(name)
                    || entry
                        .config
                        .alias
                        .as_ref()
                        .is_some_and(|alias| alias.eq_ignore_ascii_case(name))
            })
            .cloned()
        {
            return Some(entry);
        }
        if (name.is_empty() || name == "default" || name == "auto") && entries.len() == 1 {
            return entries.values().next().cloned();
        }
        if name.eq_ignore_ascii_case("whisper-1") || name.eq_ignore_ascii_case("whisper") {
            if let Some(entry) = entries.values().find(|entry| {
                entry
                    .runtime
                    .lock()
                    .expect("model runtime poisoned")
                    .session
                    .as_ref()
                    .is_some_and(|session| {
                        session.inputs_metadata().iter().any(|input| {
                            input.name == "features"
                                || input.name == "speech"
                                || (input.shape.len() == 3
                                    && input.shape.last().copied() == Some(80))
                        })
                    })
            }) {
                return Some(entry.clone());
            }
        }
        if name.starts_with("text-embedding-") {
            if let Some(entry) = entries.values().find(|entry| {
                entry
                    .runtime
                    .lock()
                    .expect("model runtime poisoned")
                    .session
                    .as_ref()
                    .is_some_and(|session| {
                        session.outputs_metadata().iter().any(|output| {
                            output.name == "sentence_embedding"
                                || output.name == "last_hidden_state"
                                || output.shape.len() == 2
                        })
                    })
            }) {
                return Some(entry.clone());
            }
        }
        if name.eq_ignore_ascii_case("jev-latest")
            || name.eq_ignore_ascii_case("jev-preview")
            || name.to_ascii_lowercase().starts_with("jev-")
        {
            let mut matching = entries.values().filter(|entry| {
                entry
                    .runtime
                    .lock()
                    .expect("model runtime poisoned")
                    .session
                    .as_ref()
                    .is_some_and(|session| session.supports_system_one())
            });
            if let Some(entry) = matching.next() {
                if matching.next().is_none() {
                    return Some(entry.clone());
                }
            }
        }
        None
    }

    pub async fn get_model(&self, name: &str) -> Result<Arc<ModelSession>, AppError> {
        let entry = self
            .find_entry(name)
            .await
            .ok_or_else(|| AppError::ModelNotFound(name.to_string()))?;
        let runtime = entry.runtime.lock().expect("model runtime poisoned");
        runtime
            .session
            .clone()
            .ok_or_else(|| AppError::ModelNotLoaded {
                name: entry.config.name.clone(),
                state: runtime.state.to_string(),
            })
    }

    pub async fn acquire_model(&self, name: &str) -> Result<ModelLease, AppError> {
        let entry = self
            .find_entry(name)
            .await
            .ok_or_else(|| AppError::ModelNotFound(name.to_string()))?;

        let (session, unavailable_state) = {
            let runtime = entry.runtime.lock().expect("model runtime poisoned");
            (
                if runtime.state == ModelState::Loaded {
                    runtime.session.clone()
                } else {
                    None
                },
                runtime.state,
            )
        };
        if let Some(session) = session {
            entry.active_requests.fetch_add(1, Ordering::AcqRel);
            entry.last_access.store(now_millis(), Ordering::Relaxed);
            let model_name = entry.config.name.clone();
            let lease = ModelLease { session, entry };
            self.evict_idle_under_pressure(Some(&model_name)).await;
            return Ok(lease);
        }

        // Lazy load-on-demand: rather than returning a "not loaded" error and loading in the
        // background (which forced callers to retry), await the load inline and serve this very
        // request. load_entry() serializes on load_lock, so a load already in flight (state
        // Loading, e.g. from startup preload) is awaited rather than duplicated.
        if matches!(
            unavailable_state,
            ModelState::Discovered
                | ModelState::Unloaded
                | ModelState::Failed
                | ModelState::Loading
        ) {
            info!(
                model = %entry.config.name,
                state = %unavailable_state,
                "model not resident; loading on demand before serving request"
            );
            self.load_entry_inner(entry.config.name.clone(), true).await?;

            let session = {
                let runtime = entry.runtime.lock().expect("model runtime poisoned");
                if runtime.state == ModelState::Loaded {
                    runtime.session.clone()
                } else {
                    None
                }
            };
            if let Some(session) = session {
                entry.active_requests.fetch_add(1, Ordering::AcqRel);
                entry.last_access.store(now_millis(), Ordering::Relaxed);
                let model_name = entry.config.name.clone();
                let lease = ModelLease { session, entry };
                self.evict_idle_under_pressure(Some(&model_name)).await;
                return Ok(lease);
            }
        }

        let runtime_state = entry
            .runtime
            .lock()
            .expect("model runtime poisoned")
            .state
            .to_string();
        Err(AppError::ModelNotLoaded {
            name: entry.config.name.clone(),
            state: runtime_state,
        })
    }

    fn request_load(&self, entry: Arc<ModelEntry>) {
        {
            let mut runtime = entry.runtime.lock().expect("model runtime poisoned");
            if matches!(runtime.state, ModelState::Loading | ModelState::Loaded) {
                return;
            }
            runtime.state = ModelState::Loading;
            runtime.load_error = None;
        }
        let registry = self.clone();
        tokio::spawn(async move {
            if let Err(error) = registry.load_entry(entry.config.name.clone()).await {
                warn!(model = %entry.config.name, error = %error, "background model load failed");
            }
        });
    }

    async fn load_entry(&self, name: String) -> Result<(), AppError> {
        self.load_entry_inner(name, false).await
    }

    /// `force` bypasses the "previously evicted, don't resurrect" guard. Set it only for
    /// explicit on-demand loads driven by an inbound request, where the caller genuinely
    /// wants an evicted model brought back to serve that request.
    async fn load_entry_inner(&self, name: String, force: bool) -> Result<(), AppError> {
        let entry = self
            .find_entry(&name)
            .await
            .ok_or_else(|| AppError::ModelNotFound(name.clone()))?;
        let _load_guard = self.load_lock.lock().await;
        {
            let mut runtime = entry.runtime.lock().expect("model runtime poisoned");
            if runtime.state == ModelState::Loaded {
                return Ok(());
            }
            if !force
                && runtime.state == ModelState::Unloaded
                && entry.load_generation.load(Ordering::Acquire) > 0
            {
                return Ok(());
            }
            runtime.state = ModelState::Loading;
            runtime.load_error = None;
        }
        let generation = entry.load_generation.fetch_add(1, Ordering::AcqRel) + 1;

        if let Err(error) = self.ensure_memory_for(&entry).await {
            let mut runtime = entry.runtime.lock().expect("model runtime poisoned");
            runtime.state = ModelState::Unloaded;
            runtime.load_error = Some(error.to_string());
            return Err(error);
        }
        let config = entry.config.clone();
        let started = Instant::now();
        let result = match tokio::task::spawn_blocking(move || ModelSession::load(&config)).await {
            Ok(result) => result,
            Err(error) => Err(AppError::Internal(format!(
                "Model load task failed: {error}"
            ))),
        };

        if entry.load_generation.load(Ordering::Acquire) != generation {
            info!(model = %entry.config.name, "discarding cancelled model load");
            return Ok(());
        }

        match result {
            Ok(session) => {
                let provider = session.active_provider().to_string();
                let mut runtime = entry.runtime.lock().expect("model runtime poisoned");
                runtime.session = Some(Arc::new(session));
                runtime.state = ModelState::Loaded;
                runtime.load_error = None;
                entry.last_access.store(now_millis(), Ordering::Relaxed);
                info!(
                    model = %entry.config.name,
                    provider,
                    load_ms = started.elapsed().as_secs_f64() * 1000.0,
                    "model residency established"
                );
                Ok(())
            }
            Err(error) => {
                let mut runtime = entry.runtime.lock().expect("model runtime poisoned");
                runtime.state = ModelState::Failed;
                runtime.load_error = Some(error.to_string());
                Err(error)
            }
        }
    }

    async fn ensure_memory_for(&self, target: &Arc<ModelEntry>) -> Result<(), AppError> {
        let minimum = automatic_memory_reserve_mb();
        let required = minimum.saturating_add(target.estimated_memory_mb);
        self.evict_until_available(required, Some(&target.config.name))
            .await;
        let available = crate::engine::memory::get_memory_snapshot().system_available_mb;
        if available > 0.0 && available < required as f64 {
            return Err(AppError::ModelNotLoaded {
                name: target.config.name.clone(),
                state: "insufficient_memory".to_string(),
            });
        }
        Ok(())
    }

    async fn evict_idle_under_pressure(&self, protected: Option<&str>) {
        let minimum = automatic_memory_reserve_mb();
        let available = crate::engine::memory::get_memory_snapshot().system_available_mb;
        if available > 0.0 && available < minimum as f64 {
            self.evict_until_available(minimum, protected).await;
        }
    }

    async fn evict_until_available(&self, required_mb: usize, protected: Option<&str>) {
        loop {
            let available = crate::engine::memory::get_memory_snapshot().system_available_mb;
            if available <= 0.0 || available >= required_mb as f64 {
                return;
            }

            let entries: Vec<Arc<ModelEntry>> =
                self.entries.read().await.values().cloned().collect();
            let victim = entries
                .into_iter()
                .filter(|entry| {
                    protected != Some(entry.config.name.as_str())
                        && entry.active_requests.load(Ordering::Acquire) == 0
                        && entry.runtime.lock().expect("model runtime poisoned").state
                            == ModelState::Loaded
                })
                .min_by_key(|entry| entry.last_access.load(Ordering::Relaxed));

            let Some(victim) = victim else {
                return;
            };
            let mut runtime = victim.runtime.lock().expect("model runtime poisoned");
            runtime.session.take();
            runtime.state = ModelState::Unloaded;
            drop(runtime);
            crate::engine::memory::trim_allocator_heap();
            info!(
                model = %victim.config.name,
                required_mb,
                available_mb = available,
                "evicted idle model under memory pressure"
            );
        }
    }

    pub async fn load_model(&self, config: &ModelConfig) -> Result<(), AppError> {
        let entry = Arc::new(ModelEntry::new(config.clone()));
        self.entries
            .write()
            .await
            .insert(config.name.clone(), entry.clone());
        self.request_load(entry);
        Ok(())
    }

    pub async fn unload_model(&self, name: &str) -> Result<(), AppError> {
        let entry = self
            .find_entry(name)
            .await
            .ok_or_else(|| AppError::ModelNotFound(name.to_string()))?;
        let active = entry.active_requests.load(Ordering::Acquire);
        entry.load_generation.fetch_add(1, Ordering::AcqRel);
        let mut runtime = entry.runtime.lock().expect("model runtime poisoned");
        if active > 0 {
            runtime.state = ModelState::Unloading;
            info!(
                model = %entry.config.name,
                active_requests = active,
                "model unload deferred until active requests complete"
            );
        } else {
            runtime.session.take();
            runtime.state = ModelState::Unloaded;
            drop(runtime);
            crate::engine::memory::trim_allocator_heap();
            info!(model = %entry.config.name, "model unloaded");
        }
        Ok(())
    }

    pub async fn list_models(&self) -> Vec<String> {
        self.entries.read().await.keys().cloned().collect()
    }

    pub async fn list_model_sessions(&self) -> Vec<Arc<ModelSession>> {
        self.entries
            .read()
            .await
            .values()
            .filter_map(|entry| {
                entry
                    .runtime
                    .lock()
                    .expect("model runtime poisoned")
                    .session
                    .clone()
            })
            .collect()
    }

    pub async fn model_statuses(&self) -> Vec<ModelStatus> {
        self.entries
            .read()
            .await
            .values()
            .map(|entry| {
                let runtime = entry.runtime.lock().expect("model runtime poisoned");
                ModelStatus {
                    name: entry.config.name.clone(),
                    alias: entry.config.alias.clone(),
                    state: runtime.state,
                    active_requests: entry.active_requests.load(Ordering::Acquire),
                    load_on_startup: entry.config.load_on_startup,
                    estimated_memory_mb: entry.estimated_memory_mb,
                    provider: runtime
                        .session
                        .as_ref()
                        .map(|session| session.active_provider().to_string()),
                    error: runtime.load_error.clone(),
                }
            })
            .collect()
    }

    pub async fn is_ready(&self) -> bool {
        self.initialized.load(Ordering::Acquire)
    }

    pub fn start_memory_monitor(&self) {
        let registry = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(2));
            loop {
                interval.tick().await;
                registry.evict_idle_under_pressure(None).await;
            }
        });
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn automatic_memory_reserve_mb() -> usize {
    let total = crate::engine::memory::get_memory_snapshot().system_total_mb;
    memory_reserve_mb(total)
}

fn memory_reserve_mb(total_mb: f64) -> usize {
    (total_mb > 0.0)
        .then(|| (total_mb as usize / 8).clamp(1024, 4096))
        .unwrap_or(2048)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_reserve_is_automatic_and_bounded() {
        assert_eq!(memory_reserve_mb(0.0), 2048);
        assert_eq!(memory_reserve_mb(4096.0), 1024);
        assert_eq!(memory_reserve_mb(16384.0), 2048);
        assert_eq!(memory_reserve_mb(65536.0), 4096);
    }

    #[tokio::test]
    async fn configured_non_preloaded_model_is_discoverable_but_unavailable() {
        let config: AppConfig = serde_yaml::from_str(
            r#"
models:
  - name: lazy-model
    path: /nonexistent/model.onnx
    load_on_startup: false
"#,
        )
        .unwrap();
        let registry = ModelRegistry::new();
        registry.init_from_config(&config).await;

        let statuses = registry.model_statuses().await;
        assert_eq!(statuses.len(), 1);
        assert_eq!(statuses[0].state, ModelState::Unloaded);

        let error = match registry.acquire_model("lazy-model").await {
            Ok(_) => panic!("unloaded model unexpectedly acquired"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            AppError::ModelNotLoaded {
                state,
                ..
            } if state == "loading"
        ));
    }
}
