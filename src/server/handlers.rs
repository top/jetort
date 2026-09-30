use crate::config::ModelConfig;
use crate::engine::registry::ModelRegistry;
use crate::engine::timings::{self, OneShotTimings, TranscriptionTimings};
use crate::error::AppError;
use crate::protocol::kserve_v2::{ModelMetadataResponse, ServerLiveResponse, ServerReadyResponse};
use crate::protocol::openai_v1::{
    EmbeddingObject, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, ModelCard,
    ModelListResponse, RerankRequest, RerankResponse, RerankResult,
};
use axum::{
    body::Body,
    extract::{FromRequest, Path, Request, State},
    http::{HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use std::time::Instant;

// Health Checks
pub async fn live_check() -> Json<ServerLiveResponse> {
    Json(ServerLiveResponse { live: true })
}

pub async fn memory_check() -> Json<crate::engine::memory::MemorySnapshot> {
    Json(crate::engine::memory::get_memory_snapshot())
}

pub async fn ready_check(
    State(registry): State<ModelRegistry>,
) -> (StatusCode, Json<ServerReadyResponse>) {
    let ready = registry.is_ready().await;
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (status, Json(ServerReadyResponse { ready }))
}

// OpenAI-compatible Models API (/v1/models)
pub async fn list_openai_models(State(registry): State<ModelRegistry>) -> Json<ModelListResponse> {
    let mut sessions = registry.list_model_sessions().await;
    sessions.sort_by(|left, right| left.name().cmp(right.name()));
    let mut statuses = registry.model_statuses().await;
    statuses.sort_by(|left, right| left.name.cmp(&right.name));
    let data = statuses
        .iter()
        .map(|status| {
            sessions
                .iter()
                .find(|session| session.name() == status.name)
                .map(|session| {
                    model_card(session, &status.state.to_string(), status.active_requests)
                })
                .unwrap_or_else(|| ModelCard {
                    id: status.name.clone(),
                    object: "model".to_string(),
                    created: 0,
                    owned_by: "jetort".to_string(),
                    shutdown_date: None,
                    meta: Some(crate::protocol::openai_v1::ModelMeta {
                        engine: status.engine.clone(),
                        provider: status.provider.clone(),
                        state: status.state.to_string(),
                        active_requests: status.active_requests,
                        context_length: None,
                        size: status.estimated_memory_mb as u64 * 1024 * 1024,
                        inputs: Vec::new(),
                        outputs: Vec::new(),
                        endpoints: Vec::new(),
                        precision: status.precision.clone(),
                    }),
                })
        })
        .collect();
    let models = sessions
        .into_iter()
        .filter(|session| session.supports_system_one())
        .flat_map(|session| {
            let release_date = unix_date(session.created_at());
            let names: Vec<String> = std::iter::once(session.name().to_string())
                .chain(session.aliases().first().cloned())
                .collect();
            names
                .into_iter()
                .map(move |name| crate::protocol::system_one::TypeSafeModelCard {
                    name,
                    description: "Local System One compatible decision model".to_string(),
                    release_date: release_date.clone(),
                })
        })
        .collect();

    Json(ModelListResponse {
        object: "list".to_string(),
        data,
        models,
    })
}

pub async fn get_openai_model(
    Path(model_name): Path<String>,
    State(registry): State<ModelRegistry>,
) -> Result<Json<ModelCard>, AppError> {
    let session = registry.acquire_model(&model_name).await?;
    Ok(Json(model_card(&session, "loaded", 0)))
}

fn model_card(
    session: &crate::engine::session::ModelSession,
    state: &str,
    active_requests: usize,
) -> ModelCard {
    ModelCard {
        id: session.name().to_string(),
        object: "model".to_string(),
        created: session.created_at(),
        owned_by: "jetort".to_string(),
        shutdown_date: None,
        meta: Some(crate::protocol::openai_v1::ModelMeta {
            engine: session.engine_name().to_string(),
            provider: Some(session.active_provider().to_string()),
            state: state.to_string(),
            active_requests,
            context_length: session.n_ctx(),
            size: session.model_size(),
            inputs: session
                .inputs_metadata()
                .iter()
                .map(|input| input.name.clone())
                .collect(),
            outputs: session
                .outputs_metadata()
                .iter()
                .map(|output| output.name.clone())
                .collect(),
            endpoints: session.supported_endpoints(),
            precision: session.precision().map(|value| value.to_string()),
        }),
    }
}

fn unix_date(timestamp: u64) -> String {
    let days = (timestamp / 86_400) as i64;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_part = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_part + 2) / 5 + 1;
    let month = month_part + if month_part < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

pub async fn system_one(State(registry): State<ModelRegistry>, request: Request) -> Response {
    let started = Instant::now();
    let request_id = format!(
        "systemone-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    let result = async {
        let body = axum::body::to_bytes(request.into_body(), 16 * 1024 * 1024)
            .await
            .map_err(|error| {
                AppError::Unprocessable(format!("failed to read request body: {error}"))
            })?;
        let request: crate::protocol::system_one::SystemOneRequest = serde_json::from_slice(&body)
            .map_err(|error| AppError::Unprocessable(format!("invalid request body: {error}")))?;
        crate::protocol::system_one::validate_request_shape(&request)?;
        let question_count = request.questions.len();
        let session = registry.acquire_model(&request.model).await?;
        let queue_started = Instant::now();
        let _permit = session.acquire_permit().await?;
        let queue_ms = queue_started.elapsed().as_secs_f64() * 1000.0;
        let model = session.name().to_string();
        let backend = session.active_provider().to_string();
        let (answers, input_tokens) = if session.uses_openjev_contract() {
            let task = tokio::task::spawn_blocking(move || {
                session.system_one_openjev(&request.state, &request.questions)
            });
            task.await.map_err(|error| {
                AppError::Internal(format!("System One task join error: {error}"))
            })??
        } else {
            let questions = crate::protocol::system_one::prepare_request(&request)?;
            let task = tokio::task::spawn_blocking(move || session.system_one(questions));
            task.await.map_err(|error| {
                AppError::Internal(format!("System One task join error: {error}"))
            })??
        };
        timings::log_one_shot(&OneShotTimings {
            task: "system_one".into(),
            model: model.clone(),
            backend,
            items: question_count,
            input_tokens,
            eval_ms: started.elapsed().as_secs_f64() * 1000.0,
            queue_ms,
        });
        let response = crate::protocol::system_one::SystemOneResponse {
            model,
            answers,
            usage: crate::protocol::system_one::SystemOneUsage {
                input_tokens,
                output_tokens: 0,
            },
        };
        serde_json::to_vec(&response).map_err(|error| {
            AppError::Internal(format!("failed to serialize System One response: {error}"))
        })
    }
    .await;

    let mut response = match result {
        Ok(body) => Response::builder()
            .status(StatusCode::OK)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from(body))
            .unwrap(),
        Err(error) => error.into_response(),
    };
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response
            .headers_mut()
            .insert("x-typesafe-request-id", value);
    }
    response
}

// Model Metadata (KServe v2)
pub async fn model_metadata(
    Path(model_name): Path<String>,
    State(registry): State<ModelRegistry>,
) -> Result<Json<ModelMetadataResponse>, AppError> {
    let session = registry.acquire_model(&model_name).await?;
    Ok(Json(ModelMetadataResponse {
        name: session.name().to_string(),
        platform: session.active_provider().to_string(),
        active_provider: session.active_provider().to_string(),
        inputs: session.inputs_metadata().to_vec(),
        outputs: session.outputs_metadata().to_vec(),
    }))
}

// KServe v2 infer is a placeholder until a model can actually serve raw tensors.
pub async fn model_infer(
    Path(_model_name): Path<String>,
    State(_registry): State<ModelRegistry>,
) -> Result<Json<serde_json::Value>, AppError> {
    Err(AppError::NotImplemented(
        "KServe v2 infer is not implemented".to_string(),
    ))
}

// OpenAI-compatible Embeddings
pub async fn openai_embeddings(
    State(registry): State<ModelRegistry>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, AppError> {
    let started = Instant::now();
    let session = registry.acquire_model(&request.model).await?;
    let queue_started = Instant::now();
    let _permit = session.acquire_permit().await?;
    let queue_ms = queue_started.elapsed().as_secs_f64() * 1000.0;
    let texts = request.input.into_vec();
    let text_count = texts.len();

    let task = tokio::task::spawn_blocking(move || session.embed_texts(texts));

    let (raw_embeddings, prompt_tokens, backend) = task
        .await
        .map_err(|e| AppError::Internal(format!("Embedding task join error: {}", e)))??;
    timings::log_one_shot(&OneShotTimings {
        task: "embedding".into(),
        model: request.model.clone(),
        backend,
        items: text_count,
        input_tokens: prompt_tokens,
        eval_ms: started.elapsed().as_secs_f64() * 1000.0,
        queue_ms,
    });

    let target_dim = request.dimensions;
    let data = raw_embeddings
        .into_iter()
        .enumerate()
        .map(|(index, mut emb)| {
            if let Some(d) = target_dim {
                if d < emb.len() {
                    emb.truncate(d);
                    let norm = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for x in &mut emb {
                            *x /= norm;
                        }
                    }
                }
            }
            EmbeddingObject {
                object: "embedding".to_string(),
                embedding: emb,
                index,
            }
        })
        .collect();

    Ok(Json(EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: request.model,
        usage: EmbeddingUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    }))
}

// Rerank Endpoint
pub async fn rerank(
    State(registry): State<ModelRegistry>,
    Json(request): Json<RerankRequest>,
) -> Result<Json<RerankResponse>, AppError> {
    let started = Instant::now();
    let session = registry.acquire_model(&request.model).await?;
    let queue_started = Instant::now();
    let _permit = session.acquire_permit().await?;
    let queue_ms = queue_started.elapsed().as_secs_f64() * 1000.0;
    let query = request.query.clone();
    let docs = request.documents.clone();

    let docs_for_task = docs.clone();
    let task =
        tokio::task::spawn_blocking(move || session.rerank_documents(&query, &docs_for_task));

    let (scores, prompt_tokens, backend) = task
        .await
        .map_err(|e| AppError::Internal(format!("Rerank task join error: {}", e)))??;
    timings::log_one_shot(&OneShotTimings {
        task: "rerank".into(),
        model: request.model.clone(),
        backend,
        items: docs.len(),
        input_tokens: prompt_tokens,
        eval_ms: started.elapsed().as_secs_f64() * 1000.0,
        queue_ms,
    });

    let mut results: Vec<RerankResult> = scores
        .into_iter()
        .enumerate()
        .map(|(index, score)| RerankResult {
            index,
            relevance_score: score,
            document: docs.get(index).cloned(),
        })
        .collect();

    // Sort descending by score
    results.sort_by(|a, b| {
        b.relevance_score
            .partial_cmp(&a.relevance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(top_n) = request.top_n {
        results.truncate(top_n);
    }

    Ok(Json(RerankResponse {
        id: "rerank-0".to_string(),
        model: request.model,
        results,
        usage: crate::protocol::openai_v1::RerankUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
        meta: crate::protocol::openai_v1::RerankMeta {
            api_version: crate::protocol::openai_v1::RerankMetaApiVersion {
                version: "1".to_string(),
            },
            billed_units: serde_json::json!({ "search_units": 1 }),
            tokens: crate::protocol::openai_v1::RerankTokens {
                input_tokens: prompt_tokens,
                output_tokens: 0,
            },
        },
    }))
}

// OpenAI-compatible Audio Transcriptions (Non-autoregressive ASR: SenseVoice / Paraformer)
pub async fn audio_transcriptions(
    State(registry): State<ModelRegistry>,
    req: Request,
) -> Result<Response, AppError> {
    let content_type = req
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let (mut model, raw_bytes, pcm_data, language, response_format) =
        if content_type.contains("multipart/form-data") {
            let mut multipart = axum::extract::Multipart::from_request(req, &())
                .await
                .map_err(|e| AppError::Internal(format!("Multipart form parsing error: {}", e)))?;

            let mut model = String::new();
            let mut language = None;
            let mut response_format = "json".to_string();
            let mut file_bytes = Vec::new();

            while let Ok(Some(field)) = multipart.next_field().await {
                let name = field.name().unwrap_or("").to_string();
                if name == "model" {
                    if let Ok(text) = field.text().await {
                        model = text;
                    }
                } else if name == "language" {
                    if let Ok(text) = field.text().await {
                        language = Some(text);
                    }
                } else if name == "response_format" {
                    if let Ok(text) = field.text().await {
                        response_format = text;
                    }
                } else if name == "file" {
                    if let Ok(bytes) = field.bytes().await {
                        file_bytes = bytes.to_vec();
                    }
                }
            }

            (model, Some(file_bytes), None, language, response_format)
        } else {
            // Fallback application/json mode
            let body_bytes = axum::body::to_bytes(req.into_body(), 50 * 1024 * 1024)
                .await
                .map_err(|e| AppError::Internal(format!("Failed to read request body: {}", e)))?;
            let json_req: crate::protocol::openai_v1::TranscriptionJsonRequest =
                serde_json::from_slice(&body_bytes)
                    .map_err(|e| AppError::Internal(format!("Invalid JSON request: {}", e)))?;

            let resp_fmt = json_req
                .response_format
                .unwrap_or_else(|| "json".to_string());
            (
                json_req.model,
                None,
                json_req.audio,
                json_req.language,
                resp_fmt,
            )
        };

    if model.trim().is_empty() {
        model = "whisper-1".to_string();
    }

    let session = registry.acquire_model(&model).await?;
    let started = Instant::now();
    let queue_started = Instant::now();
    let _permit = session.acquire_permit().await?;
    let queue_ms = queue_started.elapsed().as_secs_f64() * 1000.0;

    let pcm = if let Some(bytes) = raw_bytes {
        crate::engine::audio::decode_audio_to_pcm(&bytes).map_err(AppError::Internal)?
    } else {
        pcm_data.unwrap_or_default()
    };

    let duration = pcm.len() as f64 / 16000.0;
    let backend = session.active_provider().to_string();
    let lang_clone = language.clone();
    let (text, token_ids, total_frames): (String, Vec<u32>, usize) = {
        let task = tokio::task::spawn_blocking(move || {
            session.transcribe_audio(&pcm, lang_clone.as_deref())
        });
        task.await
            .map_err(|e| AppError::Internal(format!("Transcription task join error: {}", e)))??
    };
    let elapsed = started.elapsed().as_secs_f64();
    timings::log_transcription(&TranscriptionTimings {
        task: "transcription".into(),
        model: model.clone(),
        backend,
        audio_seconds: duration,
        frames: total_frames,
        output_tokens: token_ids.len(),
        eval_ms: elapsed * 1000.0,
        queue_ms,
    });

    if response_format == "text" {
        Ok(Response::builder()
            .header(
                axum::http::header::CONTENT_TYPE,
                "text/plain; charset=utf-8",
            )
            .body(Body::from(text))
            .unwrap())
    } else if response_format == "verbose_json" {
        let completion_tokens = token_ids.len();
        let segment = crate::protocol::openai_v1::TranscriptionSegment {
            id: 0,
            seek: 0,
            start: 0.0,
            end: duration,
            text: text.clone(),
            tokens: token_ids,
            temperature: 0.0,
            avg_logprob: 0.0,
            compression_ratio: 1.0,
            no_speech_prob: 0.0,
        };
        let resp = crate::protocol::openai_v1::TranscriptionVerboseResponse {
            task: "transcribe".to_string(),
            language: language.unwrap_or_else(|| "zh".to_string()),
            duration,
            text,
            segments: vec![segment],
            usage: crate::protocol::openai_v1::TranscriptionUsage {
                r#type: "duration".to_string(),
                seconds: duration,
                prompt_tokens: total_frames,
                completion_tokens,
                total_tokens: total_frames + completion_tokens,
            },
        };
        Ok(Response::builder()
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&resp).unwrap()))
            .unwrap())
    } else {
        let resp = crate::protocol::openai_v1::TranscriptionResponse { text };
        Ok(Response::builder()
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&resp).unwrap()))
            .unwrap())
    }
}

// Model Repository Lifecycle Management
pub async fn model_load(
    Path(model_name): Path<String>,
    State(registry): State<ModelRegistry>,
    Json(config): Json<ModelConfig>,
) -> Result<StatusCode, AppError> {
    let mut actual_cfg = config;
    actual_cfg.name = model_name;
    registry.load_model(&actual_cfg).await?;
    Ok(StatusCode::ACCEPTED)
}

pub async fn model_unload(
    Path(model_name): Path<String>,
    State(registry): State<ModelRegistry>,
) -> Result<StatusCode, AppError> {
    registry.unload_model(&model_name).await?;
    Ok(StatusCode::OK)
}
