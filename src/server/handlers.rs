use crate::config::ModelConfig;
use crate::engine::registry::ModelRegistry;
use crate::error::AppError;
use crate::protocol::codec::{json_to_ort_value, maybe_pad_tensor, ort_value_to_output};
use crate::protocol::kserve_v2::{
    InferenceRequest, InferenceResponse, ModelMetadataResponse, ServerLiveResponse,
    ServerReadyResponse,
};
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
use std::collections::HashMap;
use std::time::Instant;
use tracing::info;

fn log_inference_completed(
    task: &str,
    model: &str,
    items: usize,
    input_tokens: usize,
    started: Instant,
    queue_ms: f64,
) {
    let elapsed = started.elapsed().as_secs_f64();
    info!(
        task,
        model,
        items,
        input_tokens,
        queue_ms,
        inference_ms = elapsed * 1000.0,
        items_per_second = if elapsed > 0.0 {
            items as f64 / elapsed
        } else {
            0.0
        },
        tokens_per_second = if elapsed > 0.0 {
            input_tokens as f64 / elapsed
        } else {
            0.0
        },
        "inference completed"
    );
}

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
                        engine: "onnxruntime".to_string(),
                        provider: status.provider.clone(),
                        state: status.state.to_string(),
                        active_requests: status.active_requests,
                        context_length: None,
                        size: status.estimated_memory_mb as u64 * 1024 * 1024,
                        inputs: Vec::new(),
                        outputs: Vec::new(),
                        endpoints: Vec::new(),
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
        let (answers, input_tokens) = if session.uses_openjev_contract() {
            let task = tokio::task::spawn_blocking(move || {
                session.system_one_openjev(&request.state, &request.questions)
            });
            task.await
                .map_err(|error| AppError::Internal(format!("System One task join error: {error}")))??
        } else {
            let questions = crate::protocol::system_one::prepare_request(&request)?;
            let task = tokio::task::spawn_blocking(move || session.system_one(questions));
            task.await
                .map_err(|error| AppError::Internal(format!("System One task join error: {error}")))??
        };
        log_inference_completed(
            "system_one",
            &model,
            question_count,
            input_tokens,
            started,
            queue_ms,
        );
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
        platform: "onnxruntime".to_string(),
        active_provider: session.active_provider().to_string(),
        inputs: session.inputs_metadata().to_vec(),
        outputs: session.outputs_metadata().to_vec(),
    }))
}

// KServe v2 Inference
pub async fn model_infer(
    Path(model_name): Path<String>,
    State(registry): State<ModelRegistry>,
    Json(request): Json<InferenceRequest>,
) -> Result<Json<InferenceResponse>, AppError> {
    let started = Instant::now();
    let session = registry.acquire_model(&model_name).await?;
    let queue_started = Instant::now();
    let _permit = session.acquire_permit().await?;
    let queue_ms = queue_started.elapsed().as_secs_f64() * 1000.0;
    let input_count = request.inputs.len();

    // Offload CPU/GPU blocking forward execution to worker thread pool
    let inference_task = tokio::task::spawn_blocking(move || {
        let mut input_map = HashMap::new();
        for input_tensor in &request.inputs {
            let tensor_to_use = if let Some(target) = session.pad_seq_len() {
                if input_tensor.shape.len() == 2 && input_tensor.shape[1] > target {
                    maybe_pad_tensor(input_tensor, Some(target))
                } else {
                    input_tensor.clone()
                }
            } else if input_tensor.shape.len() == 2 && input_tensor.shape[1] > 512 {
                // Safeguard against unbounded sequence lengths causing quadratic memory spikes on edge devices
                maybe_pad_tensor(input_tensor, Some(512))
            } else {
                input_tensor.clone()
            };
            let ort_val = json_to_ort_value(&tensor_to_use)?;
            input_map.insert(tensor_to_use.name, ort_val);
        }

        let raw_outputs = session.infer_raw(input_map)?;

        let mut output_tensors = Vec::new();
        for (out_name, out_val) in &raw_outputs {
            let kserve_output = ort_value_to_output(out_name, out_val)?;
            output_tensors.push(kserve_output);
        }

        Ok::<Vec<crate::protocol::kserve_v2::KServeOutputTensor>, AppError>(output_tensors)
    });

    let outputs = inference_task
        .await
        .map_err(|e| AppError::Internal(format!("Inference task join error: {}", e)))??;
    log_inference_completed("raw", &model_name, input_count, 0, started, queue_ms);

    Ok(Json(InferenceResponse {
        model_name,
        model_version: None,
        id: None,
        outputs,
    }))
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

    let (raw_embeddings, prompt_tokens) = task
        .await
        .map_err(|e| AppError::Internal(format!("Embedding task join error: {}", e)))??;
    log_inference_completed(
        "embedding",
        &request.model,
        text_count,
        prompt_tokens,
        started,
        queue_ms,
    );

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

    let (scores, prompt_tokens) = task
        .await
        .map_err(|e| AppError::Internal(format!("Rerank task join error: {}", e)))??;
    log_inference_completed(
        "rerank",
        &request.model,
        docs.len(),
        prompt_tokens,
        started,
        queue_ms,
    );

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
    let (text, token_ids, total_frames): (String, Vec<u32>, usize) =
        if session.as_hybrid().is_ok() {
            // Hybrid multimodal models (e.g. Confucius4-R2T2) have no standalone One-Shot ASR
            // path; drive them through the same audio-embedding + autoregressive-decode pipeline
            // `chat_completions` uses for `input_audio`, but with a bare audio-only prompt.
            transcribe_hybrid_audio(&session, pcm).await?
        } else {
            let lang_clone = language.clone();
            let task = tokio::task::spawn_blocking(move || {
                session.transcribe_audio(&pcm, lang_clone.as_deref())
            });
            task.await
                .map_err(|e| AppError::Internal(format!("Transcription task join error: {}", e)))??
        };
    let elapsed = started.elapsed().as_secs_f64();
    info!(
        task = "transcription",
        model,
        audio_seconds = duration,
        frames = total_frames,
        output_tokens = token_ids.len(),
        queue_ms,
        inference_ms = elapsed * 1000.0,
        realtime_factor = if duration > 0.0 {
            elapsed / duration
        } else {
            0.0
        },
        "inference completed"
    );

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

/// Drives a Hybrid multimodal model (e.g. Confucius4-R2T2) through a bare audio-only chat turn to
/// produce a plain transcription, for use by `/v1/audio/transcriptions` (which otherwise only
/// understands the legacy One-Shot ASR models). Mirrors `chat_completions`'s `input_audio`
/// handling, but with no user-supplied text and `enable_thinking` forced on, since this model
/// family only emits recognized text inside a `<think>`-scoped `<asr_text>...</asr_text>` block
/// when given audio with no other instruction — plain non-thinking prompts have been observed to
/// leave that text unrecognized/unemitted.
async fn transcribe_hybrid_audio(
    session: &crate::engine::session::ModelEngine,
    pcm: Vec<f32>,
) -> Result<(String, Vec<u32>, usize), AppError> {
    let genai = session.as_genai()?.clone();
    let extra_model_inputs = crate::engine::hybrid::HybridSession::build_audio_extra_inputs(&pcm);
    let messages = vec![crate::engine::genai::ChatMessage {
        role: "user".to_string(),
        content: crate::engine::hybrid::HybridSession::audio_placeholder_text(),
        reasoning_content: None,
        tool_calls: None,
    }];
    let prompt = genai.render_chat_template(&messages, true, &[], Some(true))?;

    let temperature = genai.default_temperature.unwrap_or(0.0);
    let top_p = genai.default_top_p.unwrap_or(0.95);
    let top_k = genai.default_top_k;
    let repetition_penalty = genai.default_repetition_penalty;
    let (prompt_tokens, _actual_max_tokens, mut rx) = genai.generate_stream(
        &prompt,
        2048,
        temperature,
        top_p,
        top_k,
        Vec::new(),
        repetition_penalty,
        extra_model_inputs,
    )?;

    let mut full_completion = String::new();
    let mut token_count = 0usize;
    while let Some(res) = rx.recv().await {
        match res {
            Ok(tok) => {
                full_completion.push_str(&tok);
                token_count += 1;
            }
            Err(e) => return Err(e),
        }
    }

    // `render_chat_template` primes the prompt with an open `<think>\n`; restore it before
    // parsing so the model's ASR-mode output is classified as `reasoning_content` rather than
    // leaking into `content` (see `chat_completions`'s equivalent non-streaming parse step).
    let parsed = crate::protocol::tool_call_parser::parse_full_message(&format!(
        "<think>\n{}",
        full_completion
    ));

    // Confucius4-R2T2's ASR-mode output wraps the recognized text in a dedicated `<asr_text>`
    // special token pair inside the reasoning block; extract it if present, otherwise fall back
    // to whatever plain content/reasoning text the model produced.
    fn extract_asr_text(s: &str) -> Option<String> {
        let start = s.find("<asr_text>")? + "<asr_text>".len();
        let rest = &s[start..];
        let end = rest.find("</asr_text>").unwrap_or(rest.len());
        Some(rest[..end].trim().to_string())
    }

    let text = extract_asr_text(&full_completion)
        .or_else(|| parsed.reasoning_content.as_deref().and_then(extract_asr_text))
        .or_else(|| parsed.content.clone().filter(|s| !s.trim().is_empty()))
        .or_else(|| {
            parsed
                .reasoning_content
                .clone()
                .filter(|s| !s.trim().is_empty())
        })
        .unwrap_or_default();

    Ok((text, Vec::new(), prompt_tokens.max(token_count)))
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

// OpenAI-compatible Chat Completions (/v1/chat/completions)
pub async fn chat_completions(
    State(registry): State<ModelRegistry>,
    Json(request): Json<crate::protocol::openai_v1::ChatCompletionRequest>,
) -> Result<Response, AppError> {
    let queue_started = Instant::now();
    let session = registry.acquire_model(&request.model).await?;
    let _permit = session.acquire_permit().await?;
    info!(
        task = "chat_completion",
        model = %request.model,
        queue_ms = queue_started.elapsed().as_secs_f64() * 1000.0,
        "inference slot acquired"
    );
    let genai = session.as_genai()?.clone();

    // Hybrid audio-LLM support: a message's `input_audio` content part (OpenAI-style
    // `{"type": "input_audio", "input_audio": {"data": <base64>, "format": ...}}`) carries a
    // base64-encoded audio clip. For Hybrid models this is decoded to PCM, turned into a mel
    // spectrogram, and injected as extra named model inputs alongside `<|audio_pad|>` placeholder
    // tokens in the prompt (see `engine::hybrid` for the fixed-shape-export invariants this
    // relies on). Only one audio clip per request is supported, matching the model's
    // single-clip fixed-shape export.
    let audio_message_idx = request
        .messages
        .iter()
        .position(|m| m.input_audio.is_some());
    let extra_model_inputs = if let Some(idx) = audio_message_idx {
        let hybrid = session.as_hybrid().map_err(|_| {
            AppError::InvalidInput(format!(
                "Model '{}' does not support audio input (input_audio content parts require a \
                 Hybrid multimodal model)",
                request.model
            ))
        })?;
        let input_audio = request.messages[idx].input_audio.as_ref().unwrap();
        let audio_bytes = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            &input_audio.data,
        )
        .map_err(|e| AppError::InvalidInput(format!("Invalid base64 in input_audio.data: {e}")))?;
        let pcm = crate::engine::audio::decode_audio_to_pcm(&audio_bytes)
            .map_err(AppError::InvalidInput)?;
        let _ = hybrid; // presence already validated `session.as_hybrid()` succeeded above
        crate::engine::hybrid::HybridSession::build_audio_extra_inputs(&pcm)
    } else {
        Vec::new()
    };

    let chat_messages: Vec<crate::engine::genai::ChatMessage> = request
        .messages
        .iter()
        .enumerate()
        .map(|(idx, m)| {
            // Tool calls attached to a replayed assistant message: the wire
            // format's `function.arguments` is a JSON *string* (OpenAI
            // convention), but the Jinja chat_template iterates arguments with
            // `.items()`, so it needs to already be a parsed JSON object.
            let tool_calls = m.tool_calls.as_ref().map(|calls| {
                calls
                    .iter()
                    .map(|tc| {
                        let arguments: serde_json::Value =
                            serde_json::from_str(&tc.function.arguments)
                                .unwrap_or_else(|_| serde_json::json!({}));
                        serde_json::json!({ "name": tc.function.name, "arguments": arguments })
                    })
                    .collect()
            });
            let mut content = m.content.clone().unwrap_or_default();
            if Some(idx) == audio_message_idx {
                content = format!(
                    "{}{}",
                    crate::engine::hybrid::HybridSession::audio_placeholder_text(),
                    content
                );
            }
            crate::engine::genai::ChatMessage {
                role: m.role.clone(),
                content,
                reasoning_content: m.reasoning_content.clone(),
                tool_calls,
            }
        })
        .collect();


    // `tool_choice: "none"` suppresses tool definitions from the rendered
    // prompt entirely (OpenAI semantics); otherwise forward the request's
    // tool schemas verbatim into the chat_template.
    let suppress_tools = matches!(
        request.tool_choice.as_ref().and_then(|v| v.as_str()),
        Some("none")
    );
    let tools: Vec<serde_json::Value> = if suppress_tools {
        Vec::new()
    } else {
        request.tools.clone().unwrap_or_default()
    };
    // Default to reasoning enabled; a client can opt out via the vLLM/SGLang-
    // style `chat_template_kwargs: {"enable_thinking": false}` extension.
    let enable_thinking = request
        .chat_template_kwargs
        .as_ref()
        .and_then(|v| v.get("enable_thinking"))
        .and_then(|v| v.as_bool())
        .or(Some(true));

    let prompt =
        genai.render_chat_template(&chat_messages, true, &tools, enable_thinking)?;
    let max_tokens = request.max_tokens.unwrap_or(2048);
    // Resolve sampling parameters: an explicit request value always wins; otherwise fall
    // back to this model's own config.yaml defaults (llama.cpp `models.ini`-style per-model
    // temp/top-p/top-k/repeat-penalty), and only then to the server-wide hardcoded default.
    let temperature = request
        .temperature
        .or(genai.default_temperature)
        .unwrap_or(0.7);
    let top_p = request.top_p.or(genai.default_top_p).unwrap_or(0.95);
    let top_k = request.top_k.or(genai.default_top_k);
    let model_name = request.model.clone();
    let stop_sequences = match request.stop {
        Some(crate::protocol::openai_v1::StringOrArray::Single(s)) => vec![s],
        Some(crate::protocol::openai_v1::StringOrArray::Multiple(v)) => v,
        None => Vec::new(),
    };
    let repetition_penalty = request
        .presence_penalty
        .or(request.frequency_penalty)
        .or(genai.default_repetition_penalty);
    let include_usage = request
        .stream_options
        .as_ref()
        .and_then(|o| o.include_usage)
        .unwrap_or(false);

    let (prompt_tokens, actual_max_tokens, mut rx) = genai.generate_stream(
        &prompt,
        max_tokens,
        temperature,
        top_p,
        top_k,
        stop_sequences,
        repetition_penalty,
        extra_model_inputs,
    )?;

    if request.stream.unwrap_or(false) {
        let stream = async_stream::stream! {
            let _model_lease = session;
            let _inference_permit = _permit;
            let request_id = format!(
                "chatcmpl-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis()
            );
            let created = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let system_fingerprint = "fp_jetort".to_string();

            // 1. Initial chunk establishing assistant role
            let initial_chunk = crate::protocol::openai_v1::ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_name.clone(),
                system_fingerprint: system_fingerprint.clone(),
                choices: vec![crate::protocol::openai_v1::ChatCompletionChunkChoice {
                    index: 0,
                    delta: crate::protocol::openai_v1::ChatCompletionChunkDelta {
                        role: Some("assistant".to_string()),
                        content: Some(String::new()),
                        reasoning_content: None,
                        tool_calls: None,
                    },
                    logprobs: None,
                    finish_reason: None,
                }],
                usage: None,
            };
            if let Ok(json_str) = serde_json::to_string(&initial_chunk) {
                yield Ok::<_, std::convert::Infallible>(format!("data: {}\n\n", json_str));
            }

            // 2. Token chunks: fed through the tool-call/reasoning parser so
            // `<think>...</think>` and `<function>...</function>` never leak
            // as raw/partial text to the client.
            let mut generated_count = 0;
            let mut saw_tool_call = false;
            let mut parser = crate::protocol::tool_call_parser::StreamParser::new();
            let mut emit_event = |ev: crate::protocol::tool_call_parser::StreamEvent| {
                use crate::protocol::tool_call_parser::StreamEvent;
                let delta = match ev {
                    StreamEvent::Content(text) => crate::protocol::openai_v1::ChatCompletionChunkDelta {
                        role: None,
                        content: Some(text),
                        reasoning_content: None,
                        tool_calls: None,
                    },
                    StreamEvent::Reasoning(text) => crate::protocol::openai_v1::ChatCompletionChunkDelta {
                        role: None,
                        content: None,
                        reasoning_content: Some(text),
                        tool_calls: None,
                    },
                    StreamEvent::ToolCall(tc) => {
                        saw_tool_call = true;
                        crate::protocol::openai_v1::ChatCompletionChunkDelta {
                            role: None,
                            content: None,
                            reasoning_content: None,
                            tool_calls: Some(vec![crate::protocol::openai_v1::ToolCallDelta {
                                index: 0,
                                id: Some(tc.id),
                                call_type: Some(tc.call_type),
                                function: Some(crate::protocol::openai_v1::ToolCallFunctionDelta {
                                    name: Some(tc.function.name),
                                    arguments: Some(tc.function.arguments),
                                }),
                            }]),
                        }
                    }
                };
                crate::protocol::openai_v1::ChatCompletionChunk {
                    id: request_id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_name.clone(),
                    system_fingerprint: system_fingerprint.clone(),
                    choices: vec![crate::protocol::openai_v1::ChatCompletionChunkChoice {
                        index: 0,
                        delta,
                        logprobs: None,
                        finish_reason: None,
                    }],
                    usage: None,
                }
            };
            // See non-streaming path: when we primed the prompt's generation
            // prefix with an open `<think>\n`, the model's completion never
            // repeats that tag, so prime the parser's state the same way
            // before consuming the first real token.
            if enable_thinking == Some(true) {
                for ev in parser.feed("<think>\n") {
                    let chunk = emit_event(ev);
                    if let Ok(json_str) = serde_json::to_string(&chunk) {
                        yield Ok(format!("data: {}\n\n", json_str));
                    }
                }
            }
            while let Some(res) = rx.recv().await {
                match res {
                    Ok(token_str) => {
                        generated_count += 1;
                        for ev in parser.feed(&token_str) {
                            let chunk = emit_event(ev);
                            if let Ok(json_str) = serde_json::to_string(&chunk) {
                                yield Ok(format!("data: {}\n\n", json_str));
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Stream error during GenAI token generation: {}", e);
                        break;
                    }
                }
            }
            for ev in parser.finish() {
                let chunk = emit_event(ev);
                if let Ok(json_str) = serde_json::to_string(&chunk) {
                    yield Ok(format!("data: {}\n\n", json_str));
                }
            }

            // 3. Final choice chunk with finish_reason
            let finish_reason = if generated_count >= actual_max_tokens {
                "length".to_string()
            } else if saw_tool_call {
                "tool_calls".to_string()
            } else {
                "stop".to_string()
            };

            let finish_chunk = crate::protocol::openai_v1::ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model_name.clone(),
                system_fingerprint: system_fingerprint.clone(),
                choices: vec![crate::protocol::openai_v1::ChatCompletionChunkChoice {
                    index: 0,
                    delta: crate::protocol::openai_v1::ChatCompletionChunkDelta {
                        role: None,
                        content: None,
                        reasoning_content: None,
                        tool_calls: None,
                    },
                    logprobs: None,
                    finish_reason: Some(finish_reason),
                }],
                usage: None,
            };
            if let Ok(json_str) = serde_json::to_string(&finish_chunk) {
                yield Ok(format!("data: {}\n\n", json_str));
            }

            // 4. Optional usage chunk if stream_options.include_usage == true
            if include_usage {
                let usage_chunk = crate::protocol::openai_v1::ChatCompletionChunk {
                    id: request_id,
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_name,
                    system_fingerprint,
                    choices: vec![],
                    usage: Some(crate::protocol::openai_v1::ChatCompletionUsage {
                        prompt_tokens,
                        completion_tokens: generated_count,
                        total_tokens: prompt_tokens + generated_count,
                    }),
                };
                if let Ok(json_str) = serde_json::to_string(&usage_chunk) {
                    yield Ok(format!("data: {}\n\n", json_str));
                }
            }

            // 5. Termination marker
            yield Ok("data: [DONE]\n\n".to_string());
        };

        Ok(Response::builder()
            .header(
                axum::http::header::CONTENT_TYPE,
                "text/event-stream; charset=utf-8",
            )
            .header(axum::http::header::CACHE_CONTROL, "no-cache, no-transform")
            .header(axum::http::header::CONNECTION, "keep-alive")
            .header("x-accel-buffering", "no")
            .body(Body::from_stream(stream))
            .unwrap())
    } else {
        let mut full_content = String::new();
        let mut token_count = 0;
        while let Some(res) = rx.recv().await {
            match res {
                Ok(token_str) => {
                    full_content.push_str(&token_str);
                    token_count += 1;
                }
                Err(e) => return Err(e),
            }
        }

        let parsed = if enable_thinking == Some(true) {
            // We primed the prompt's generation-prefix with an open `<think>\n`
            // (see render_chat_template), so the model's raw completion never
            // repeats that opening tag itself — it goes straight into
            // reasoning text. Restore the synthetic opening tag before
            // parsing so that text is correctly classified as
            // `reasoning_content` instead of leaking into `content`.
            crate::protocol::tool_call_parser::parse_full_message(&format!(
                "<think>\n{}",
                full_content
            ))
        } else {
            crate::protocol::tool_call_parser::parse_full_message(&full_content)
        };
        let has_tool_calls = !parsed.tool_calls.is_empty();
        let finish_reason = if token_count >= actual_max_tokens {
            "length".to_string()
        } else if has_tool_calls {
            "tool_calls".to_string()
        } else {
            "stop".to_string()
        };

        let resp = crate::protocol::openai_v1::ChatCompletionResponse {
            id: format!(
                "chatcmpl-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis()
            ),
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            model: request.model,
            system_fingerprint: "fp_jetort".to_string(),
            choices: vec![crate::protocol::openai_v1::ChatCompletionChoice {
                index: 0,
                message: crate::protocol::openai_v1::ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: parsed.content,
                    reasoning_content: parsed.reasoning_content,
                    tool_calls: if has_tool_calls {
                        Some(parsed.tool_calls)
                    } else {
                        None
                    },
                    tool_call_id: None,
                    name: None,
                    input_audio: None,
                },
                logprobs: None,
                finish_reason: Some(finish_reason),
            }],
            usage: crate::protocol::openai_v1::ChatCompletionUsage {
                prompt_tokens,
                completion_tokens: token_count,
                total_tokens: prompt_tokens + token_count,
            },
        };

        Ok(Response::builder()
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(&resp).unwrap()))
            .unwrap())
    }
}
