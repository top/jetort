use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use jetort::{config::AppConfig, create_router, engine::ModelRegistry};
use serde_json::Value;
use tower::ServiceExt;

#[tokio::test]
async fn test_health_live_endpoint() {
    let registry = ModelRegistry::new();
    let app = create_router(registry);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v2/health/live")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert!(response.headers().contains_key("x-request-id"));

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["live"], true);
}

#[tokio::test]
async fn test_preserves_request_id() {
    let app = create_router(ModelRegistry::new());
    let response = app
        .oneshot(
            Request::builder()
                .uri("/v2/health/live")
                .header("x-request-id", "caller-request-42")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        response.headers().get("x-request-id").unwrap(),
        "caller-request-42"
    );
}

#[tokio::test]
async fn test_health_ready_endpoint_empty_registry() {
    let registry = ModelRegistry::new();
    let app = create_router(registry);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v2/health/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // With 0 models loaded, service is not ready
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["ready"], false);
}

#[tokio::test]
async fn test_model_not_found_metadata() {
    let registry = ModelRegistry::new();
    let app = create_router(registry);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v2/models/non-existent-model")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(json["error"].as_str().unwrap().contains("not found"));
}

#[tokio::test]
async fn test_model_not_found_infer() {
    let registry = ModelRegistry::new();
    let app = create_router(registry);

    let req_body = serde_json::json!({
        "inputs": [
            {
                "name": "input_ids",
                "shape": [1, 2],
                "datatype": "INT64",
                "data": [1, 2]
            }
        ]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v2/models/unknown-model/infer")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_unloaded_model_returns_retryable_service_unavailable() {
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
    let app = create_router(registry);

    let models_response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let models_body = models_response
        .into_body()
        .collect()
        .await
        .unwrap()
        .to_bytes();
    let models_json: Value = serde_json::from_slice(&models_body).unwrap();
    assert_eq!(models_json["data"][0]["id"], "lazy-model");
    assert_eq!(models_json["data"][0]["meta"]["state"], "unloaded");

    let request = serde_json::json!({
        "model": "lazy-model",
        "input": "hello"
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(response.headers().get("retry-after").unwrap(), "2");
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["code"], "model_not_loaded");
    assert_eq!(json["state"], "loading");
    assert_eq!(json["retryable"], true);
}

#[tokio::test]
async fn test_audio_transcriptions_model_not_found() {
    let registry = ModelRegistry::new();
    let app = create_router(registry);

    let req_body = serde_json::json!({
        "model": "non-existent-asr",
        "audio": [0.01, 0.02]
    });

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/transcriptions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_openai_models_list_endpoint() {
    let registry = ModelRegistry::new();
    let app = create_router(registry);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "list");
    assert!(json["data"].is_array());
    assert!(json["models"].is_array());
}

#[tokio::test]
async fn test_system_one_validation_and_request_id() {
    let registry = ModelRegistry::new();
    let app = create_router(registry);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/systemone")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"","state":"x","questions":{}}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert!(response.headers().contains_key("x-typesafe-request-id"));

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/systemone")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{
                        "model":"missing",
                        "state":"ticket",
                        "questions":{
                            "urgent":{"type":"noul","instructions":"Is this urgent?"}
                        }
                    }"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    assert!(response.headers().contains_key("x-typesafe-request-id"));
}

#[tokio::test]
async fn test_route_aliases() {
    let registry = ModelRegistry::new();
    let app = create_router(registry);

    // 1. GET /models alias
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // 2. POST /embeddings route alias
    let req_body = serde_json::json!({
        "model": "non-existent",
        "input": "test"
    });
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/embeddings")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(json["error"].as_str().unwrap().contains("not found"));

    // 3. POST /chat/completions route alias
    let chat_body = serde_json::json!({
        "model": "non-existent",
        "messages": [{"role": "user", "content": "hi"}]
    });
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&chat_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_lenient_content_type_handling() {
    let registry = ModelRegistry::new();
    let app = create_router(registry);

    let req_body = serde_json::json!({
        "model": "non-existent",
        "input": "test query"
    });

    // 1. Completely missing Content-Type header
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    // Axum handler should receive the JSON and return 404 (model not found), NOT 415 Unsupported Media Type
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(json["error"].as_str().unwrap().contains("not found"));

    // 2. Default curl -d header (application/x-www-form-urlencoded)
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);

    // 3. text/plain header
    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header("content-type", "text/plain")
                .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_audio_transcriptions_multipart() {
    let registry = ModelRegistry::new();
    let app = create_router(registry);

    let boundary = "---------------------------974767299852498929531610575";
    let mut body_bytes = Vec::new();
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"model\"\r\n\r\nnon-existent-asr\r\n",
    );
    body_bytes.extend_from_slice(format!("--{}\r\n", boundary).as_bytes());
    body_bytes.extend_from_slice(b"Content-Disposition: form-data; name=\"file\"; filename=\"speech.wav\"\r\nContent-Type: audio/wav\r\n\r\n");
    // Minimal 44-byte valid WAV header
    body_bytes.extend_from_slice(b"RIFF\x24\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x80\x3e\0\0\0\x7d\0\0\x02\0\x10\0data\0\0\0\0");
    body_bytes.extend_from_slice(b"\r\n");
    body_bytes.extend_from_slice(format!("--{}--\r\n", boundary).as_bytes());

    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/transcriptions")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={}", boundary),
                )
                .body(Body::from(body_bytes))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(json["error"].as_str().unwrap().contains("not found"));
}

#[tokio::test]
async fn test_rerank_model_not_found() {
    let registry = ModelRegistry::new();
    let app = create_router(registry);

    let req_body = serde_json::json!({
        "model": "non-existent-reranker",
        "query": "hello",
        "documents": ["world"]
    });

    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/rerank")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(json["error"].as_str().unwrap().contains("not found"));
}

#[tokio::test]
async fn test_health_memory_endpoint() {
    let registry = ModelRegistry::new();
    let app = create_router(registry);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/health/memory")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(json.get("process_rss_mb").is_some());
    assert!(json.get("system_available_mb").is_some());
}

#[tokio::test]
async fn test_chat_completions_model_not_found() {
    let registry = ModelRegistry::new();
    let app = create_router(registry);

    let req_body = serde_json::json!({
        "model": "non-existent-llm",
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    });

    let res = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&req_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::NOT_FOUND);
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(json["error"].as_str().unwrap().contains("not found"));
}

#[test]
fn test_chat_template_rendering() {
    use jetort::engine::genai::{ChatMessage, GenAiSession};
    use std::path::PathBuf;
    use std::sync::Arc;

    let session = GenAiSession {
        name: "test-qwen".to_string(),
        aliases: vec![],
        active_provider: "test".to_string(),
        inputs_metadata: vec![],
        outputs_metadata: vec![],
        model_dir: PathBuf::from("."),
        chat_template: Some(
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}".to_string()
        ),
        hf_tokenizer: None,
        model: None,
        tokenizer: None,
        inference_semaphore: Arc::new(tokio::sync::Semaphore::new(1)),
        init_error: None,
        n_ctx_train: 131072,
        n_ctx: 4096,
        n_batch: 512,
        model_path: PathBuf::from("model.onnx"),
        default_temperature: None,
        default_top_p: None,
        default_top_k: None,
        default_repetition_penalty: None,
    };

    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant.".to_string(),
            reasoning_content: None,
            tool_calls: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Hello, how are you?".to_string(),
            reasoning_content: None,
            tool_calls: None,
        },
    ];

    let rendered = session
        .render_chat_template(&messages, true, &[], Some(true))
        .expect("Rendering should succeed");
    assert!(rendered.contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>"));
    assert!(rendered.contains("<|im_start|>user\nHello, how are you?<|im_end|>"));
    assert!(rendered.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn test_openai_streaming_chunks_format() {
    use jetort::protocol::openai_v1::{
        ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionChunkDelta,
        ChatCompletionUsage,
    };

    // 1. Initial chunk
    let initial_chunk = ChatCompletionChunk {
        id: "chatcmpl-test".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1727188400,
        model: "MiniCPM5-2B".to_string(),
        system_fingerprint: "fp_jetort".to_string(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: ChatCompletionChunkDelta {
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
    let initial_json: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&initial_chunk).unwrap()).unwrap();
    assert_eq!(initial_json["object"], "chat.completion.chunk");
    assert_eq!(initial_json["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(initial_json["choices"][0]["delta"]["content"], "");
    assert!(initial_json["choices"][0]["logprobs"].is_null());
    assert!(initial_json["choices"][0]["finish_reason"].is_null());
    assert!(initial_json.get("usage").is_none());

    // 2. Token content chunk
    let token_chunk = ChatCompletionChunk {
        id: "chatcmpl-test".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1727188400,
        model: "MiniCPM5-2B".to_string(),
        system_fingerprint: "fp_jetort".to_string(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: ChatCompletionChunkDelta {
                role: None,
                content: Some("Hello".to_string()),
                reasoning_content: None,
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: None,
        }],
        usage: None,
    };
    let token_json: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&token_chunk).unwrap()).unwrap();
    assert_eq!(token_json["choices"][0]["delta"]["content"], "Hello");
    assert!(token_json["choices"][0]["delta"].get("role").is_none());
    assert!(token_json["choices"][0]["finish_reason"].is_null());

    // 3. Final choice chunk
    let finish_chunk = ChatCompletionChunk {
        id: "chatcmpl-test".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1727188400,
        model: "MiniCPM5-2B".to_string(),
        system_fingerprint: "fp_jetort".to_string(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: ChatCompletionChunkDelta {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: Some("stop".to_string()),
        }],
        usage: None,
    };
    let finish_json: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&finish_chunk).unwrap()).unwrap();
    assert_eq!(finish_json["choices"][0]["finish_reason"], "stop");
    assert_eq!(finish_json["choices"][0]["delta"], serde_json::json!({}));

    // 4. Optional usage chunk
    let usage_chunk = ChatCompletionChunk {
        id: "chatcmpl-test".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1727188400,
        model: "MiniCPM5-2B".to_string(),
        system_fingerprint: "fp_jetort".to_string(),
        choices: vec![],
        usage: Some(ChatCompletionUsage {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
        }),
    };
    let usage_json: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&usage_chunk).unwrap()).unwrap();
    assert_eq!(usage_json["choices"], serde_json::json!([]));
    assert_eq!(usage_json["usage"]["total_tokens"], 30);
}

#[tokio::test]
async fn test_context_length_exceeded_error_response() {
    use axum::response::IntoResponse;
    use jetort::error::AppError;

    let err = AppError::ContextLengthExceeded(
        "This model's maximum context length is 4096 tokens. However, your messages resulted in 5000 tokens. Please reduce the length of the messages.".to_string(),
    );
    let resp = err.into_response();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["error"]
        .as_str()
        .unwrap()
        .contains("maximum context length is 4096"));
}
