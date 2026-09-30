use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StringOrArray {
    Single(String),
    Multiple(Vec<String>),
}

impl StringOrArray {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StringOrArray::Single(s) => vec![s],
            StringOrArray::Multiple(v) => v,
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAI Embeddings Protocol (/v1/embeddings)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    #[serde(alias = "prompt")]
    pub input: StringOrArray,
    #[serde(default)]
    pub dimensions: Option<usize>,
    #[serde(default)]
    pub encoding_format: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingObject {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

// ---------------------------------------------------------------------------
// Rerank Extension Protocol (/v1/rerank)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    #[serde(default)]
    pub top_n: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankTokens {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankMetaApiVersion {
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankMeta {
    pub api_version: RerankMetaApiVersion,
    pub billed_units: serde_json::Value,
    pub tokens: RerankTokens,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResponse {
    pub id: String,
    pub model: String,
    pub results: Vec<RerankResult>,
    pub usage: RerankUsage,
    pub meta: RerankMeta,
}

// ---------------------------------------------------------------------------
// OpenAI Audio Transcriptions Protocol (/v1/audio/transcriptions)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionUsage {
    pub r#type: String,
    pub seconds: f64,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    pub id: usize,
    pub seek: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub tokens: Vec<u32>,
    pub temperature: f32,
    pub avg_logprob: f32,
    pub compression_ratio: f32,
    pub no_speech_prob: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionVerboseResponse {
    pub task: String,
    pub language: String,
    pub duration: f64,
    pub text: String,
    pub segments: Vec<TranscriptionSegment>,
    pub usage: TranscriptionUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionJsonRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub audio: Option<Vec<f32>>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub response_format: Option<String>,
}

// ---------------------------------------------------------------------------
// OpenAI Model Discovery Protocol (/v1/models)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shutdown_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<ModelMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMeta {
    pub engine: String,
    pub provider: Option<String>,
    pub state: String,
    pub active_requests: usize,
    pub context_length: Option<usize>,
    pub size: u64,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub endpoints: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub precision: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelCard>,
    pub models: Vec<crate::protocol::system_one::TypeSafeModelCard>,
}

// ---------------------------------------------------------------------------
// OpenAI Chat Completions Protocol (/v1/chat/completions)
// ---------------------------------------------------------------------------

fn default_tool_call_type() -> String {
    "function".to_string()
}

/// A single OpenAI-style function tool call, as attached to an assistant
/// message's `tool_calls` array (either in the request, replaying prior
/// conversation turns, or in a response we generated by parsing the model's
/// `<function name="...">...</function>` output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON-encoded string, per the OpenAI convention (not a nested object).
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default = "default_tool_call_type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

/// A single base64-encoded audio clip from an OpenAI-style
/// `{"type": "input_audio", "input_audio": {"data": ..., "format": ...}}` content part.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudioData {
    /// Base64-encoded audio bytes (a WAV/RIFF container, or raw 16-bit PCM).
    pub data: String,
    #[serde(default)]
    pub format: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionMessage {
    pub role: String,
    /// `None` for assistant messages that are pure tool calls (no visible text).
    #[serde(default)]
    pub content: Option<String>,
    /// Model "thinking"/reasoning text, split out of `<think>...</think>`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Present on `role: "tool"` messages, echoing the call this responds to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Present on `role: "tool"` messages: the function name that was called.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// An `input_audio` content part, present only when this incoming message's `content`
    /// was an array containing one (Hybrid audio-LLM models, e.g. ASR). Never emitted on
    /// outgoing responses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_audio: Option<InputAudioData>,
}

// `content` on an incoming request message is either a plain string (the common case) or an
// OpenAI-style array of content parts (`{"type": "text", ...}` / `{"type": "input_audio", ...}`),
// used by Hybrid multimodal models. A derived Deserialize can't populate two struct fields
// (`content` text + `input_audio`) from one JSON key, so this is implemented by hand instead.
impl<'de> Deserialize<'de> for ChatCompletionMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Raw {
            role: String,
            #[serde(default)]
            content: Option<serde_json::Value>,
            #[serde(default)]
            reasoning_content: Option<String>,
            #[serde(default)]
            tool_calls: Option<Vec<ToolCall>>,
            #[serde(default)]
            tool_call_id: Option<String>,
            #[serde(default)]
            name: Option<String>,
        }

        let raw = Raw::deserialize(deserializer)?;
        let mut text = String::new();
        let mut input_audio = None;
        let mut had_content = false;

        match raw.content {
            Some(serde_json::Value::String(s)) => {
                had_content = true;
                text = s;
            }
            Some(serde_json::Value::Array(parts)) => {
                had_content = true;
                for part in parts {
                    match part.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                                text.push_str(t);
                            }
                        }
                        Some("input_audio") => {
                            if let Some(ia) = part.get("input_audio") {
                                if let Ok(parsed) =
                                    serde_json::from_value::<InputAudioData>(ia.clone())
                                {
                                    input_audio = Some(parsed);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }

        Ok(ChatCompletionMessage {
            role: raw.role,
            content: if had_content { Some(text) } else { None },
            reasoning_content: raw.reasoning_content,
            tool_calls: raw.tool_calls,
            tool_call_id: raw.tool_call_id,
            name: raw.name,
            input_audio,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    #[serde(default, alias = "max_completion_tokens")]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Non-standard OpenAI extension (llama.cpp/vLLM-style) for top-k sampling.
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub stop: Option<StringOrArray>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub user: Option<String>,
    /// OpenAI-style function/tool definitions, forwarded verbatim into the
    /// model's Jinja chat_template as `tools` (raw JSON schema objects).
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    /// Accepted for OpenAI compatibility. Only `"none"` is meaningfully acted
    /// on (suppresses tool definitions from the rendered prompt); other values
    /// ("auto"/"required"/a specific function) are passed through as-is since
    /// there is no constrained-decoding enforcement here.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// vLLM/SGLang-style escape hatch for extra chat_template variables, e.g.
    /// `{"enable_thinking": false}` to suppress the model's `<think>` block.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatCompletionMessage,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: ChatCompletionUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolCallFunctionDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunkChoice {
    pub index: usize,
    pub delta: ChatCompletionChunkDelta,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ChatCompletionUsage>,
}

/// One SSE error record. Failure after the stream has started must be this
/// object and nothing else: no finish_reason and no `data: [DONE]`.
pub fn sse_error_frame(message: &str, error_type: &str, code: &str) -> String {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        }
    });
    format!("data: {body}\n\n")
}

/// OpenAI presence/frequency penalties have no GenAI equivalent. A non-default
/// value is rejected. `0` is the OpenAI default and is accepted.
pub fn unsupported_sampling_penalty(
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
) -> Result<(), String> {
    let non_default = |value: Option<f32>| value.is_some_and(|item| item.abs() > f32::EPSILON);
    if non_default(presence_penalty) || non_default(frequency_penalty) {
        return Err("presence_penalty and frequency_penalty are not supported".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arch_sse_error_record_shape() {
        let frame = sse_error_frame("boom", "server_error", "inference_error");
        assert!(frame.starts_with("data: "));
        assert!(frame.ends_with("\n\n"));
        assert!(!frame.contains("[DONE]"));
        assert!(!frame.contains("finish_reason"));
        let json: serde_json::Value =
            serde_json::from_str(frame.trim().trim_start_matches("data: ")).unwrap();
        assert_eq!(json["error"]["message"], "boom");
        assert_eq!(json["error"]["type"], "server_error");
        assert_eq!(json["error"]["code"], "inference_error");
    }

    #[test]
    fn arch_penalties_are_not_mapped() {
        assert!(unsupported_sampling_penalty(None, None).is_ok());
        assert!(unsupported_sampling_penalty(Some(0.0), Some(0.0)).is_ok());
        assert!(unsupported_sampling_penalty(Some(0.5), None).is_err());
        assert!(unsupported_sampling_penalty(None, Some(0.2)).is_err());
    }
}
