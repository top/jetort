use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerLiveResponse {
    pub live: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerReadyResponse {
    pub ready: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadataResponse {
    pub name: String,
    pub platform: String,
    pub active_provider: String,
    pub inputs: Vec<TensorMetadata>,
    pub outputs: Vec<TensorMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KServeInputTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub datatype: String,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub parameters: Option<HashMap<String, serde_json::Value>>,
    pub inputs: Vec<KServeInputTensor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KServeOutputTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub datatype: String,
    pub data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub model_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub outputs: Vec<KServeOutputTensor>,
}
