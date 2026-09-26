use crate::config::ModelConfig;
use crate::engine::genai::{ExtraModelInput, GenAiSession};
use crate::error::AppError;
use crate::protocol::kserve_v2::TensorMetadata;
use std::sync::Arc;
use tracing::info;

/// Number of audio placeholder tokens (`<|audio_pad|>`) the fused embedding graph expects,
/// per audio clip. Fixed because the audio encoder was exported with a fixed-shape 3000-frame
/// (30s) input: every clip, regardless of its real duration, produces exactly this many
/// `audio_features` rows (see the export-time note in `qwen3_asr_audio_export.py`).
pub const AUDIO_TOKEN_COUNT: usize = 390;
pub const AUDIO_BOS_TOKEN: &str = "<|audio_start|>";
pub const AUDIO_PAD_TOKEN: &str = "<|audio_pad|>";
pub const AUDIO_EOS_TOKEN: &str = "<|audio_end|>";

/// Hybrid Multimodal Session: an audio (or vision) + text model whose encoder, embedding-fusion,
/// and decoder graphs are all wired together in one `genai_config.json` (ORT-GenAI's generic
/// `MultiModalLanguageModel` pipeline, dispatched via `model.type: "lfm2_audio"`). Unlike the
/// older Type-C design, there is no separate One-Shot encoder session here: ORT-GenAI loads and
/// runs the encoder + embedding-fusion + decoder graphs itself from the single `backbone`
/// `GenAiSession`; jetort's only remaining job is computing the mel spectrogram and injecting it
/// (plus its mask/size companions) as extra named model inputs before generation.
pub struct HybridSession {
    pub name: String,
    pub aliases: Vec<String>,
    pub backbone: Arc<GenAiSession>,
}

impl HybridSession {
    pub fn new(name: String, aliases: Vec<String>, backbone: Arc<GenAiSession>) -> Self {
        Self {
            name,
            aliases,
            backbone,
        }
    }

    pub fn load(
        config: &ModelConfig,
        model_path: std::path::PathBuf,
        tokenizer_path: Option<std::path::PathBuf>,
        inspection: crate::engine::onnx_introspect::OnnxModelInspection,
    ) -> Result<Self, AppError> {
        info!(
            model = %config.name,
            path = %model_path.display(),
            "Loading Hybrid multimodal model (single GenAI-driven encoder+embedding+decoder pipeline)..."
        );

        // `genai_config.json`'s `model.type: "lfm2_audio"` routes the whole load through
        // ORT-GenAI's generic MultiModalLanguageModel; GenAiSession::load already points
        // OgaCreateModel at the containing directory, so no separate encoder session is needed.
        let backbone = Arc::new(GenAiSession::load(
            config,
            model_path,
            tokenizer_path,
            inspection,
        )?);

        Ok(Self {
            name: config.name.clone(),
            aliases: config.alias.iter().cloned().collect(),
            backbone,
        })
    }

    pub fn active_provider(&self) -> &str {
        &self.backbone.active_provider
    }

    pub fn inputs_metadata(&self) -> &[TensorMetadata] {
        &self.backbone.inputs_metadata
    }

    pub fn outputs_metadata(&self) -> &[TensorMetadata] {
        &self.backbone.outputs_metadata
    }

    pub async fn acquire_permit(&self) -> Result<tokio::sync::OwnedSemaphorePermit, AppError> {
        self.backbone.acquire_permit().await
    }

    /// Builds the `<|audio_start|><|audio_pad|>*390<|audio_end|>` placeholder text a message's
    /// content should contain wherever this audio clip appears, matching the fixed
    /// `AUDIO_TOKEN_COUNT` the embedding-fusion graph was exported with.
    pub fn audio_placeholder_text() -> String {
        let mut s = String::with_capacity(
            AUDIO_BOS_TOKEN.len() + AUDIO_PAD_TOKEN.len() * AUDIO_TOKEN_COUNT + AUDIO_EOS_TOKEN.len(),
        );
        s.push_str(AUDIO_BOS_TOKEN);
        for _ in 0..AUDIO_TOKEN_COUNT {
            s.push_str(AUDIO_PAD_TOKEN);
        }
        s.push_str(AUDIO_EOS_TOKEN);
        s
    }

    /// Computes the mel spectrogram for one audio clip's PCM samples and packages it as the
    /// three `ExtraModelInput`s the `"lfm2_audio"` speech session expects
    /// (`input_features`, `input_features_mask`, `audio_sizes`; see `genai_config.json`'s
    /// `model.speech.inputs`). The mask is always all-ones and `audio_sizes` is always the
    /// constant `AUDIO_TOKEN_COUNT`, since the encoder graph was exported with a fixed 3000-frame
    /// input (shorter clips are simply zero-padded by `compute_log_mel_128`).
    pub fn build_audio_extra_inputs(pcm_samples: &[f32]) -> Vec<ExtraModelInput> {
        let mel = crate::engine::audio::compute_log_mel_128(pcm_samples);
        const N_MELS: i64 = 128;
        const NB_MAX_FRAMES: i64 = 3000;

        let mel_input =
            ExtraModelInput::from_f32("input_features", mel, vec![1, N_MELS, NB_MAX_FRAMES]);
        let mask_input = ExtraModelInput::from_i64(
            "input_features_mask",
            vec![1i64; NB_MAX_FRAMES as usize],
            vec![1, NB_MAX_FRAMES],
        );
        let sizes_input =
            ExtraModelInput::from_i64("audio_sizes", vec![AUDIO_TOKEN_COUNT as i64], vec![1]);

        vec![mel_input, mask_input, sizes_input]
    }
}
