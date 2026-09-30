//! llama.cpp-style inference timing lines.
//!
//! One reporter for every successful inference path. The message text itself is
//! the printable block (not only structured fields), so it always shows at INFO
//! regardless of how a subscriber formats spans.

use std::fmt::Write as _;
use std::time::Instant;

/// Successful one-shot / embedding-style timing.
#[derive(Debug, Clone)]
pub struct OneShotTimings {
    pub task: String,
    pub model: String,
    pub backend: String,
    pub items: usize,
    pub input_tokens: usize,
    pub eval_ms: f64,
    pub queue_ms: f64,
}

/// Chat / generation timing (streaming prints once when the stream ends).
#[derive(Debug, Clone)]
pub struct ChatTimings {
    pub task: String,
    pub model: String,
    pub backend: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prompt_ms: f64,
    pub generation_ms: f64,
    pub total_ms: f64,
    pub queue_ms: f64,
}

/// Audio transcription timing.
#[derive(Debug, Clone)]
pub struct TranscriptionTimings {
    pub task: String,
    pub model: String,
    pub backend: String,
    pub audio_seconds: f64,
    pub frames: usize,
    pub output_tokens: usize,
    pub eval_ms: f64,
    pub queue_ms: f64,
}

pub fn format_one_shot(t: &OneShotTimings) -> String {
    let tps = if t.eval_ms > 0.0 {
        t.input_tokens as f64 / (t.eval_ms / 1000.0)
    } else {
        0.0
    };
    format!(
        "print_timings: task={} model={} backend={} items={} input={} eval_ms={:.2} tokens_per_second={:.2} queue_ms={:.2}",
        t.task, t.model, t.backend, t.items, t.input_tokens, t.eval_ms, tps, t.queue_ms
    )
}

pub fn format_chat(t: &ChatTimings) -> String {
    let prompt_tps = if t.prompt_ms > 0.0 {
        t.prompt_tokens as f64 / (t.prompt_ms / 1000.0)
    } else {
        0.0
    };
    let gen_tps = if t.generation_ms > 0.0 && t.generated_tokens > 0 {
        t.generated_tokens as f64 / (t.generation_ms / 1000.0)
    } else {
        0.0
    };
    format!(
        "print_timings: task={} model={} backend={} prompt_n={} prompt_ms={:.2} prompt_tok_s={:.2} gen_n={} gen_ms={:.2} gen_tok_s={:.2} total_ms={:.2} queue_ms={:.2}",
        t.task,
        t.model,
        t.backend,
        t.prompt_tokens,
        t.prompt_ms,
        prompt_tps,
        t.generated_tokens,
        t.generation_ms,
        gen_tps,
        t.total_ms,
        t.queue_ms
    )
}

pub fn format_transcription(t: &TranscriptionTimings) -> String {
    let rtf = if t.audio_seconds > 0.0 {
        (t.eval_ms / 1000.0) / t.audio_seconds
    } else {
        0.0
    };
    format!(
        "print_timings: task={} model={} backend={} audio_s={:.2} frames={} output_tokens={} eval_ms={:.2} realtime_factor={:.3} queue_ms={:.2}",
        t.task,
        t.model,
        t.backend,
        t.audio_seconds,
        t.frames,
        t.output_tokens,
        t.eval_ms,
        rtf,
        t.queue_ms
    )
}

pub fn log_one_shot(t: &OneShotTimings) {
    let line = format_one_shot(t);
    tracing::info!("{line}");
}

pub fn log_chat(t: &ChatTimings) {
    let line = format_chat(t);
    tracing::info!("{line}");
}

pub fn log_transcription(t: &TranscriptionTimings) {
    let line = format_transcription(t);
    tracing::info!("{line}");
}

/// Convenience: build one-shot timings from wall clocks.
pub fn one_shot_from_clocks(
    task: &str,
    model: &str,
    backend: &str,
    items: usize,
    input_tokens: usize,
    started: Instant,
    queue_ms: f64,
) -> OneShotTimings {
    OneShotTimings {
        task: task.to_string(),
        model: model.to_string(),
        backend: backend.to_string(),
        items,
        input_tokens,
        eval_ms: started.elapsed().as_secs_f64() * 1000.0,
        queue_ms,
    }
}

/// Append key=value pairs without allocating intermediate formatters when needed by tests.
pub fn render_kv(pairs: &[(&str, String)]) -> String {
    let mut out = String::from("print_timings:");
    for (k, v) in pairs {
        let _ = write!(out, " {k}={v}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timing_line_embedding_shape() {
        let line = format_one_shot(&OneShotTimings {
            task: "embedding".into(),
            model: "Qwen3-Embedding-0.6B".into(),
            backend: "TensorRT".into(),
            items: 10,
            input_tokens: 1657,
            eval_ms: 42.5,
            queue_ms: 1.2,
        });
        assert!(line.starts_with("print_timings:"));
        assert!(line.contains("task=embedding"));
        assert!(line.contains("model=Qwen3-Embedding-0.6B"));
        assert!(line.contains("backend=TensorRT"));
        assert!(line.contains("items=10"));
        assert!(line.contains("input=1657"));
        assert!(line.contains("eval_ms=42.50"));
        assert!(line.contains("tokens_per_second="));
        assert!(line.contains("queue_ms=1.20"));
    }

    #[test]
    fn timing_line_chat_shape() {
        let line = format_chat(&ChatTimings {
            task: "chat_completion".into(),
            model: "MiniCPM5-2B".into(),
            backend: "CUDA".into(),
            prompt_tokens: 128,
            generated_tokens: 64,
            prompt_ms: 80.0,
            generation_ms: 200.0,
            total_ms: 280.0,
            queue_ms: 0.5,
        });
        assert!(line.starts_with("print_timings:"));
        assert!(line.contains("prompt_n=128"));
        assert!(line.contains("gen_n=64"));
        assert!(line.contains("prompt_tok_s="));
        assert!(line.contains("gen_tok_s="));
        assert!(line.contains("total_ms=280.00"));
        assert!(line.contains("backend=CUDA"));
    }

    #[test]
    fn timing_line_zero_eval_no_div0() {
        let line = format_one_shot(&OneShotTimings {
            task: "rerank".into(),
            model: "r".into(),
            backend: "CPU".into(),
            items: 1,
            input_tokens: 10,
            eval_ms: 0.0,
            queue_ms: 0.0,
        });
        assert!(line.contains("tokens_per_second=0.00"));
    }
}
