//! Parses model-generated text into structured `reasoning_content` / `tool_calls`
//! for MiniCPM-style output, which wraps reasoning in `<think>...</think>` and
//! tool invocations in `<function name="...">‌<param name="...">value</param></function>`
//! XML tags (see the model's `chat_template.jinja`). Without this parsing, jetort
//! would return the raw tags as plain text, which OpenAI-compatible agent clients
//! cannot consume (they expect `message.tool_calls`, not embedded XML).

use crate::protocol::openai_v1::{ToolCall, ToolCallFunction};
use serde_json::{Map, Value};

/// Result of parsing a complete (non-streamed) assistant completion.
#[derive(Debug, Default, Clone)]
pub struct ParsedMessage {
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

/// Splits `<think>...</think>` out of `text`, returning `(reasoning, remainder)`.
/// Tolerates an unterminated `<think>` (e.g. generation cut off mid-thought by
/// `max_tokens`) by treating everything after the opening tag as reasoning.
fn extract_think(text: &str) -> (Option<String>, String) {
    if let Some(start) = text.find("<think>") {
        let after_open = start + "<think>".len();
        if let Some(rel_end) = text[after_open..].find("</think>") {
            let reasoning = text[after_open..after_open + rel_end].trim().to_string();
            let before = &text[..start];
            let after = &text[after_open + rel_end + "</think>".len()..];
            let remainder = format!("{}{}", before, after);
            return (
                if reasoning.is_empty() { None } else { Some(reasoning) },
                remainder,
            );
        }
        // Unterminated <think>: everything after it is reasoning, nothing follows.
        let reasoning = text[after_open..].trim().to_string();
        let before = text[..start].to_string();
        return (
            if reasoning.is_empty() { None } else { Some(reasoning) },
            before,
        );
    }
    (None, text.to_string())
}

/// Unwraps a `<![CDATA[...]]>` block if present, otherwise returns the input unescaped.
fn decode_param_value(raw: &str) -> String {
    let trimmed = raw.trim();
    if let Some(inner) = trimmed
        .strip_prefix("<![CDATA[")
        .and_then(|s| s.strip_suffix("]]>"))
    {
        inner.to_string()
    } else {
        trimmed
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
    }
}

/// Coerces a decoded param string into a JSON value: valid JSON (number/bool/
/// array/object) is parsed as such, otherwise it's kept as a JSON string.
fn coerce_json_value(raw: &str) -> Value {
    let trimmed = raw.trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        if v.is_number() || v.is_boolean() || v.is_array() || v.is_object() || v.is_null() {
            return v;
        }
    }
    Value::String(raw.to_string())
}

fn extract_attr<'a>(tag: &'a str, attr: &str) -> Option<&'a str> {
    let needle = format!("{attr}=\"");
    let start = tag.find(&needle)? + needle.len();
    let end = start + tag[start..].find('"')?;
    Some(&tag[start..end])
}

/// Parses all `<param name="...">value</param>` occurrences out of a function
/// call's body text into a JSON arguments object.
fn parse_params(body: &str) -> Map<String, Value> {
    let mut args = Map::new();
    let mut cursor = 0usize;
    while let Some(rel_start) = body[cursor..].find("<param") {
        let start = cursor + rel_start;
        let Some(rel_tag_end) = body[start..].find('>') else {
            break;
        };
        let tag_end = start + rel_tag_end + 1;
        let opening = &body[start..tag_end];
        let Some(rel_close) = body[tag_end..].find("</param>") else {
            break;
        };
        let close = tag_end + rel_close;
        let value_body = &body[tag_end..close];
        cursor = close + "</param>".len();

        if let Some(name) = extract_attr(opening, "name") {
            args.insert(name.to_string(), coerce_json_value(&decode_param_value(value_body)));
        }
    }
    args
}

/// Builds a `ToolCall` from a function name, its raw body, and a sequential
/// index (used to generate a synthetic `call_N` id, since ORT-GenAI doesn't
/// produce one).
fn build_tool_call(name: String, body: &str, index: usize) -> ToolCall {
    let args = parse_params(body);
    ToolCall {
        id: format!("call_{index}"),
        call_type: "function".to_string(),
        function: ToolCallFunction {
            name,
            arguments: Value::Object(args).to_string(),
        },
    }
}

/// Parses all `<function name="...">...</function>` blocks out of `text`.
/// Returns the remaining plain-text content (with function blocks removed) and
/// the extracted tool calls. An unterminated trailing `<function` fragment
/// (generation cut off mid-call) is dropped from the visible content rather
/// than leaked to the client as broken text.
fn extract_function_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut remainder = String::new();
    let mut calls = Vec::new();
    let mut cursor = 0usize;

    while let Some(rel_start) = text[cursor..].find("<function") {
        let abs_start = cursor + rel_start;
        remainder.push_str(&text[cursor..abs_start]);

        // Find the end of the opening tag ('>') to read its attributes.
        let Some(rel_tag_end) = text[abs_start..].find('>') else {
            // Unterminated opening tag; drop the trailing fragment entirely.
            cursor = text.len();
            break;
        };
        let tag_end = abs_start + rel_tag_end + 1;
        let opening_tag = &text[abs_start..tag_end];
        let name = extract_attr(opening_tag, "name").unwrap_or("").to_string();

        let Some(rel_close) = text[tag_end..].find("</function>") else {
            // Unterminated call body; drop the trailing fragment entirely.
            cursor = text.len();
            break;
        };
        let body_end = tag_end + rel_close;
        let body = &text[tag_end..body_end];
        cursor = body_end + "</function>".len();

        if !name.is_empty() {
            calls.push(build_tool_call(name, body, calls.len()));
        }
    }
    remainder.push_str(&text[cursor..]);
    (remainder, calls)
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const FUNC_OPEN_PREFIX: &str = "<function";
const FUNC_CLOSE: &str = "</function>";

/// Returns the length of the longest suffix of `s` that is a strict prefix of
/// `marker` (i.e. text that *might* still turn into `marker` given more
/// input, and so must not be emitted as plain text yet).
fn ambiguous_suffix_len(s: &str, marker: &str) -> usize {
    let max_len = marker.len().saturating_sub(1).min(s.len());
    for len in (1..=max_len).rev() {
        let start = s.len() - len;
        if !s.is_char_boundary(start) {
            continue;
        }
        if marker.starts_with(&s[start..]) {
            return len;
        }
    }
    0
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Content(String),
    Reasoning(String),
    ToolCall(ToolCall),
}

#[derive(Default)]
enum ModeState {
    #[default]
    Plain,
    Think,
    Function {
        name: String,
    },
}

/// Incremental version of [`parse_full_message`] for use while streaming
/// tokens to the client. Buffers text internally whenever it might be a
/// partial tag, so no visible `content`/`reasoning_content` chunk ever
/// contains a half-printed `<think>`/`<function>` tag. A `<function>` call
/// that never closes (e.g. generation cut off by `max_tokens`) is silently
/// dropped rather than leaked to the client as broken text.
#[derive(Default)]
pub struct StreamParser {
    buf: String,
    call_count: usize,
    mode: ModeState,
}

impl StreamParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a newly generated text chunk. Returns zero or more events ready
    /// to emit to the client immediately.
    pub fn feed(&mut self, chunk: &str) -> Vec<StreamEvent> {
        self.buf.push_str(chunk);
        self.drain(false)
    }

    /// Call once generation ends to flush any remaining buffered text.
    pub fn finish(mut self) -> Vec<StreamEvent> {
        self.drain(true)
    }

    fn drain(&mut self, is_final: bool) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        loop {
            match std::mem::take(&mut self.mode) {
                ModeState::Plain => {
                    let think_pos = self.buf.find(THINK_OPEN);
                    let func_pos = self.buf.find(FUNC_OPEN_PREFIX);
                    let earliest = match (think_pos, func_pos) {
                        (Some(t), Some(f)) => Some(t.min(f)),
                        (Some(t), None) => Some(t),
                        (None, Some(f)) => Some(f),
                        (None, None) => None,
                    };
                    match earliest {
                        Some(pos) if think_pos == Some(pos) => {
                            if pos > 0 {
                                events.push(StreamEvent::Content(self.buf[..pos].to_string()));
                            }
                            self.buf.drain(..pos + THINK_OPEN.len());
                            self.mode = ModeState::Think;
                        }
                        Some(pos) => {
                            // func_pos == Some(pos): need the '>' closing the opening tag.
                            if let Some(rel_gt) = self.buf[pos..].find('>') {
                                if pos > 0 {
                                    events.push(StreamEvent::Content(self.buf[..pos].to_string()));
                                }
                                let tag_end = pos + rel_gt + 1;
                                let name = extract_attr(&self.buf[pos..tag_end], "name")
                                    .unwrap_or("")
                                    .to_string();
                                self.buf.drain(..tag_end);
                                self.mode = ModeState::Function { name };
                            } else if is_final {
                                if pos > 0 {
                                    events.push(StreamEvent::Content(self.buf[..pos].to_string()));
                                }
                                self.buf.clear();
                                self.mode = ModeState::Plain;
                                break;
                            } else {
                                if pos > 0 {
                                    let text: String = self.buf.drain(..pos).collect();
                                    events.push(StreamEvent::Content(text));
                                }
                                self.mode = ModeState::Plain;
                                break;
                            }
                        }
                        None => {
                            let safe_len = if is_final {
                                self.buf.len()
                            } else {
                                let think_amb = ambiguous_suffix_len(&self.buf, THINK_OPEN);
                                let func_amb = ambiguous_suffix_len(&self.buf, FUNC_OPEN_PREFIX);
                                self.buf.len() - think_amb.max(func_amb)
                            };
                            if safe_len > 0 {
                                let text: String = self.buf.drain(..safe_len).collect();
                                events.push(StreamEvent::Content(text));
                            }
                            self.mode = ModeState::Plain;
                            break;
                        }
                    }
                }
                ModeState::Think => {
                    if let Some(pos) = self.buf.find(THINK_CLOSE) {
                        let reasoning: String = self.buf.drain(..pos + THINK_CLOSE.len()).collect();
                        let reasoning = reasoning[..reasoning.len() - THINK_CLOSE.len()].to_string();
                        if !reasoning.is_empty() {
                            events.push(StreamEvent::Reasoning(reasoning));
                        }
                        self.mode = ModeState::Plain;
                    } else {
                        let safe_len = if is_final {
                            self.buf.len()
                        } else {
                            self.buf.len() - ambiguous_suffix_len(&self.buf, THINK_CLOSE)
                        };
                        if safe_len > 0 {
                            let text: String = self.buf.drain(..safe_len).collect();
                            events.push(StreamEvent::Reasoning(text));
                        }
                        self.mode = ModeState::Think;
                        break;
                    }
                }
                ModeState::Function { name } => {
                    if let Some(pos) = self.buf.find(FUNC_CLOSE) {
                        let body: String = self.buf.drain(..pos + FUNC_CLOSE.len()).collect();
                        let body = &body[..body.len() - FUNC_CLOSE.len()];
                        if !name.is_empty() {
                            events.push(StreamEvent::ToolCall(build_tool_call(
                                name,
                                body,
                                self.call_count,
                            )));
                            self.call_count += 1;
                        }
                        self.mode = ModeState::Plain;
                    } else if is_final {
                        // Unterminated tool call at end of generation: drop silently
                        // instead of leaking broken tags to the client.
                        self.buf.clear();
                        self.mode = ModeState::Plain;
                        break;
                    } else {
                        self.mode = ModeState::Function { name };
                        break;
                    }
                }
            }
        }
        events
    }
}

/// Parses a complete assistant completion into content / reasoning / tool_calls.
pub fn parse_full_message(text: &str) -> ParsedMessage {
    let (reasoning_content, after_think) = extract_think(text);
    let (remaining, tool_calls) = extract_function_calls(&after_think);
    let trimmed = remaining.trim();
    ParsedMessage {
        content: if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        },
        reasoning_content,
        tool_calls,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_reasoning_and_plain_content() {
        let text = "<think>\nI should greet the user\n</think>\n\nHello there!";
        let parsed = parse_full_message(text);
        assert_eq!(parsed.reasoning_content.as_deref(), Some("I should greet the user"));
        assert_eq!(parsed.content.as_deref(), Some("Hello there!"));
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn parses_single_function_call() {
        let text = "<think>\nLet me call notebook.list\n</think>\n\n<function name=\"notebook.list\"><param name=\"limit\">10</param></function>";
        let parsed = parse_full_message(text);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].function.name, "notebook.list");
        assert_eq!(parsed.tool_calls[0].function.arguments, "{\"limit\":10}");
        assert!(parsed.content.is_none());
    }

    #[test]
    fn parses_cdata_param_value() {
        let text = "<function name=\"notes.write\"><param name=\"body\"><![CDATA[a <b> & c]]></param></function>";
        let parsed = parse_full_message(text);
        assert_eq!(parsed.tool_calls[0].function.arguments, "{\"body\":\"a <b> & c\"}");
    }

    #[test]
    fn drops_unterminated_trailing_function_tag() {
        let text = "我需要先找到笔记。\n<function name=\"notebook.list\"><param name=\"q\">Gen2";
        let parsed = parse_full_message(text);
        assert!(parsed.tool_calls.is_empty());
        assert_eq!(parsed.content.as_deref(), Some("我需要先找到笔记。"));
    }

    #[test]
    fn unterminated_think_becomes_reasoning() {
        let text = "<think>\nstill thinking about this";
        let parsed = parse_full_message(text);
        assert_eq!(parsed.reasoning_content.as_deref(), Some("still thinking about this"));
        assert!(parsed.content.is_none());
    }

    fn feed_char_by_char(parser: &mut StreamParser, text: &str) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        for ch in text.chars() {
            events.extend(parser.feed(&ch.to_string()));
        }
        events
    }

    #[test]
    fn stream_splits_reasoning_and_content_token_by_token() {
        let mut parser = StreamParser::new();
        let mut events = feed_char_by_char(&mut parser, "<think>\nhi\n</think>\n\nHello!");
        events.extend(parser.finish());

        let reasoning: String = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Reasoning(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        let content: String = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Content(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning.trim(), "hi");
        assert_eq!(content.trim(), "Hello!");
    }

    #[test]
    fn stream_emits_complete_tool_call_and_never_leaks_partial_tags() {
        let mut parser = StreamParser::new();
        let text = "before <function name=\"notebook.list\"><param name=\"q\">Gen2</param></function> after";
        let mut events = feed_char_by_char(&mut parser, text);
        events.extend(parser.finish());

        let tool_calls: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCall(tc) => Some(tc.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "notebook.list");
        assert_eq!(tool_calls[0].function.arguments, "{\"q\":\"Gen2\"}");

        let content: String = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Content(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert!(!content.contains('<'), "no raw tag characters should leak into content: {content:?}");
        assert_eq!(content.trim(), "before  after");
    }

    #[test]
    fn stream_drops_unterminated_tool_call_at_finish() {
        let mut parser = StreamParser::new();
        let mut events = feed_char_by_char(
            &mut parser,
            "我需要先找到笔记。\n<function name=\"notebook.list\"><param name=\"q\">Gen2",
        );
        events.extend(parser.finish());

        assert!(events.iter().all(|e| !matches!(e, StreamEvent::ToolCall(_))));
        let content: String = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Content(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert!(!content.contains('<'), "no partial tag should leak: {content:?}");
    }
}
