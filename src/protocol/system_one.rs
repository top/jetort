use crate::error::AppError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const ABSTENTION_ID: &str = "__insufficient_evidence__";
pub const ABSTENTION_DESCRIPTION: &str = "insufficient evidence";
pub const MAX_OPENJEV_OPTIONS: usize = 24;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemOneRequest {
    pub model: String,
    pub state: Value,
    pub questions: BTreeMap<String, SystemOneQuestion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SystemOneQuestion {
    Noul {
        #[serde(default)]
        instructions: Value,
        #[serde(default)]
        criteria: Option<NoulCriteria>,
    },
    Choice {
        #[serde(default)]
        instructions: Value,
        criteria: BTreeMap<String, Value>,
    },
    Score {
        #[serde(default)]
        instructions: Value,
        criteria: Vec<Value>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(rename = "true", default)]
    pub true_value: Value,
    #[serde(rename = "false", default)]
    pub false_value: Value,
}

#[derive(Debug, Clone)]
pub struct PreparedQuestion {
    pub id: String,
    pub kind: PreparedQuestionKind,
    pub prompt: String,
    pub candidate_ids: Vec<String>,
    pub candidate_descriptions: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum PreparedQuestionKind {
    Noul,
    Choice,
    Score { legend: BTreeMap<String, Value> },
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: BTreeMap<String, SystemOneAnswer>,
    pub usage: SystemOneUsage,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SystemOneAnswer {
    Noul {
        noul: f32,
    },
    Choice {
        choice: String,
        confidence: f32,
        probabilities: BTreeMap<String, f32>,
    },
    Score {
        score: f32,
        confidence: f32,
        legend: BTreeMap<String, Value>,
        probabilities: BTreeMap<String, f32>,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemOneUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeSafeModelCard {
    pub name: String,
    pub description: String,
    pub release_date: String,
}

pub fn validate_request_shape(request: &SystemOneRequest) -> Result<(), AppError> {
    if request.model.trim().is_empty() {
        return Err(AppError::Unprocessable(
            "model must not be empty".to_string(),
        ));
    }
    validate_entry(&request.state, "state")?;
    if request.questions.is_empty() {
        return Err(AppError::Unprocessable(
            "questions must contain at least one question".to_string(),
        ));
    }
    Ok(())
}

pub fn prepare_request(request: &SystemOneRequest) -> Result<Vec<PreparedQuestion>, AppError> {
    validate_request_shape(request)?;

    request
        .questions
        .iter()
        .map(|(id, question)| prepare_question(id, &request.state, question))
        .collect()
}

fn prepare_question(
    id: &str,
    state: &Value,
    question: &SystemOneQuestion,
) -> Result<PreparedQuestion, AppError> {
    if id.trim().is_empty() {
        return Err(AppError::Unprocessable(
            "question ids must not be empty".to_string(),
        ));
    }

    let state = entry_text(state);
    match question {
        SystemOneQuestion::Choice {
            instructions,
            criteria,
        } => {
            validate_entry(instructions, &format!("questions.{id}.instructions"))?;
            if !(2..=MAX_OPENJEV_OPTIONS).contains(&criteria.len()) {
                return Err(AppError::Unprocessable(format!(
                    "questions.{id}.criteria must contain 2 to {MAX_OPENJEV_OPTIONS} options for this model"
                )));
            }
            let mut candidate_ids = Vec::with_capacity(criteria.len() + 1);
            let mut candidate_descriptions = Vec::with_capacity(criteria.len() + 1);
            for (option, description) in criteria {
                if option.is_empty() || option == ABSTENTION_ID {
                    return Err(AppError::Unprocessable(format!(
                        "questions.{id}.criteria contains an invalid option id"
                    )));
                }
                validate_entry(description, &format!("questions.{id}.criteria.{option}"))?;
                candidate_ids.push(option.clone());
                candidate_descriptions.push(format!("It is {option}: {}", entry_text(description)));
            }
            candidate_ids.push(ABSTENTION_ID.to_string());
            candidate_descriptions.push(ABSTENTION_DESCRIPTION.to_string());
            Ok(PreparedQuestion {
                id: id.to_string(),
                kind: PreparedQuestionKind::Choice,
                prompt: build_prompt(&entry_text(instructions), &state, &candidate_descriptions),
                candidate_ids,
                candidate_descriptions,
            })
        }
        SystemOneQuestion::Score {
            instructions,
            criteria,
        } => {
            validate_entry(instructions, &format!("questions.{id}.instructions"))?;
            if !(2..=10).contains(&criteria.len()) {
                return Err(AppError::Unprocessable(format!(
                    "questions.{id}.criteria must contain 2 to 10 levels"
                )));
            }
            let mut legend = BTreeMap::new();
            let mut candidate_ids = Vec::with_capacity(criteria.len() + 1);
            let mut candidate_descriptions = Vec::with_capacity(criteria.len() + 1);
            for (index, description) in criteria.iter().enumerate() {
                validate_entry(description, &format!("questions.{id}.criteria.{index}"))?;
                let key = index.to_string();
                legend.insert(key.clone(), description.clone());
                candidate_ids.push(key);
                candidate_descriptions
                    .push(format!("{} (Value: {index})", entry_text(description)));
            }
            candidate_ids.push(ABSTENTION_ID.to_string());
            candidate_descriptions.push(ABSTENTION_DESCRIPTION.to_string());
            Ok(PreparedQuestion {
                id: id.to_string(),
                kind: PreparedQuestionKind::Score { legend },
                prompt: build_prompt(&entry_text(instructions), &state, &candidate_descriptions),
                candidate_ids,
                candidate_descriptions,
            })
        }
        SystemOneQuestion::Noul {
            instructions,
            criteria,
        } => {
            validate_entry(instructions, &format!("questions.{id}.instructions"))?;
            let proposition = entry_text(instructions);
            let (true_description, false_description) = criteria
                .as_ref()
                .map(|criteria| {
                    (
                        if criteria.true_value.is_null() {
                            proposition.clone()
                        } else {
                            entry_text(&criteria.true_value)
                        },
                        if criteria.false_value.is_null() {
                            format!("not {proposition}")
                        } else {
                            entry_text(&criteria.false_value)
                        },
                    )
                })
                .unwrap_or_else(|| (proposition.clone(), format!("not {proposition}")));
            let candidate_ids = vec![
                "true".to_string(),
                "false".to_string(),
                ABSTENTION_ID.to_string(),
            ];
            let candidate_descriptions = vec![
                format!("true: {true_description}"),
                format!("false: {false_description}"),
                ABSTENTION_DESCRIPTION.to_string(),
            ];
            let text = format!("Context:\n{state}\n\nEvaluate proposition: {proposition}");
            Ok(PreparedQuestion {
                id: id.to_string(),
                kind: PreparedQuestionKind::Noul,
                prompt: format_prompt(&text, &candidate_descriptions),
                candidate_ids,
                candidate_descriptions,
            })
        }
    }
}

fn build_prompt(question: &str, state: &str, labels: &[String]) -> String {
    format_prompt(
        &format!("Question: {question}\n\nContext:\n{state}"),
        labels,
    )
}

fn format_prompt(text: &str, labels: &[String]) -> String {
    let prefix = labels
        .iter()
        .map(|label| format!("<<LABEL>>{label}"))
        .collect::<String>();
    format!("{prefix}<<SEP>>{text}")
}

fn validate_entry(value: &Value, path: &str) -> Result<(), AppError> {
    if matches!(
        value,
        Value::String(_) | Value::Array(_) | Value::Object(_) | Value::Null
    ) {
        Ok(())
    } else {
        Err(AppError::Unprocessable(format!(
            "{path} must be a string, object, array, or null"
        )))
    }
}

pub fn entry_text(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => String::new(),
        value => serde_json::to_string(value).unwrap_or_default(),
    }
}

pub fn calibrated_probabilities(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let temperature = if temperature.is_finite() && temperature > 0.0 {
        temperature
    } else {
        1.0
    };
    let max = logits
        .iter()
        .map(|value| value / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let exponents: Vec<f32> = logits
        .iter()
        .map(|value| (value / temperature - max).exp())
        .collect();
    let sum = exponents.iter().sum::<f32>();
    exponents.into_iter().map(|value| value / sum).collect()
}

pub fn visible_probabilities(ids: &[String], probabilities: &[f32]) -> BTreeMap<String, f32> {
    let visible_mass = ids
        .iter()
        .zip(probabilities)
        .filter(|(id, _)| id.as_str() != ABSTENTION_ID)
        .map(|(_, probability)| probability)
        .sum::<f32>();

    ids.iter()
        .zip(probabilities)
        .filter(|(id, _)| id.as_str() != ABSTENTION_ID)
        .map(|(id, probability)| {
            (
                id.clone(),
                if visible_mass > 0.0 {
                    probability / visible_mass
                } else {
                    0.0
                },
            )
        })
        .collect()
}

pub fn distribution_confidence(probabilities: &BTreeMap<String, f32>) -> f32 {
    let count = probabilities.len();
    if count <= 1 {
        return 1.0;
    }
    let peak = probabilities.values().copied().fold(0.0, f32::max);
    ((count as f32 * peak - 1.0) / (count as f32 - 1.0)).clamp(0.0, 1.0)
}

pub fn answer_question(
    question: &PreparedQuestion,
    logits: &[f32],
    temperature: f32,
) -> Result<SystemOneAnswer, AppError> {
    if logits.len() < question.candidate_ids.len() {
        return Err(AppError::Inference(format!(
            "System One output for '{}' has {} logits, expected at least {}",
            question.id,
            logits.len(),
            question.candidate_ids.len()
        )));
    }

    let calibrated = calibrated_probabilities(&logits[..question.candidate_ids.len()], temperature);
    let probabilities = visible_probabilities(&question.candidate_ids, &calibrated);

    match &question.kind {
        PreparedQuestionKind::Noul => {
            let true_probability = probabilities.get("true").copied().unwrap_or(0.5);
            Ok(SystemOneAnswer::Noul {
                noul: true_probability,
            })
        }
        PreparedQuestionKind::Choice => {
            let choice = probabilities
                .iter()
                .max_by(|left, right| {
                    left.1
                        .partial_cmp(right.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(id, _)| id.clone())
                .ok_or_else(|| {
                    AppError::Inference(format!(
                        "System One question '{}' produced no visible choices",
                        question.id
                    ))
                })?;
            let confidence = distribution_confidence(&probabilities);
            Ok(SystemOneAnswer::Choice {
                choice,
                confidence,
                probabilities,
            })
        }
        PreparedQuestionKind::Score { legend } => {
            let score = probabilities
                .iter()
                .filter_map(|(level, probability)| {
                    level.parse::<f32>().ok().map(|level| level * probability)
                })
                .sum();
            let confidence = distribution_confidence(&probabilities);
            Ok(SystemOneAnswer::Score {
                score,
                confidence,
                legend: legend.clone(),
                probabilities,
            })
        }
    }
}

// --- OpenJev "rl-agent" contract (e.g. Laya) -----------------------------------------------
//
// A distinct, bespoke ONNX contract from GLiClass: a ModernBERT-style encoder + decision head
// that scores per-option `[MASK]` marker positions instead of doing prompt-based zero-shot
// classification. There is no abstention candidate; every rendered option is scored directly.
// This mirrors `rl_common.py`'s `QTYPES`/`render_options`/`temp_bucket` reference implementation
// exactly, since the calibration temperatures were fitted against that precise algorithm.

/// Matches `rl_common.QTYPES = {"choice": 0, "score": 1, "noul": 2}`.
pub fn openjev_qtype_index(question: &SystemOneQuestion) -> usize {
    match question {
        SystemOneQuestion::Choice { .. } => 0,
        SystemOneQuestion::Score { .. } => 1,
        SystemOneQuestion::Noul { .. } => 2,
    }
}

const OPENJEV_QTYPE_NAMES: [&str; 3] = ["choice", "score", "noul"];

/// Matches `rl_common.render_options`: option display texts in marker order. Noul is always
/// `[false, true]` so `probabilities[1]` is directly the `noul` probability.
pub fn openjev_render_options(question: &SystemOneQuestion) -> Vec<String> {
    match question {
        SystemOneQuestion::Choice { criteria, .. } => criteria
            .iter()
            .map(|(key, value)| {
                let falsy = matches!(value, Value::Null)
                    || matches!(value, Value::String(s) if s.is_empty())
                    || matches!(value, Value::Bool(false));
                if falsy {
                    key.clone()
                } else {
                    format!("{key}: {}", entry_text(value))
                }
            })
            .collect(),
        SystemOneQuestion::Score { criteria, .. } => criteria
            .iter()
            .enumerate()
            .map(|(index, description)| format!("level {index}: {}", entry_text(description)))
            .collect(),
        SystemOneQuestion::Noul {
            instructions,
            criteria,
        } => {
            let proposition = entry_text(instructions);
            let (true_description, false_description) = criteria
                .as_ref()
                .map(|criteria| {
                    (
                        if criteria.true_value.is_null() {
                            proposition.clone()
                        } else {
                            entry_text(&criteria.true_value)
                        },
                        if criteria.false_value.is_null() {
                            "no, the statement does not hold".to_string()
                        } else {
                            entry_text(&criteria.false_value)
                        },
                    )
                })
                .unwrap_or_else(|| {
                    (
                        proposition.clone(),
                        "no, the statement does not hold".to_string(),
                    )
                });
            vec![
                format!("false: {false_description}"),
                format!("true: {true_description}"),
            ]
        }
    }
}

/// Candidate ids in the same order as `openjev_render_options`, used to key the `Choice`/`Score`
/// answer's `probabilities` map and to resolve the winning `Choice`.
pub fn openjev_option_ids(question: &SystemOneQuestion) -> Vec<String> {
    match question {
        SystemOneQuestion::Choice { criteria, .. } => criteria.keys().cloned().collect(),
        SystemOneQuestion::Score { criteria, .. } => {
            (0..criteria.len()).map(|index| index.to_string()).collect()
        }
        SystemOneQuestion::Noul { .. } => vec!["false".to_string(), "true".to_string()],
    }
}

/// The `"<type> question: <instructions>"` head text fed into `build_sequence`'s tokenizer call.
pub fn openjev_head_text(question: &SystemOneQuestion) -> String {
    let (kind, instructions) = match question {
        SystemOneQuestion::Choice { instructions, .. } => ("choice", instructions),
        SystemOneQuestion::Score { instructions, .. } => ("score", instructions),
        SystemOneQuestion::Noul { instructions, .. } => ("noul", instructions),
    };
    format!("{kind} question: {}", entry_text(instructions))
}

/// Matches `rl_common.temp_bucket`: the per-cardinality calibration key, e.g. `"choice:3-5"`.
pub fn openjev_temp_bucket(qtype: usize, option_count: usize) -> String {
    let size = if option_count <= 2 {
        "2"
    } else if option_count <= 5 {
        "3-5"
    } else if option_count <= 10 {
        "6-10"
    } else {
        "11+"
    };
    format!("{}:{}", OPENJEV_QTYPE_NAMES[qtype.min(2)], size)
}

/// Matches `rl_common.confidence_from_probs`: `1 - normalized_entropy`, over the visible
/// (non-padded) option probabilities only.
pub fn openjev_confidence(probabilities: &[f32]) -> f32 {
    let k = probabilities.len();
    if k < 2 {
        return 1.0;
    }
    let entropy: f32 = -probabilities
        .iter()
        .map(|p| p.max(1e-12) * p.max(1e-12).ln())
        .sum::<f32>();
    1.0 - entropy / (k as f32).ln()
}

/// Maps a question + its calibrated per-marker probabilities (already softmaxed, one entry per
/// rendered option, no abstention) back to the shared `SystemOneAnswer` wire format.
pub fn answer_question_openjev(
    id: &str,
    question: &SystemOneQuestion,
    option_ids: &[String],
    probabilities: &[f32],
) -> Result<SystemOneAnswer, AppError> {
    let confidence = openjev_confidence(probabilities);
    match question {
        SystemOneQuestion::Noul { .. } => {
            let noul = probabilities.get(1).copied().unwrap_or(0.5);
            Ok(SystemOneAnswer::Noul { noul })
        }
        SystemOneQuestion::Choice { .. } => {
            let (best_index, _) = probabilities
                .iter()
                .enumerate()
                .max_by(|left, right| {
                    left.1
                        .partial_cmp(right.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .ok_or_else(|| {
                    AppError::Inference(format!(
                        "System One question '{id}' produced no OpenJev choices"
                    ))
                })?;
            let choice = option_ids
                .get(best_index)
                .cloned()
                .unwrap_or_else(|| best_index.to_string());
            let probabilities = option_ids
                .iter()
                .cloned()
                .zip(probabilities.iter().copied())
                .collect();
            Ok(SystemOneAnswer::Choice {
                choice,
                confidence,
                probabilities,
            })
        }
        SystemOneQuestion::Score { criteria, .. } => {
            let score: f32 = probabilities
                .iter()
                .enumerate()
                .map(|(index, probability)| index as f32 * probability)
                .sum();
            let legend = criteria
                .iter()
                .enumerate()
                .map(|(index, description)| (index.to_string(), description.clone()))
                .collect();
            let probabilities = option_ids
                .iter()
                .cloned()
                .zip(probabilities.iter().copied())
                .collect();
            Ok(SystemOneAnswer::Score {
                score,
                confidence,
                legend,
                probabilities,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn prepares_all_question_types() {
        let request: SystemOneRequest = serde_json::from_value(json!({
            "model": "openjev",
            "state": {"message": "charged twice"},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": "payments", "support": "technical help"}
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe?",
                    "criteria": ["low", "high"]
                },
                "urgent": {
                    "type": "noul",
                    "instructions": "Is this urgent?"
                }
            }
        }))
        .unwrap();

        let prepared = prepare_request(&request).unwrap();
        assert_eq!(prepared.len(), 3);
        assert!(prepared
            .iter()
            .all(|question| question.prompt.contains("<<SEP>>")));
        assert!(prepared
            .iter()
            .all(|question| question.candidate_ids.last().unwrap() == ABSTENTION_ID));
    }

    #[test]
    fn rejects_unsupported_choice_capacity() {
        let criteria = (0..25)
            .map(|index| (format!("option-{index}"), json!(null)))
            .collect::<serde_json::Map<_, _>>();
        let request: SystemOneRequest = serde_json::from_value(json!({
            "model": "openjev",
            "state": "state",
            "questions": {
                "choice": {
                    "type": "choice",
                    "instructions": "choose",
                    "criteria": criteria
                }
            }
        }))
        .unwrap();
        assert!(prepare_request(&request).is_err());
    }

    #[test]
    fn removes_abstention_and_normalizes_probabilities() {
        let ids = vec!["a".to_string(), "b".to_string(), ABSTENTION_ID.to_string()];
        let visible = visible_probabilities(&ids, &[0.4, 0.1, 0.5]);
        assert!((visible["a"] - 0.8).abs() < 1e-6);
        assert!((visible["b"] - 0.2).abs() < 1e-6);
        assert!((distribution_confidence(&visible) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn converts_logits_to_wire_answers() {
        let request: SystemOneRequest = serde_json::from_value(json!({
            "model": "openjev",
            "state": "state",
            "questions": {
                "score": {
                    "type": "score",
                    "instructions": "rate",
                    "criteria": ["low", "medium", "high"]
                }
            }
        }))
        .unwrap();
        let question = prepare_request(&request).unwrap().remove(0);
        let answer = answer_question(&question, &[0.0, 0.0, 10.0, -10.0], 1.0).unwrap();
        match answer {
            SystemOneAnswer::Score { score, .. } => assert!(score > 1.99),
            _ => panic!("expected score"),
        }
    }
}
