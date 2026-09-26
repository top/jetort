use crate::error::AppError;
use crate::protocol::kserve_v2::{KServeInputTensor, KServeOutputTensor};
use half::f16;
use ort::value::{DynValue, Tensor};
use serde_json::Value as JsonValue;

pub fn normalize_datatype(dt: &str) -> String {
    match dt.to_uppercase().as_str() {
        "INT64" | "I64" => "INT64".to_string(),
        "INT32" | "I32" => "INT32".to_string(),
        "FP16" | "FLOAT16" | "HALF" | "F16" => "FP16".to_string(),
        "FP32" | "FLOAT" | "FLOAT32" | "F32" => "FP32".to_string(),
        "BOOL" | "BOOLEAN" => "BOOL".to_string(),
        "UINT8" | "U8" => "UINT8".to_string(),
        other => other.to_string(),
    }
}

/// Automatically pads or truncates 2D sequence tensors (like input_ids, attention_mask)
/// to the target sequence length configured for the model.
pub fn maybe_pad_tensor(
    tensor: &KServeInputTensor,
    target_seq_len: Option<usize>,
) -> KServeInputTensor {
    let target = match target_seq_len {
        Some(t) if t > 0 => t,
        _ => return tensor.clone(),
    };

    // Pad sequence dimension for 2D inputs when name indicates sequence or dimension differs
    if tensor.shape.len() == 2
        && (tensor.name == "input_ids"
            || tensor.name == "attention_mask"
            || tensor.name == "token_type_ids")
    {
        let batch = tensor.shape[0];
        let current_len = tensor.shape[1];
        if current_len == target {
            return tensor.clone();
        }

        if let Some(arr) = tensor.data.as_array() {
            let mut new_arr = Vec::with_capacity(batch * target);
            for b in 0..batch {
                let start = b * current_len;
                let end = (start + current_len).min(arr.len());
                let row = if start < arr.len() {
                    &arr[start..end]
                } else {
                    &[]
                };

                // Copy up to target tokens
                for item in row.iter().take(target) {
                    new_arr.push(item.clone());
                }

                // Fill remainder with 0
                for _ in row.len()..target {
                    new_arr.push(serde_json::json!(0));
                }
            }

            return KServeInputTensor {
                name: tensor.name.clone(),
                shape: vec![batch, target],
                datatype: tensor.datatype.clone(),
                data: JsonValue::Array(new_arr),
            };
        }
    }

    tensor.clone()
}

pub fn json_to_ort_value(tensor: &KServeInputTensor) -> Result<DynValue, AppError> {
    let shape = &tensor.shape;
    let total_elements: usize = shape.iter().product();

    let json_array = tensor.data.as_array().ok_or_else(|| {
        AppError::InvalidInput(format!(
            "Tensor '{}' field 'data' must be a JSON array",
            tensor.name
        ))
    })?;

    if json_array.len() != total_elements {
        return Err(AppError::InvalidInput(format!(
            "Tensor '{}' data length ({}) does not match shape {:?} (expected {} elements)",
            tensor.name,
            json_array.len(),
            shape,
            total_elements
        )));
    }

    let dt = normalize_datatype(&tensor.datatype);

    match dt.as_str() {
        "INT64" => {
            let mut vec = Vec::with_capacity(total_elements);
            for (idx, val) in json_array.iter().enumerate() {
                let v = val.as_i64().ok_or_else(|| {
                    AppError::InvalidInput(format!(
                        "Element at index {} in tensor '{}' is not an integer",
                        idx, tensor.name
                    ))
                })?;
                vec.push(v);
            }
            let val = Tensor::from_array((shape.clone(), vec.into_boxed_slice()))
                .map_err(|e| AppError::Inference(format!("Failed to create ORT tensor: {}", e)))?;
            Ok(val.into_dyn())
        }
        "INT32" => {
            let mut vec = Vec::with_capacity(total_elements);
            for (idx, val) in json_array.iter().enumerate() {
                let v = val
                    .as_i64()
                    .and_then(|x| i32::try_from(x).ok())
                    .ok_or_else(|| {
                        AppError::InvalidInput(format!(
                            "Element at index {} in tensor '{}' is not a valid i32",
                            idx, tensor.name
                        ))
                    })?;
                vec.push(v);
            }
            let val = Tensor::from_array((shape.clone(), vec.into_boxed_slice()))
                .map_err(|e| AppError::Inference(format!("Failed to create ORT tensor: {}", e)))?;
            Ok(val.into_dyn())
        }
        "FP16" => {
            let mut vec = Vec::with_capacity(total_elements);
            for (idx, val) in json_array.iter().enumerate() {
                let v = val
                    .as_f64()
                    .map(|x| f16::from_f32(x as f32))
                    .ok_or_else(|| {
                        AppError::InvalidInput(format!(
                            "Element at index {} in tensor '{}' is not a float",
                            idx, tensor.name
                        ))
                    })?;
                vec.push(v);
            }
            let val = Tensor::from_array((shape.clone(), vec.into_boxed_slice()))
                .map_err(|e| AppError::Inference(format!("Failed to create ORT tensor: {}", e)))?;
            Ok(val.into_dyn())
        }
        "FP32" => {
            let mut vec = Vec::with_capacity(total_elements);
            for (idx, val) in json_array.iter().enumerate() {
                let v = val.as_f64().map(|x| x as f32).ok_or_else(|| {
                    AppError::InvalidInput(format!(
                        "Element at index {} in tensor '{}' is not a float",
                        idx, tensor.name
                    ))
                })?;
                vec.push(v);
            }
            let val = Tensor::from_array((shape.clone(), vec.into_boxed_slice()))
                .map_err(|e| AppError::Inference(format!("Failed to create ORT tensor: {}", e)))?;
            Ok(val.into_dyn())
        }
        "BOOL" => {
            let mut vec = Vec::with_capacity(total_elements);
            for (idx, val) in json_array.iter().enumerate() {
                let v = val.as_bool().ok_or_else(|| {
                    AppError::InvalidInput(format!(
                        "Element at index {} in tensor '{}' is not a boolean",
                        idx, tensor.name
                    ))
                })?;
                vec.push(v);
            }
            let val = Tensor::from_array((shape.clone(), vec.into_boxed_slice()))
                .map_err(|e| AppError::Inference(format!("Failed to create ORT tensor: {}", e)))?;
            Ok(val.into_dyn())
        }
        "UINT8" => {
            let mut vec = Vec::with_capacity(total_elements);
            for (idx, val) in json_array.iter().enumerate() {
                let v = val
                    .as_u64()
                    .and_then(|x| u8::try_from(x).ok())
                    .ok_or_else(|| {
                        AppError::InvalidInput(format!(
                            "Element at index {} in tensor '{}' is not a valid u8",
                            idx, tensor.name
                        ))
                    })?;
                vec.push(v);
            }
            let val = Tensor::from_array((shape.clone(), vec.into_boxed_slice()))
                .map_err(|e| AppError::Inference(format!("Failed to create ORT tensor: {}", e)))?;
            Ok(val.into_dyn())
        }
        unsupported => Err(AppError::InvalidInput(format!(
            "Unsupported datatype '{}' for input tensor '{}'",
            unsupported, tensor.name
        ))),
    }
}

pub fn ort_value_to_output(name: &str, value: &DynValue) -> Result<KServeOutputTensor, AppError> {
    if let Ok((shape, data)) = value.try_extract_tensor::<f32>() {
        let shape_vec = shape.iter().map(|&x| x as usize).collect();
        let json_data = JsonValue::Array(data.iter().map(|&x| JsonValue::from(x)).collect());
        return Ok(KServeOutputTensor {
            name: name.to_string(),
            shape: shape_vec,
            datatype: "FP32".to_string(),
            data: json_data,
        });
    }

    if let Ok((shape, data)) = value.try_extract_tensor::<f16>() {
        let shape_vec = shape.iter().map(|&x| x as usize).collect();
        let json_data =
            JsonValue::Array(data.iter().map(|&x| JsonValue::from(x.to_f32())).collect());
        return Ok(KServeOutputTensor {
            name: name.to_string(),
            shape: shape_vec,
            datatype: "FP16".to_string(),
            data: json_data,
        });
    }

    if let Ok((shape, data)) = value.try_extract_tensor::<i64>() {
        let shape_vec = shape.iter().map(|&x| x as usize).collect();
        let json_data = JsonValue::Array(data.iter().map(|&x| JsonValue::from(x)).collect());
        return Ok(KServeOutputTensor {
            name: name.to_string(),
            shape: shape_vec,
            datatype: "INT64".to_string(),
            data: json_data,
        });
    }

    if let Ok((shape, data)) = value.try_extract_tensor::<i32>() {
        let shape_vec = shape.iter().map(|&x| x as usize).collect();
        let json_data = JsonValue::Array(data.iter().map(|&x| JsonValue::from(x)).collect());
        return Ok(KServeOutputTensor {
            name: name.to_string(),
            shape: shape_vec,
            datatype: "INT32".to_string(),
            data: json_data,
        });
    }

    if let Ok((shape, data)) = value.try_extract_tensor::<bool>() {
        let shape_vec = shape.iter().map(|&x| x as usize).collect();
        let json_data = JsonValue::Array(data.iter().map(|&x| JsonValue::from(x)).collect());
        return Ok(KServeOutputTensor {
            name: name.to_string(),
            shape: shape_vec,
            datatype: "BOOL".to_string(),
            data: json_data,
        });
    }

    if let Ok((shape, data)) = value.try_extract_tensor::<u8>() {
        let shape_vec = shape.iter().map(|&x| x as usize).collect();
        let json_data = JsonValue::Array(data.iter().map(|&x| JsonValue::from(x)).collect());
        return Ok(KServeOutputTensor {
            name: name.to_string(),
            shape: shape_vec,
            datatype: "UINT8".to_string(),
            data: json_data,
        });
    }

    Err(AppError::Inference(format!(
        "Failed to extract tensor data for output '{}': unsupported or dynamic output type",
        name
    )))
}
