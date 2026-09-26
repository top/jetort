use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OnnxDim {
    Fixed(i64),
    Dynamic(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OnnxInputTensorInfo {
    pub name: String,
    pub elem_type: i32,
    pub dims: Vec<OnnxDim>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrtDynamicProfiles {
    pub min_shapes: String,
    pub opt_shapes: String,
    pub max_shapes: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationKind {
    None,              // Standard unquantized FP32 / FP16
    CpuDynamicInt8,    // MatMulInteger / DynamicQuantizeLinear (Dynamic quantization)
    StaticQLinearInt8, // QLinearMatMul / QLinearConv (Static INT8)
    ExplicitQdqInt8,   // QuantizeLinear + DequantizeLinear (TensorRT QDQ)
    Fp16,              // Native Float16
}

impl std::fmt::Display for QuantizationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantizationKind::None => write!(f, "Standard (FP32/FP16 unquantized)"),
            QuantizationKind::CpuDynamicInt8 => {
                write!(f, "Dynamic INT8 (MatMulInteger/DynamicQuantizeLinear)")
            }
            QuantizationKind::StaticQLinearInt8 => {
                write!(f, "Static INT8 (QLinearMatMul/QLinearConv)")
            }
            QuantizationKind::ExplicitQdqInt8 => {
                write!(f, "Explicit QDQ INT8 (Tensor Core accelerated)")
            }
            QuantizationKind::Fp16 => write!(f, "Native FP16 (Float16 tensors)"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OnnxModelInspection {
    pub ir_version: Option<i64>,
    pub opset_version: Option<i64>,
    pub producer_name: Option<String>,
    pub producer_version: Option<String>,
    pub metadata_props: HashMap<String, String>,
    pub inputs: Vec<OnnxInputTensorInfo>,
    pub outputs: Vec<OnnxInputTensorInfo>,
    pub op_types: HashSet<String>,
    pub quantization_kind: QuantizationKind,
    pub has_uppercase_einsum: bool,
    pub is_audio_model: bool,
    pub recommended_provider: String,
    pub candidate_providers: Vec<String>,
    pub trt_compilation_feasible: bool,
    pub trt_skip_reason: Option<String>,
    pub synthesized_profiles: Option<TrtDynamicProfiles>,
}

impl OnnxModelInspection {
    pub fn print_inspection_banner(&self, model_name: &str) {
        let opset = self
            .opset_version
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let producer = self.producer_name.as_deref().unwrap_or("unknown");
        let inputs_desc: Vec<String> = self
            .inputs
            .iter()
            .map(|i| {
                let dims_str: Vec<String> = i
                    .dims
                    .iter()
                    .map(|d| match d {
                        OnnxDim::Fixed(v) => v.to_string(),
                        OnnxDim::Dynamic(s) => {
                            if s.is_empty() {
                                "Dynamic".to_string()
                            } else {
                                s.clone()
                            }
                        }
                    })
                    .collect();
                format!("{}[{}]", i.name, dims_str.join("x"))
            })
            .collect();

        tracing::info!(
            model = %model_name,
            opset = %opset,
            producer = %producer,
            quantization = %self.quantization_kind,
            recommended = %self.recommended_provider,
            candidates = ?self.candidate_providers,
            "📋 [Model DNA Inspection] Opset: {} | Producer: {} | Quant: {} | Inputs: {:?} | Recommended Runtime: '{}' (Candidates: {:?})",
            opset, producer, self.quantization_kind, inputs_desc, self.recommended_provider, self.candidate_providers
        );

        if let Some(ref reason) = self.trt_skip_reason {
            tracing::info!(
                model = %model_name,
                reason = %reason,
                "🛡️ [Dispatch Decision] TensorRT pre-flight: {} -> Direct routing to '{}'",
                reason, self.recommended_provider
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming Zero-Copy ONNX Protobuf Wire-Format Parser
// ---------------------------------------------------------------------------

fn read_varint<R: Read>(reader: &mut R) -> std::io::Result<u64> {
    let mut res = 0u64;
    let mut shift = 0;
    loop {
        let mut b = [0u8; 1];
        reader.read_exact(&mut b)?;
        let byte = b[0];
        res |= ((byte & 0x7F) as u64) << shift;
        if (byte & 0x80) == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Protobuf varint overflow",
            ));
        }
    }
    Ok(res)
}

fn decode_varint(buf: &[u8], offset: &mut usize) -> Option<u64> {
    let mut res = 0u64;
    let mut shift = 0;
    while *offset < buf.len() {
        let byte = buf[*offset];
        *offset += 1;
        res |= ((byte & 0x7F) as u64) << shift;
        if (byte & 0x80) == 0 {
            return Some(res);
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
    None
}

fn skip_field_in_file(file: &mut File, wire_type: u64) -> std::io::Result<()> {
    match wire_type {
        0 => {
            // Varint
            let _ = read_varint(file)?;
        }
        1 => {
            // 64-bit
            file.seek(SeekFrom::Current(8))?;
        }
        2 => {
            // Length-delimited: read len and seek forward
            let len = read_varint(file)?;
            file.seek(SeekFrom::Current(len as i64))?;
        }
        5 => {
            // 32-bit
            file.seek(SeekFrom::Current(4))?;
        }
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported protobuf wire type: {}", wire_type),
            ));
        }
    }
    Ok(())
}

fn skip_field_in_buffer(buf: &[u8], offset: &mut usize, wire_type: u64) -> bool {
    match wire_type {
        0 => decode_varint(buf, offset).is_some(),
        1 => {
            if *offset + 8 <= buf.len() {
                *offset += 8;
                true
            } else {
                false
            }
        }
        2 => {
            if let Some(len) = decode_varint(buf, offset) {
                let len = len as usize;
                if *offset + len <= buf.len() {
                    *offset += len;
                    true
                } else {
                    false
                }
            } else {
                false
            }
        }
        5 if *offset + 4 <= buf.len() => {
            *offset += 4;
            true
        }
        _ => false,
    }
}

/// Parses an ONNX `ValueInfoProto` byte slice into `OnnxInputTensorInfo`.
/// ValueInfoProto schema:
/// - field 1: string name
/// - field 2: TypeProto type
///   - field 1: Tensor tensor_type
///     - field 1: int32 elem_type (varint)
///     - field 2: TensorShapeProto shape
///       - field 1: repeated Dimension dim
///         - field 1: int64 dim_value
///         - field 2: string dim_param
pub fn parse_value_info_bytes(buf: &[u8]) -> Option<OnnxInputTensorInfo> {
    let mut offset = 0;
    let mut name = String::new();
    let mut type_bytes: Option<&[u8]> = None;

    while offset < buf.len() {
        let tag = decode_varint(buf, &mut offset)?;
        let field_num = tag >> 3;
        let wire_type = tag & 0x7;

        match (field_num, wire_type) {
            (1, 2) => {
                let len = decode_varint(buf, &mut offset)? as usize;
                if offset + len > buf.len() {
                    return None;
                }
                name = String::from_utf8_lossy(&buf[offset..offset + len]).to_string();
                offset += len;
            }
            (2, 2) => {
                let len = decode_varint(buf, &mut offset)? as usize;
                if offset + len > buf.len() {
                    return None;
                }
                type_bytes = Some(&buf[offset..offset + len]);
                offset += len;
            }
            _ => {
                if !skip_field_in_buffer(buf, &mut offset, wire_type) {
                    return None;
                }
            }
        }
    }

    if name.is_empty() {
        return None;
    }

    let mut elem_type = 0;
    let mut dims = Vec::new();

    if let Some(t_buf) = type_bytes {
        let mut t_offset = 0;
        let mut tensor_type_bytes: Option<&[u8]> = None;

        while t_offset < t_buf.len() {
            let tag = decode_varint(t_buf, &mut t_offset)?;
            let field_num = tag >> 3;
            let wire_type = tag & 0x7;
            match (field_num, wire_type) {
                (1, 2) => {
                    let len = decode_varint(t_buf, &mut t_offset)? as usize;
                    if t_offset + len > t_buf.len() {
                        return None;
                    }
                    tensor_type_bytes = Some(&t_buf[t_offset..t_offset + len]);
                    t_offset += len;
                }
                _ => {
                    if !skip_field_in_buffer(t_buf, &mut t_offset, wire_type) {
                        return None;
                    }
                }
            }
        }

        if let Some(tt_buf) = tensor_type_bytes {
            let mut tt_offset = 0;
            let mut shape_bytes: Option<&[u8]> = None;

            while tt_offset < tt_buf.len() {
                let tag = decode_varint(tt_buf, &mut tt_offset)?;
                let field_num = tag >> 3;
                let wire_type = tag & 0x7;
                match (field_num, wire_type) {
                    (1, 0) => {
                        elem_type = decode_varint(tt_buf, &mut tt_offset)? as i32;
                    }
                    (2, 2) => {
                        let len = decode_varint(tt_buf, &mut tt_offset)? as usize;
                        if tt_offset + len > tt_buf.len() {
                            return None;
                        }
                        shape_bytes = Some(&tt_buf[tt_offset..tt_offset + len]);
                        tt_offset += len;
                    }
                    _ => {
                        if !skip_field_in_buffer(tt_buf, &mut tt_offset, wire_type) {
                            return None;
                        }
                    }
                }
            }

            if let Some(s_buf) = shape_bytes {
                let mut s_offset = 0;
                while s_offset < s_buf.len() {
                    let tag = decode_varint(s_buf, &mut s_offset)?;
                    let field_num = tag >> 3;
                    let wire_type = tag & 0x7;
                    if field_num == 1 && wire_type == 2 {
                        // Dimension
                        let len = decode_varint(s_buf, &mut s_offset)? as usize;
                        if s_offset + len > s_buf.len() {
                            return None;
                        }
                        let d_buf = &s_buf[s_offset..s_offset + len];
                        s_offset += len;

                        let mut d_offset = 0;
                        let mut dim_opt: Option<OnnxDim> = None;

                        while d_offset < d_buf.len() {
                            let d_tag = decode_varint(d_buf, &mut d_offset)?;
                            let d_field = d_tag >> 3;
                            let d_wire = d_tag & 0x7;
                            match (d_field, d_wire) {
                                (1, 0) => {
                                    let val = decode_varint(d_buf, &mut d_offset)? as i64;
                                    dim_opt = Some(OnnxDim::Fixed(val));
                                }
                                (2, 2) => {
                                    let d_len = decode_varint(d_buf, &mut d_offset)? as usize;
                                    if d_offset + d_len <= d_buf.len() {
                                        let p = String::from_utf8_lossy(
                                            &d_buf[d_offset..d_offset + d_len],
                                        )
                                        .to_string();
                                        dim_opt = Some(OnnxDim::Dynamic(p));
                                        d_offset += d_len;
                                    }
                                }
                                _ => {
                                    if !skip_field_in_buffer(d_buf, &mut d_offset, d_wire) {
                                        break;
                                    }
                                }
                            }
                        }
                        dims.push(dim_opt.unwrap_or_else(|| OnnxDim::Dynamic("".to_string())));
                    } else {
                        if !skip_field_in_buffer(s_buf, &mut s_offset, wire_type) {
                            return None;
                        }
                    }
                }
            }
        }
    }

    Some(OnnxInputTensorInfo {
        name,
        elem_type,
        dims,
    })
}

fn parse_string_string_entry(buf: &[u8]) -> Option<(String, String)> {
    let mut offset = 0;
    let mut key = String::new();
    let mut val = String::new();
    while offset < buf.len() {
        let tag = decode_varint(buf, &mut offset)?;
        let field_num = tag >> 3;
        let wire_type = tag & 0x7;
        match (field_num, wire_type) {
            (1, 2) => {
                let len = decode_varint(buf, &mut offset)? as usize;
                if offset + len <= buf.len() {
                    key = String::from_utf8_lossy(&buf[offset..offset + len]).to_string();
                    offset += len;
                } else {
                    return None;
                }
            }
            (2, 2) => {
                let len = decode_varint(buf, &mut offset)? as usize;
                if offset + len <= buf.len() {
                    val = String::from_utf8_lossy(&buf[offset..offset + len]).to_string();
                    offset += len;
                } else {
                    return None;
                }
            }
            _ => {
                if !skip_field_in_buffer(buf, &mut offset, wire_type) {
                    return None;
                }
            }
        }
    }
    if !key.is_empty() {
        Some((key, val))
    } else {
        None
    }
}

fn parse_opset_import(buf: &[u8]) -> Option<(String, i64)> {
    let mut offset = 0;
    let mut domain = String::new();
    let mut version = 0i64;
    while offset < buf.len() {
        let tag = decode_varint(buf, &mut offset)?;
        let field_num = tag >> 3;
        let wire_type = tag & 0x7;
        match (field_num, wire_type) {
            (1, 2) => {
                let len = decode_varint(buf, &mut offset)? as usize;
                if offset + len <= buf.len() {
                    domain = String::from_utf8_lossy(&buf[offset..offset + len]).to_string();
                    offset += len;
                } else {
                    return None;
                }
            }
            (2, 0) => {
                version = decode_varint(buf, &mut offset)? as i64;
            }
            _ => {
                if !skip_field_in_buffer(buf, &mut offset, wire_type) {
                    return None;
                }
            }
        }
    }
    Some((domain, version))
}

fn parse_node_summary(buf: &[u8]) -> (Option<String>, bool) {
    let mut offset = 0;
    let mut op_type = None;
    let mut has_uppercase_einsum = false;

    while offset < buf.len() {
        let tag = match decode_varint(buf, &mut offset) {
            Some(t) => t,
            None => break,
        };
        let field_num = tag >> 3;
        let wire_type = tag & 0x7;
        match (field_num, wire_type) {
            (4, 2) => {
                // op_type
                if let Some(len) = decode_varint(buf, &mut offset) {
                    let len = len as usize;
                    if offset + len <= buf.len() {
                        let op = String::from_utf8_lossy(&buf[offset..offset + len]).to_string();
                        offset += len;
                        op_type = Some(op);
                    }
                }
            }
            _ => {
                if !skip_field_in_buffer(buf, &mut offset, wire_type) {
                    break;
                }
            }
        }
    }

    if op_type.as_deref() == Some("Einsum")
        && buf.windows(b"->".len()).any(|w| w == b"->")
        && buf.windows(b"equation".len()).any(|w| w == b"equation")
    {
        has_uppercase_einsum = buf.iter().any(|&b| b.is_ascii_uppercase());
    }

    (op_type, has_uppercase_einsum)
}

/// Comprehensive zero-copy streaming ONNX inspection that extracts model metadata,
/// input/output shapes, operator profiles, and quantization types upfront.
pub fn inspect_onnx_model(
    path: &Path,
    config: &crate::config::ModelConfig,
) -> std::io::Result<OnnxModelInspection> {
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();

    let mut ir_version = None;
    let mut opset_version = None;
    let mut producer_name = None;
    let mut producer_version = None;
    let mut metadata_props = HashMap::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut op_types = HashSet::new();
    let mut has_uppercase_einsum = false;

    // Stream ModelProto
    loop {
        let tag = match read_varint(&mut file) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        };

        let field_num = tag >> 3;
        let wire_type = tag & 0x7;

        match (field_num, wire_type) {
            (1, 0) => {
                ir_version = Some(read_varint(&mut file)? as i64);
            }
            (2, 2) => {
                let len = read_varint(&mut file)? as usize;
                if len < 4096 {
                    let mut buf = vec![0u8; len];
                    file.read_exact(&mut buf)?;
                    producer_name = Some(String::from_utf8_lossy(&buf).to_string());
                } else {
                    file.seek(SeekFrom::Current(len as i64))?;
                }
            }
            (3, 2) => {
                let len = read_varint(&mut file)? as usize;
                if len < 4096 {
                    let mut buf = vec![0u8; len];
                    file.read_exact(&mut buf)?;
                    producer_version = Some(String::from_utf8_lossy(&buf).to_string());
                } else {
                    file.seek(SeekFrom::Current(len as i64))?;
                }
            }
            (8, 2) => {
                // opset_import: repeated OperatorSetIdProto
                let len = read_varint(&mut file)? as usize;
                if len < 65536 {
                    let mut buf = vec![0u8; len];
                    file.read_exact(&mut buf)?;
                    if let Some((_, ver)) = parse_opset_import(&buf) {
                        if opset_version.is_none() || ver > opset_version.unwrap_or(0) {
                            opset_version = Some(ver);
                        }
                    }
                } else {
                    file.seek(SeekFrom::Current(len as i64))?;
                }
            }
            (14, 2) => {
                let len = read_varint(&mut file)? as usize;
                if len < 65536 {
                    let mut buf = vec![0u8; len];
                    file.read_exact(&mut buf)?;
                    if let Some((k, v)) = parse_string_string_entry(&buf) {
                        metadata_props.insert(k, v);
                    }
                } else {
                    file.seek(SeekFrom::Current(len as i64))?;
                }
            }
            (7, 2) => {
                // Found GraphProto (field 7 in ONNX specification)
                let graph_len = read_varint(&mut file)?;
                let current_pos = file.stream_position()?;
                let graph_end = current_pos + graph_len;

                while file.stream_position()? < graph_end {
                    let g_tag = match read_varint(&mut file) {
                        Ok(t) => t,
                        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                        Err(e) => return Err(e),
                    };
                    let g_field = g_tag >> 3;
                    let g_wire = g_tag & 0x7;

                    match g_field {
                        1 if g_wire == 2 => {
                            // GraphProto.node: NodeProto
                            let node_len = read_varint(&mut file)?;
                            if node_len < 65536 {
                                let mut buf = vec![0u8; node_len as usize];
                                file.read_exact(&mut buf)?;
                                let (op, upper_einsum) = parse_node_summary(&buf);
                                if let Some(op) = op {
                                    op_types.insert(op);
                                }
                                if upper_einsum {
                                    has_uppercase_einsum = true;
                                }
                            } else {
                                file.seek(SeekFrom::Current(node_len as i64))?;
                            }
                        }
                        11 if g_wire == 2 => {
                            // GraphProto.input: ValueInfoProto
                            let input_len = read_varint(&mut file)?;
                            if input_len < 65536 {
                                let mut buf = vec![0u8; input_len as usize];
                                file.read_exact(&mut buf)?;
                                if let Some(info) = parse_value_info_bytes(&buf) {
                                    inputs.push(info);
                                }
                            } else {
                                file.seek(SeekFrom::Current(input_len as i64))?;
                            }
                        }
                        12 if g_wire == 2 => {
                            // GraphProto.output: ValueInfoProto
                            let output_len = read_varint(&mut file)?;
                            if output_len < 65536 {
                                let mut buf = vec![0u8; output_len as usize];
                                file.read_exact(&mut buf)?;
                                if let Some(info) = parse_value_info_bytes(&buf) {
                                    outputs.push(info);
                                }
                            } else {
                                file.seek(SeekFrom::Current(output_len as i64))?;
                            }
                        }
                        _ => {
                            // Skip initializers (field 5 weights) and other graph elements
                            skip_field_in_file(&mut file, g_wire)?;
                        }
                    }
                }
            }
            _ => {
                skip_field_in_file(&mut file, wire_type)?;
            }
        }
    }

    let is_dynamic_int8 =
        op_types.contains("MatMulInteger") || op_types.contains("DynamicQuantizeLinear");

    let is_static_qlinear = !is_dynamic_int8
        && (op_types.contains("QLinearMatMul") || op_types.contains("QLinearConv"));

    let is_explicit_qdq = !is_dynamic_int8
        && !is_static_qlinear
        && op_types.contains("QuantizeLinear")
        && op_types.contains("DequantizeLinear");

    let is_fp16 = !is_dynamic_int8
        && !is_static_qlinear
        && !is_explicit_qdq
        && (inputs.iter().any(|i| i.elem_type == 10)
            || outputs.iter().any(|o| o.elem_type == 10)
            || path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.contains("fp16") || s.contains("half"))
                .unwrap_or(false));

    let quantization_kind = if is_explicit_qdq {
        QuantizationKind::ExplicitQdqInt8
    } else if is_static_qlinear {
        QuantizationKind::StaticQLinearInt8
    } else if is_dynamic_int8 {
        QuantizationKind::CpuDynamicInt8
    } else if is_fp16 {
        QuantizationKind::Fp16
    } else {
        QuantizationKind::None
    };

    let is_audio_model = inputs.iter().any(|inp| {
        (inp.dims.len() == 3
            && match inp.dims.last() {
                Some(OnnxDim::Fixed(c)) => *c == 80 || *c == 128 || *c == 64,
                _ => false,
            })
            || inp.name == "features"
            || inp.name == "speech"
            || inp.name == "waveform"
    });

    let len = config.pad_seq_len.unwrap_or(512);
    let max_batch = config.max_batch_size.clamp(1, 8);
    let synthesized_profiles = synthesize_trt_profiles(&inputs, max_batch, len);

    let (trt_compilation_feasible, trt_skip_reason) = {
        let (feasible, reason) =
            crate::engine::provider::is_trt_compilation_feasible(config, Some(path), file_len);
        if feasible {
            (true, None)
        } else {
            (false, Some(reason))
        }
    };

    let candidate_providers = crate::engine::provider::get_candidate_providers(config);

    let recommended_provider = {
        #[cfg(target_os = "macos")]
        {
            if candidate_providers
                .iter()
                .any(|p| p.eq_ignore_ascii_case("coreml"))
            {
                "CoreML".to_string()
            } else {
                "CPU".to_string()
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            if candidate_providers
                .iter()
                .any(|p| p.eq_ignore_ascii_case("tensorrt") || p.eq_ignore_ascii_case("trt"))
                && trt_compilation_feasible
            {
                "TensorRT".to_string()
            } else if candidate_providers
                .iter()
                .any(|p| p.eq_ignore_ascii_case("cuda"))
            {
                "CUDA".to_string()
            } else if candidate_providers
                .iter()
                .any(|p| p.eq_ignore_ascii_case("directml"))
            {
                "DirectML".to_string()
            } else {
                "CPU".to_string()
            }
        }
    };

    Ok(OnnxModelInspection {
        ir_version,
        opset_version,
        producer_name,
        producer_version,
        metadata_props,
        inputs,
        outputs,
        op_types,
        quantization_kind,
        has_uppercase_einsum,
        is_audio_model,
        recommended_provider,
        candidate_providers,
        trt_compilation_feasible,
        trt_skip_reason,
        synthesized_profiles,
    })
}

/// Streams and seeks an ONNX file on disk to extract input tensor metadata.
pub fn introspect_onnx_inputs(path: &Path) -> std::io::Result<Vec<OnnxInputTensorInfo>> {
    let dummy_config = crate::config::ModelConfig {
        name: "anonymous".to_string(),
        path: None,
        hf_repo: None,
        hf_file: None,
        alias: None,
        load_on_startup: true,
        tokenizer_path: None,
        providers: vec!["auto".to_string()],
        pad_seq_len: None,
        max_batch_size: 1,
        trt_cache_dir: None,
        trt_profile_min_shapes: None,
        trt_profile_opt_shapes: None,
        trt_profile_max_shapes: None,
        intra_threads: 1,
        inter_threads: 1,
        cuda_memory_limit_mb: None,
        trt_max_workspace_mb: None,
        enable_mem_pattern: false,
        context_memory_sharing: true,
        concurrency_limit: None,
        session_pool_size: 1,
        config_dir: None,
        n_ctx: None,
        n_batch: 512,
        temperature: None,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };
    let inspection = inspect_onnx_model(path, &dummy_config)?;
    Ok(inspection.inputs)
}

// ---------------------------------------------------------------------------
// Semantic TensorRT Dynamic Profile Synthesizer
// ---------------------------------------------------------------------------

/// Intelligently synthesizes TensorRT optimization profile shapes based on
/// tensor geometric rank, dimensionality characteristics, and domain heuristics.
///
/// Handles:
/// - Rank 0 / scalar: 1x1x...
/// - Rank 1: scalars/lengths (e.g. features_lens, qtype, language, textnorm)
/// - Rank 2: sequence models (NLP token sequences: input_ids, attention_mask, etc.)
/// - Rank 3: time-series & acoustic features (e.g. Mel-filterbank [batch, frames, 80])
/// - Rank 4: computer vision tensors ([batch, channels, height, width])
pub fn synthesize_trt_profiles(
    inputs: &[OnnxInputTensorInfo],
    max_batch: usize,
    default_seq_len: usize,
) -> Option<TrtDynamicProfiles> {
    // Detect if model is an acoustic / speech model (Mel-filterbanks or waveform inputs)
    let is_audio_model = inputs.iter().any(|inp| {
        (inp.dims.len() == 3
            && match inp.dims.last() {
                Some(OnnxDim::Fixed(c)) => *c == 80 || *c == 128 || *c == 64,
                _ => false,
            })
            || inp.name == "features"
            || inp.name == "speech"
            || inp.name == "waveform"
    });

    // Audio speech models with long sequence lengths (e.g. 3000 frames = 30s) have 50+ self-attention layers.
    // Setting max batch > 1 on edge UMA hardware leads to gigabytes of scratch memory allocated per tactic.
    // For audio models, inference is serialized/single-stream; bounding batch to 1 prevents memory exhaustion.
    let batch = if is_audio_model {
        1
    } else {
        max_batch.clamp(1, 16)
    };
    let seq_len = default_seq_len.clamp(64, 4096);

    let mut min_parts = Vec::new();
    let mut opt_parts = Vec::new();
    let mut max_parts = Vec::new();

    for inp in inputs {
        let name = &inp.name;
        let rank = inp.dims.len();

        match rank {
            0 => {
                // Scalar: 1
                min_parts.push(format!("{name}:1"));
                opt_parts.push(format!("{name}:1"));
                max_parts.push(format!("{name}:1"));
            }
            1 => {
                // 1D tensor: e.g. features_lens, labels, qtype, language, textnorm
                match &inp.dims[0] {
                    OnnxDim::Fixed(1) => {
                        min_parts.push(format!("{name}:1"));
                        opt_parts.push(format!("{name}:1"));
                        max_parts.push(format!("{name}:1"));
                    }
                    OnnxDim::Fixed(len) => {
                        min_parts.push(format!("{name}:{len}"));
                        opt_parts.push(format!("{name}:{len}"));
                        max_parts.push(format!("{name}:{len}"));
                    }
                    OnnxDim::Dynamic(_) => {
                        min_parts.push(format!("{name}:1"));
                        opt_parts.push(format!("{name}:1"));
                        max_parts.push(format!("{name}:{batch}"));
                    }
                }
            }
            2 => {
                // 2D tensor: e.g. [batch, seq_len] in NLP, Embedding, Reranker
                let b_min = 1;
                let b_opt = 1;
                let b_max = match inp.dims[0] {
                    OnnxDim::Fixed(b) => b,
                    OnnxDim::Dynamic(_) => batch as i64,
                };

                let (s_min, s_opt, s_max) = match inp.dims[1] {
                    OnnxDim::Fixed(s) => (s, s, s),
                    OnnxDim::Dynamic(_) => (1, seq_len as i64, seq_len as i64),
                };

                min_parts.push(format!("{name}:{b_min}x{s_min}"));
                opt_parts.push(format!("{name}:{b_opt}x{s_opt}"));
                max_parts.push(format!("{name}:{b_max}x{s_max}"));
            }
            3 => {
                // 3D tensor: e.g. Audio Mel-features [batch, frames, channels=80]
                let b_min = 1;
                let b_opt = 1;
                let b_max = match inp.dims[0] {
                    OnnxDim::Fixed(b) => b,
                    OnnxDim::Dynamic(_) => batch as i64,
                };

                // Inspect dim 2: if fixed 80 or 128 (standard Log-Mel Fbank channels)
                let is_fbank_audio = match inp.dims[2] {
                    OnnxDim::Fixed(c) => c == 80 || c == 128 || c == 64,
                    _ => false,
                };

                if is_fbank_audio {
                    // Frame rates: 1 frame = 10ms. 10 frames = 100ms, 1000 frames = 10s, 3000 frames = 30s.
                    let c = match inp.dims[2] {
                        OnnxDim::Fixed(c) => c,
                        _ => 80,
                    };
                    min_parts.push(format!("{name}:{b_min}x10x{c}"));
                    opt_parts.push(format!("{name}:{b_opt}x1000x{c}"));
                    max_parts.push(format!("{name}:{b_max}x3000x{c}"));
                } else {
                    // General 3D tensor
                    let (d1_min, d1_opt, d1_max) = match inp.dims[1] {
                        OnnxDim::Fixed(d) => (d, d, d),
                        OnnxDim::Dynamic(_) => (1, seq_len as i64, seq_len as i64),
                    };
                    let (d2_min, d2_opt, d2_max) = match inp.dims[2] {
                        OnnxDim::Fixed(d) => (d, d, d),
                        OnnxDim::Dynamic(_) => (1, 64, 256),
                    };
                    min_parts.push(format!("{name}:{b_min}x{d1_min}x{d2_min}"));
                    opt_parts.push(format!("{name}:{b_opt}x{d1_opt}x{d2_opt}"));
                    max_parts.push(format!("{name}:{b_max}x{d1_max}x{d2_max}"));
                }
            }
            4 => {
                // 4D tensor: e.g. Computer Vision [batch, channels, height, width]
                let b_min = 1;
                let b_opt = 1;
                let b_max = match inp.dims[0] {
                    OnnxDim::Fixed(b) => b,
                    OnnxDim::Dynamic(_) => batch as i64,
                };

                let c = match inp.dims[1] {
                    OnnxDim::Fixed(c) => c,
                    OnnxDim::Dynamic(_) => 3,
                };

                let (h_min, h_opt, h_max) = match inp.dims[2] {
                    OnnxDim::Fixed(h) => (h, h, h),
                    OnnxDim::Dynamic(_) => (128, 224, 512),
                };
                let (w_min, w_opt, w_max) = match inp.dims[3] {
                    OnnxDim::Fixed(w) => (w, w, w),
                    OnnxDim::Dynamic(_) => (128, 224, 512),
                };

                min_parts.push(format!("{name}:{b_min}x{c}x{h_min}x{w_min}"));
                opt_parts.push(format!("{name}:{b_opt}x{c}x{h_opt}x{w_opt}"));
                max_parts.push(format!("{name}:{b_max}x{c}x{h_max}x{w_max}"));
            }
            _ => {
                // High dimensional or arbitrary tensor: bound dynamics safely
                let mut min_str = format!("{name}:");
                let mut opt_str = format!("{name}:");
                let mut max_str = format!("{name}:");

                for (i, dim) in inp.dims.iter().enumerate() {
                    let sep = if i == 0 { "" } else { "x" };
                    match dim {
                        OnnxDim::Fixed(val) => {
                            min_str.push_str(&format!("{sep}{val}"));
                            opt_str.push_str(&format!("{sep}{val}"));
                            max_str.push_str(&format!("{sep}{val}"));
                        }
                        OnnxDim::Dynamic(_) => {
                            let max_d = if i == 0 { batch } else { 64 };
                            min_str.push_str(&format!("{sep}1"));
                            opt_str.push_str(&format!("{sep}16"));
                            max_str.push_str(&format!("{sep}{max_d}"));
                        }
                    }
                }
                min_parts.push(min_str);
                opt_parts.push(opt_str);
                max_parts.push(max_str);
            }
        }
    }

    Some(TrtDynamicProfiles {
        min_shapes: min_parts.join(","),
        opt_shapes: opt_parts.join(","),
        max_shapes: max_parts.join(","),
    })
}
