#![allow(non_camel_case_types, non_snake_case, dead_code)]

use crate::error::AppError;
use libloading::{Library, Symbol};
use std::ffi::{c_char, c_double, CStr};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use tracing::{debug, info, warn};

// Opaque C API type declarations
pub enum OgaResult {}
pub enum OgaModel {}
pub enum OgaGeneratorParams {}
pub enum OgaGenerator {}
pub enum OgaTokenizer {}
pub enum OgaTokenizerStream {}
pub enum OgaTensor {}

/// Mirrors `OgaElementType` from `ort_genai_c.h`. Only the variants jetort actually constructs
/// tensors for are given explicit values; the rest are declared for documentation/completeness.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OgaElementType {
    Undefined = 0,
    Float32 = 1,
    Uint8 = 2,
    Int8 = 3,
    Uint16 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    String = 8,
    Bool = 9,
    Float16 = 10,
    Float64 = 11,
    Uint32 = 12,
    Uint64 = 13,
    Complex64 = 14,
    Complex128 = 15,
    Bfloat16 = 16,
}

pub type OgaResultGetErrorFn = unsafe extern "C" fn(result: *const OgaResult) -> *const c_char;
pub type OgaDestroyResultFn = unsafe extern "C" fn(result: *mut OgaResult);

pub type OgaCreateModelFn = unsafe extern "C" fn(config_path: *const c_char, out: *mut *mut OgaModel) -> *mut OgaResult;
pub type OgaDestroyModelFn = unsafe extern "C" fn(model: *mut OgaModel);

pub type OgaCreateTokenizerFn = unsafe extern "C" fn(model: *const OgaModel, out: *mut *mut OgaTokenizer) -> *mut OgaResult;
pub type OgaDestroyTokenizerFn = unsafe extern "C" fn(tokenizer: *mut OgaTokenizer);
pub type OgaTokenizerDecodeFn = unsafe extern "C" fn(
    tokenizer: *const OgaTokenizer,
    tokens: *const i32,
    token_count: usize,
    out_str: *mut *const c_char,
) -> *mut OgaResult;

pub type OgaCreateTokenizerStreamFn = unsafe extern "C" fn(
    tokenizer: *const OgaTokenizer,
    out: *mut *mut OgaTokenizerStream,
) -> *mut OgaResult;
pub type OgaDestroyTokenizerStreamFn = unsafe extern "C" fn(stream: *mut OgaTokenizerStream);
pub type OgaTokenizerStreamDecodeFn = unsafe extern "C" fn(
    stream: *mut OgaTokenizerStream,
    token: i32,
    out_str: *mut *const c_char,
) -> *mut OgaResult;

pub type OgaCreateGeneratorParamsFn = unsafe extern "C" fn(
    model: *const OgaModel,
    out: *mut *mut OgaGeneratorParams,
) -> *mut OgaResult;
pub type OgaDestroyGeneratorParamsFn = unsafe extern "C" fn(params: *mut OgaGeneratorParams);
pub type OgaGeneratorParamsSetSearchNumberFn = unsafe extern "C" fn(
    params: *mut OgaGeneratorParams,
    name: *const c_char,
    val: c_double,
) -> *mut OgaResult;
pub type OgaGeneratorParamsSetSearchBoolFn = unsafe extern "C" fn(
    params: *mut OgaGeneratorParams,
    name: *const c_char,
    val: bool,
) -> *mut OgaResult;
pub type OgaGeneratorParamsSetInputIDsFn = unsafe extern "C" fn(
    params: *mut OgaGeneratorParams,
    input_ids: *const i32,
    input_ids_count: usize,
    sequence_length: usize,
    batch_size: usize,
) -> *mut OgaResult;
pub type OgaGenerator_AppendTokensFn = unsafe extern "C" fn(
    generator: *mut OgaGenerator,
    input_ids: *const i32,
    input_ids_count: usize,
) -> *mut OgaResult;

pub type OgaCreateGeneratorFn = unsafe extern "C" fn(
    model: *const OgaModel,
    params: *const OgaGeneratorParams,
    out: *mut *mut OgaGenerator,
) -> *mut OgaResult;
pub type OgaDestroyGeneratorFn = unsafe extern "C" fn(generator: *mut OgaGenerator);
pub type OgaGenerator_IsDoneFn = unsafe extern "C" fn(generator: *const OgaGenerator) -> bool;
pub type OgaGenerator_GenerateNextTokenFn = unsafe extern "C" fn(generator: *mut OgaGenerator) -> *mut OgaResult;
pub type OgaGenerator_GetNextTokensFn = unsafe extern "C" fn(
    generator: *const OgaGenerator,
    out_tokens: *mut *const i32,
    out_count: *mut usize,
) -> *mut OgaResult;

/// Creates an OgaTensor from a caller-owned buffer. `data` must remain valid for the lifetime of
/// the returned OgaTensor (jetort keeps the backing Vec alive alongside the tensor handle).
pub type OgaCreateTensorFromBufferFn = unsafe extern "C" fn(
    data: *mut std::ffi::c_void,
    shape_dims: *const i64,
    shape_dims_count: usize,
    element_type: OgaElementType,
    out: *mut *mut OgaTensor,
) -> *mut OgaResult;
pub type OgaDestroyTensorFn = unsafe extern "C" fn(tensor: *mut OgaTensor);

/// For additional named model inputs onnxruntime-genai doesn't handle internally (jetort uses this
/// to inject fused audio features / mel spectrograms into Hybrid multimodal models).
pub type OgaGenerator_SetModelInputFn = unsafe extern "C" fn(
    generator: *mut OgaGenerator,
    name: *const c_char,
    tensor: *mut OgaTensor,
) -> *mut OgaResult;

/// Soft-loaded dynamic table for ONNX Runtime GenAI C API
pub struct OgaApi {
    _lib: Library,
    pub ResultGetError: OgaResultGetErrorFn,
    pub DestroyResult: OgaDestroyResultFn,
    pub CreateModel: OgaCreateModelFn,
    pub DestroyModel: OgaDestroyModelFn,
    pub CreateTokenizer: OgaCreateTokenizerFn,
    pub DestroyTokenizer: OgaDestroyTokenizerFn,
    pub TokenizerDecode: OgaTokenizerDecodeFn,
    pub CreateTokenizerStream: Option<OgaCreateTokenizerStreamFn>,
    pub DestroyTokenizerStream: Option<OgaDestroyTokenizerStreamFn>,
    pub TokenizerStreamDecode: Option<OgaTokenizerStreamDecodeFn>,
    pub CreateGeneratorParams: OgaCreateGeneratorParamsFn,
    pub DestroyGeneratorParams: OgaDestroyGeneratorParamsFn,
    pub GeneratorParamsSetSearchNumber: OgaGeneratorParamsSetSearchNumberFn,
    pub GeneratorParamsSetSearchBool: Option<OgaGeneratorParamsSetSearchBoolFn>,
    pub GeneratorParamsSetInputIDs: Option<OgaGeneratorParamsSetInputIDsFn>,
    pub Generator_AppendTokens: Option<OgaGenerator_AppendTokensFn>,
    pub CreateGenerator: OgaCreateGeneratorFn,
    pub DestroyGenerator: OgaDestroyGeneratorFn,
    pub Generator_IsDone: OgaGenerator_IsDoneFn,
    pub Generator_GenerateNextToken: OgaGenerator_GenerateNextTokenFn,
    pub Generator_GetNextTokens: OgaGenerator_GetNextTokensFn,
    pub CreateTensorFromBuffer: Option<OgaCreateTensorFromBufferFn>,
    pub DestroyTensor: Option<OgaDestroyTensorFn>,
    pub Generator_SetModelInput: Option<OgaGenerator_SetModelInputFn>,
}

static OGA_API_INSTANCE: OnceLock<Option<Arc<OgaApi>>> = OnceLock::new();

impl OgaApi {
    /// Discovers and loads the ONNX Runtime GenAI dynamic library
    pub fn get() -> Option<Arc<OgaApi>> {
        OGA_API_INSTANCE
            .get_or_init(|| match Self::discover_and_load() {
                Ok(api) => {
                    info!("Successfully loaded libonnxruntime-genai via soft dynamic linking");
                    Some(Arc::new(api))
                }
                Err(e) => {
                    info!("libonnxruntime-genai is not available in system ({}); GenAI execution core will be in standby mode", e);
                    None
                }
            })
            .clone()
    }

    fn discover_and_load() -> Result<Self, AppError> {
        let candidate_paths = Self::get_candidate_library_paths();
        let mut last_load_error = None;

        for path in &candidate_paths {
            if path.exists() {
                debug!("Attempting to load ORT-GenAI dylib at: {}", path.display());
                match unsafe { Library::new(path) } {
                    Ok(lib) => match unsafe { Self::load_symbols(lib) } {
                        Ok(api) => {
                            info!("Loaded ORT-GenAI shared library from: {}", path.display());
                            return Ok(api);
                        }
                        Err(e) => {
                            warn!(
                                path = %path.display(),
                                error = %e,
                                "Found candidate ORT-GenAI dynamic library, but failed to resolve required symbols"
                            );
                            last_load_error = Some(e);
                        }
                    },
                    Err(e) => {
                        debug!(
                            path = %path.display(),
                            error = %e,
                            "Failed to dlopen candidate ORT-GenAI dynamic library"
                        );
                        last_load_error = Some(AppError::Internal(format!("Failed to dlopen '{}': {}", path.display(), e)));
                    }
                }
            }
        }

        // Try standard system linker name
        #[cfg(target_os = "macos")]
        let default_name = "libonnxruntime-genai.dylib";
        #[cfg(target_os = "windows")]
        let default_name = "onnxruntime-genai.dll";
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        let default_name = "libonnxruntime-genai.so";

        match unsafe { Library::new(default_name) } {
            Ok(lib) => unsafe { Self::load_symbols(lib) },
            Err(e) => {
                if let Some(err) = last_load_error {
                    Err(err)
                } else {
                    Err(AppError::Internal(format!("Failed to load '{}': {}", default_name, e)))
                }
            }
        }
    }

    fn get_candidate_library_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();

        if let Ok(env_path) = std::env::var("ORT_GENAI_DYLIB_PATH") {
            paths.push(PathBuf::from(env_path));
        }
        if let Ok(env_path) = std::env::var("OGA_DYLIB_PATH") {
            let p = PathBuf::from(env_path);
            if !paths.contains(&p) {
                paths.push(p);
            }
        }

        #[cfg(target_os = "macos")]
        {
            paths.push(PathBuf::from("/usr/local/lib/libonnxruntime-genai.dylib"));
            paths.push(PathBuf::from("/opt/homebrew/lib/libonnxruntime-genai.dylib"));
            paths.push(PathBuf::from("libonnxruntime-genai.dylib"));
            paths.push(PathBuf::from("lib/libonnxruntime-genai.dylib"));
        }

        #[cfg(target_os = "windows")]
        {
            paths.push(PathBuf::from("onnxruntime-genai.dll"));
            paths.push(PathBuf::from("lib/onnxruntime-genai.dll"));
        }

        #[cfg(all(target_os = "linux"))]
        {
            paths.push(PathBuf::from("/usr/local/lib/libonnxruntime-genai.so"));
            paths.push(PathBuf::from("/usr/lib/libonnxruntime-genai.so"));
            paths.push(PathBuf::from("/usr/lib/aarch64-linux-gnu/libonnxruntime-genai.so"));
            paths.push(PathBuf::from("/usr/lib/x86_64-linux-gnu/libonnxruntime-genai.so"));
            paths.push(PathBuf::from("libonnxruntime-genai.so"));
            paths.push(PathBuf::from("lib/libonnxruntime-genai.so"));
            paths.push(PathBuf::from("target/release/libonnxruntime-genai.so"));
        }

        paths
    }

    unsafe fn load_symbols(lib: Library) -> Result<Self, AppError> {
        macro_rules! load_sym {
            ($lib:expr, $name:expr) => {
                ($lib.get($name.as_bytes())
                    .map_err(|e| AppError::Internal(format!("Failed to resolve symbol '{}': {}", $name, e)))?)
            };
        }

        let result_get_error: Symbol<OgaResultGetErrorFn> = load_sym!(lib, "OgaResultGetError");
        let destroy_result: Symbol<OgaDestroyResultFn> = load_sym!(lib, "OgaDestroyResult");

        let create_model: Symbol<OgaCreateModelFn> = load_sym!(lib, "OgaCreateModel");
        let destroy_model: Symbol<OgaDestroyModelFn> = load_sym!(lib, "OgaDestroyModel");

        let create_tokenizer: Symbol<OgaCreateTokenizerFn> = load_sym!(lib, "OgaCreateTokenizer");
        let destroy_tokenizer: Symbol<OgaDestroyTokenizerFn> = load_sym!(lib, "OgaDestroyTokenizer");
        let tokenizer_decode: Symbol<OgaTokenizerDecodeFn> = load_sym!(lib, "OgaTokenizerDecode");

        let create_tokenizer_stream: Option<OgaCreateTokenizerStreamFn> = lib
            .get(b"OgaCreateTokenizerStream")
            .ok()
            .map(|s: Symbol<OgaCreateTokenizerStreamFn>| *s);
        let destroy_tokenizer_stream: Option<OgaDestroyTokenizerStreamFn> = lib
            .get(b"OgaDestroyTokenizerStream")
            .ok()
            .map(|s: Symbol<OgaDestroyTokenizerStreamFn>| *s);
        let tokenizer_stream_decode: Option<OgaTokenizerStreamDecodeFn> = lib
            .get(b"OgaTokenizerStreamDecode")
            .ok()
            .map(|s: Symbol<OgaTokenizerStreamDecodeFn>| *s);

        let create_params: Symbol<OgaCreateGeneratorParamsFn> = load_sym!(lib, "OgaCreateGeneratorParams");
        let destroy_params: Symbol<OgaDestroyGeneratorParamsFn> = load_sym!(lib, "OgaDestroyGeneratorParams");
        let params_set_num: Symbol<OgaGeneratorParamsSetSearchNumberFn> = load_sym!(lib, "OgaGeneratorParamsSetSearchNumber");
        let params_set_bool: Option<OgaGeneratorParamsSetSearchBoolFn> = lib
            .get(b"OgaGeneratorParamsSetSearchBool")
            .ok()
            .map(|s: Symbol<OgaGeneratorParamsSetSearchBoolFn>| *s);

        let params_set_input_ids: Option<OgaGeneratorParamsSetInputIDsFn> = lib
            .get(b"OgaGeneratorParamsSetInputIDs")
            .ok()
            .map(|s: Symbol<OgaGeneratorParamsSetInputIDsFn>| *s);

        let gen_append_tokens: Option<OgaGenerator_AppendTokensFn> = lib
            .get(b"OgaGenerator_AppendTokens")
            .ok()
            .map(|s: Symbol<OgaGenerator_AppendTokensFn>| *s);

        if params_set_input_ids.is_none() && gen_append_tokens.is_none() {
            return Err(AppError::Internal(
                "Neither modern 'OgaGenerator_AppendTokens' nor legacy 'OgaGeneratorParamsSetInputIDs' symbol was found in libonnxruntime-genai".to_string()
            ));
        }

        let create_gen: Symbol<OgaCreateGeneratorFn> = load_sym!(lib, "OgaCreateGenerator");
        let destroy_gen: Symbol<OgaDestroyGeneratorFn> = load_sym!(lib, "OgaDestroyGenerator");
        let gen_is_done: Symbol<OgaGenerator_IsDoneFn> = load_sym!(lib, "OgaGenerator_IsDone");
        let gen_next_token: Symbol<OgaGenerator_GenerateNextTokenFn> = load_sym!(lib, "OgaGenerator_GenerateNextToken");
        let gen_get_tokens: Symbol<OgaGenerator_GetNextTokensFn> = load_sym!(lib, "OgaGenerator_GetNextTokens");

        let create_tensor_from_buffer: Option<OgaCreateTensorFromBufferFn> = lib
            .get(b"OgaCreateTensorFromBuffer")
            .ok()
            .map(|s: Symbol<OgaCreateTensorFromBufferFn>| *s);
        let destroy_tensor: Option<OgaDestroyTensorFn> = lib
            .get(b"OgaDestroyTensor")
            .ok()
            .map(|s: Symbol<OgaDestroyTensorFn>| *s);
        let generator_set_model_input: Option<OgaGenerator_SetModelInputFn> = lib
            .get(b"OgaGenerator_SetModelInput")
            .ok()
            .map(|s: Symbol<OgaGenerator_SetModelInputFn>| *s);

        Ok(Self {
            ResultGetError: *result_get_error,
            DestroyResult: *destroy_result,
            CreateModel: *create_model,
            DestroyModel: *destroy_model,
            CreateTokenizer: *create_tokenizer,
            DestroyTokenizer: *destroy_tokenizer,
            TokenizerDecode: *tokenizer_decode,
            CreateTokenizerStream: create_tokenizer_stream,
            DestroyTokenizerStream: destroy_tokenizer_stream,
            TokenizerStreamDecode: tokenizer_stream_decode,
            CreateGeneratorParams: *create_params,
            DestroyGeneratorParams: *destroy_params,
            GeneratorParamsSetSearchNumber: *params_set_num,
            GeneratorParamsSetSearchBool: params_set_bool,
            GeneratorParamsSetInputIDs: params_set_input_ids,
            Generator_AppendTokens: gen_append_tokens,
            CreateGenerator: *create_gen,
            DestroyGenerator: *destroy_gen,
            Generator_IsDone: *gen_is_done,
            Generator_GenerateNextToken: *gen_next_token,
            Generator_GetNextTokens: *gen_get_tokens,
            CreateTensorFromBuffer: create_tensor_from_buffer,
            DestroyTensor: destroy_tensor,
            Generator_SetModelInput: generator_set_model_input,
            _lib: lib,
        })
    }

    /// Checks an OgaResult pointer and translates to Rust Result
    pub unsafe fn check_result(&self, res: *mut OgaResult) -> Result<(), AppError> {
        if res.is_null() {
            return Ok(());
        }
        let err_ptr = (self.ResultGetError)(res);
        let msg = if !err_ptr.is_null() {
            CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
        } else {
            "Unknown ORT-GenAI error".to_string()
        };
        (self.DestroyResult)(res);
        Err(AppError::Inference(format!("ORT-GenAI: {}", msg)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_library_paths_env() {
        std::env::set_var("ORT_GENAI_DYLIB_PATH", "/custom/path/libonnxruntime-genai.so");
        std::env::set_var("OGA_DYLIB_PATH", "/custom/oga/libonnxruntime-genai.so");

        let paths = OgaApi::get_candidate_library_paths();
        assert!(paths.contains(&PathBuf::from("/custom/path/libonnxruntime-genai.so")));
        assert!(paths.contains(&PathBuf::from("/custom/oga/libonnxruntime-genai.so")));

        std::env::remove_var("ORT_GENAI_DYLIB_PATH");
        std::env::remove_var("OGA_DYLIB_PATH");
    }
}

