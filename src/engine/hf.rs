use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

pub struct ResolvedModelPaths {
    pub model_path: PathBuf,
    pub tokenizer_path: Option<PathBuf>,
}

/// Resolves model and optional tokenizer from local Hugging Face CLI cache (~/.cache/huggingface/hub/...)
pub fn resolve_from_hf_cache(hf_repo: &str, hf_file: Option<&str>) -> Option<ResolvedModelPaths> {
    let hub_dir = get_hf_hub_cache_dir()?;
    let repo_dir_name = format!("models--{}", hf_repo.replace('/', "--"));
    let repo_dir = hub_dir.join(repo_dir_name);

    if !repo_dir.exists() {
        warn!(
            hf_repo = %hf_repo,
            repo_dir = %repo_dir.display(),
            "Hugging Face repository directory not found in local cache. Have you run 'huggingface-cli download'?"
        );
        return None;
    }

    let snapshots_dir = repo_dir.join("snapshots");
    if !snapshots_dir.exists() {
        warn!(
            snapshots_dir = %snapshots_dir.display(),
            "Snapshots directory does not exist in HF repo cache"
        );
        return None;
    }

    // Find the latest snapshot directory by modification time
    let snapshot_dir = find_latest_snapshot(&snapshots_dir)?;
    info!(hf_repo = %hf_repo, snapshot = %snapshot_dir.display(), "Located local HF snapshot");

    // Resolve model file
    let model_path = if let Some(file_name) = hf_file {
        let direct_path = snapshot_dir.join(file_name);
        if direct_path.exists() {
            direct_path
        } else {
            warn!(
                file = %file_name,
                snapshot = %snapshot_dir.display(),
                "Specified hf_file not found in snapshot"
            );
            return None;
        }
    } else {
        // Auto-discover any .onnx file in the snapshot
        find_first_onnx_file(&snapshot_dir)?
    };

    // Auto-discover tokenizer.json, vocab.json, vocab.txt, or tokens.txt
    let tokenizer_path = find_tokenizer_file(&snapshot_dir, &snapshots_dir);
    if let Some(ref tok) = tokenizer_path {
        info!(
            tokenizer = %tok.display(),
            "Automatically associated tokenizer/vocabulary from HF cache"
        );
    }

    Some(ResolvedModelPaths {
        model_path,
        tokenizer_path,
    })
}

fn get_hf_hub_cache_dir() -> Option<PathBuf> {
    if let Ok(hf_hub_cache) = env::var("HUGGINGFACE_HUB_CACHE") {
        let p = PathBuf::from(hf_hub_cache);
        if p.exists() {
            return Some(p);
        }
    }

    if let Ok(hf_home) = env::var("HF_HOME") {
        let p = PathBuf::from(hf_home).join("hub");
        if p.exists() {
            return Some(p);
        }
    }

    if let Ok(home) = env::var("HOME") {
        let p = PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub");
        if p.exists() {
            return Some(p);
        }
    }

    None
}

fn find_latest_snapshot(snapshots_dir: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(snapshots_dir).ok()?;
    let mut snapshots = Vec::new();

    for entry in entries.flatten() {
        if let Ok(file_type) = entry.file_type() {
            if file_type.is_dir() {
                let metadata = entry.metadata().ok();
                let modified = metadata
                    .and_then(|m| m.modified().ok())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                snapshots.push((modified, entry.path()));
            }
        }
    }

    // Sort descending by modified time
    snapshots.sort_by(|a, b| b.0.cmp(&a.0));
    snapshots.into_iter().next().map(|(_, p)| p)
}

fn find_first_onnx_file(dir: &Path) -> Option<PathBuf> {
    let priority_names = [
        "model.onnx",
        "model_fp16.onnx",
        "model_quantized.onnx",
        "model_int8.onnx",
        "model_q4.onnx",
    ];
    for name in &priority_names {
        let p = dir.join(name);
        if p.exists() {
            return Some(p);
        }
    }

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(found) = find_first_onnx_file(&path) {
                    return Some(found);
                }
            } else if path.extension().is_some_and(|ext| ext == "onnx") {
                return Some(path);
            }
        }
    }
    None
}

fn find_tokenizer_file(snapshot_dir: &Path, snapshots_dir: &Path) -> Option<PathBuf> {
    let priority_names = [
        "tokenizer.json",
        "vocab.json",
        "vocab.txt",
        "tokens.txt",
        "dict.txt",
    ];

    // 1. Direct check in snapshot_dir
    for name in &priority_names {
        let p = snapshot_dir.join(name);
        if p.exists() {
            return Some(p);
        }
    }

    // 2. Recursive search in snapshot_dir
    if let Some(found) = find_file_recursive(snapshot_dir, &priority_names) {
        return Some(found);
    }

    // 3. Search across all snapshots for this repo
    if let Ok(entries) = fs::read_dir(snapshots_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && path != snapshot_dir {
                for name in &priority_names {
                    let p = path.join(name);
                    if p.exists() {
                        return Some(p);
                    }
                }
                if let Some(found) = find_file_recursive(&path, &priority_names) {
                    return Some(found);
                }
            }
        }
    }

    // 4. Fuzzy check for any file containing vocab or tokens
    find_fuzzy_vocab_recursive(snapshot_dir)
}

fn find_file_recursive(dir: &Path, target_names: &[&str]) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_file_recursive(&path, target_names) {
                return Some(found);
            }
        } else if let Some(fname) = path.file_name().and_then(|n| n.to_str()) {
            for target in target_names {
                if fname.eq_ignore_ascii_case(target) {
                    return Some(path);
                }
            }
        }
    }
    None
}

fn find_fuzzy_vocab_recursive(dir: &Path) -> Option<PathBuf> {
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_fuzzy_vocab_recursive(&path) {
                return Some(found);
            }
        } else if let Some(fname) = path.file_name().and_then(|n| n.to_str()) {
            let lower = fname.to_ascii_lowercase();
            if (lower.contains("vocab") && (lower.ends_with(".txt") || lower.ends_with(".json")))
                || (lower.contains("tokens") && lower.ends_with(".txt"))
            {
                return Some(path);
            }
        }
    }
    None
}
