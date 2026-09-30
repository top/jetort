fn main() {
    println!("cargo:rerun-if-env-changed=JETORT_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=TARGET_CUDA_ARCH");
    let arch = std::env::var("JETORT_CUDA_ARCH")
        .or_else(|_| std::env::var("TARGET_CUDA_ARCH"))
        .unwrap_or_default();
    let arch = arch.trim().to_string();
    if !arch.is_empty() && arch.chars().all(|c| c.is_ascii_digit()) {
        println!("cargo:rustc-env=JETORT_CUDA_ARCH={arch}");
    }
}
