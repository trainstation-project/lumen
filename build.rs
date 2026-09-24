//! Build script: locate the CUDA toolkit so the (optional) `cuda`
//! feature can link `cudart` — mirroring how PyTorch's build system
//! probes `CUDA_HOME` and disables CUDA when it isn't found.
//!
//! On a machine with no toolkit (e.g. macOS), `cargo test --all-features`
//! still works: we set `cfg(lumen_cuda_linked)` only when `cudart` is
//! actually available, and the FFI module is gated on that cfg.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    // Declare our custom cfgs so `-D warnings` (via RUSTFLAGS in the
    // Makefile/CI) doesn't reject them as unexpected conditions.
    println!("cargo:rustc-check-cfg=cfg(lumen_cuda_linked)");
    println!("cargo:rustc-check-cfg=cfg(lumen_mps_linked)");

    // Re-run only when the relevant configuration changes.
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDART_LIB_DIR");
    println!("cargo:rerun-if-changed=build.rs");

    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        detect_cuda();
    }

    if env::var_os("CARGO_FEATURE_MPS").is_some() {
        build_mps_shim();
    }
}

fn detect_cuda() {
    let Some((lib_dir, _)) = find_cudart() else {
        // No toolkit: leave `lumen_cuda_linked` unset. The `cuda`
        // feature stays "on" but compiles to a stub (see src/allocator/cuda.rs).
        println!(
            "cargo:warning=CUDA feature enabled but cudart was not found; \
             building the no-GPU caching allocator only (set CUDA_HOME or \
             CUDART_LIB_DIR to link the real backend)"
        );
        return;
    };

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-cfg=lumen_cuda_linked");
}

/// Compile the Objective-C++ Metal shim (macOS only), mirroring how PyTorch
/// builds its MPS .mm files only on Apple targets.
fn build_mps_shim() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        println!(
            "cargo:warning=feature `mps` is only supported on macOS; \
             building without the Metal backend"
        );
        return;
    }

    println!("cargo:rerun-if-changed=csrc/mps_shim.mm");

    cc::Build::new()
        .file("csrc/mps_shim.mm")
        .cpp(true)
        .flag("-std=c++17")
        .flag("-fobjc-arc")
        .compile("lumen_mps_shim");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-cfg=lumen_mps_linked");
}

/// Find the directory containing the CUDA runtime library, honoring the
/// same env vars as the CUDA/PyTorch toolchains.
fn find_cudart() -> Option<(PathBuf, String)> {
    // 1. Explicit override.
    if let Some(dir) = env::var_os("CUDART_LIB_DIR") {
        let dir = PathBuf::from(dir);
        if has_cudart(&dir) {
            return Some((dir, "CUDART_LIB_DIR".to_owned()));
        }
    }

    // 2. `CUDA_HOME` / `CUDA_PATH` with the standard sub-directories.
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        let Some(home) = env::var_os(var) else {
            continue;
        };
        let home = PathBuf::from(home);
        for sub in [
            "lib64",
            "lib/x64",
            "lib",
            "lib64/stubs",
            "targets/aarch64-linux/lib",
        ] {
            let dir = home.join(sub);
            if has_cudart(&dir) {
                return Some((dir, var.to_owned()));
            }
        }
    }

    // 3. Common install locations.
    let candidates = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/lib/x64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda/targets/aarch64-linux/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
    ];
    for dir in candidates {
        let dir = PathBuf::from(dir);
        if has_cudart(&dir) {
            return Some((dir, "/usr/local/cuda".to_owned()));
        }
    }

    None
}

/// True if `dir` contains a `cudart` shared or static library.
fn has_cudart(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    entries.filter_map(Result::ok).any(|e| {
        let name = e.file_name();
        let name = name.to_string_lossy();
        name.starts_with("libcudart") && (name.ends_with(".so") || name.ends_with(".a"))
            || name == "cudart.lib" // Windows
    })
}
