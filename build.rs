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
    println!("cargo:rustc-check-cfg=cfg(lumen_cupti_linked)");

    // Re-run only when the relevant configuration changes.
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDART_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUPTI_LIB_DIR");
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
        // feature stays "on" but compiles to a stub (see lumen/allocator/cuda.rs).
        println!(
            "cargo:warning=CUDA feature enabled but cudart was not found; \
             building the no-GPU caching allocator only (set CUDA_HOME or \
             CUDART_LIB_DIR to link the real backend)"
        );
        return;
    };

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    // The driver API (`libcuda`, for element memsets) links against the
    // toolkit's stub; the driver's real library is loaded at run time.
    println!(
        "cargo:rustc-link-search=native={}",
        lib_dir.join("stubs").display()
    );
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-cfg=lumen_cuda_linked");

    // CUPTI, for the profiler's CUDA timing (PyTorch's kineto uses it too).
    // It ships with the toolkit, often outside the loader's default paths,
    // so the library records where it was found (rpath).
    let Some(cupti_dir) = find_cupti(&lib_dir) else {
        println!(
            "cargo:warning=cudart was found but not CUPTI; the profiler will not \
             time CUDA work (set CUPTI_LIB_DIR to enable it)"
        );
        return;
    };
    println!("cargo:rustc-link-search=native={}", cupti_dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cupti_dir.display());
    println!("cargo:rustc-link-lib=cupti");
    println!("cargo:rustc-cfg=lumen_cupti_linked");
}

/// The directory holding `libcupti`: `CUPTI_LIB_DIR`, next to cudart, or
/// the toolkit's `extras/CUPTI`.
fn find_cupti(cudart_dir: &Path) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = env::var_os("CUPTI_LIB_DIR")
        .map(PathBuf::from)
        .into_iter()
        .collect();
    candidates.push(cudart_dir.to_path_buf());
    for home in ["CUDA_HOME", "CUDA_PATH"].iter().filter_map(env::var_os) {
        candidates.push(PathBuf::from(home).join("extras/CUPTI/lib64"));
    }
    // cudart_dir is usually <toolkit>/lib64 or <toolkit>/targets/<arch>/lib.
    for up in cudart_dir.ancestors().skip(1).take(3) {
        candidates.push(up.join("extras/CUPTI/lib64"));
    }
    candidates.push(PathBuf::from("/usr/local/cuda/extras/CUPTI/lib64"));
    candidates.into_iter().find(|dir| {
        std::fs::read_dir(dir).is_ok_and(|entries| {
            entries.filter_map(Result::ok).any(|e| {
                let name = e.file_name();
                let name = name.to_string_lossy();
                name.starts_with("libcupti.so") || name == "cupti.lib"
            })
        })
    })
}

/// Compile the Objective-C++ Metal shim (macOS only), mirroring how PyTorch
/// builds its MPS .mm files only on Apple targets.
fn build_mps_shim() {
    // `mps` is a default feature, so skip quietly elsewhere (PyTorch
    // likewise just builds without MPS off Apple platforms).
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    println!("cargo:rerun-if-changed=lumen/allocator/mps_shim.mm");
    println!("cargo:rerun-if-changed=lumen/ops/fill/mps/mps_fill.mm");
    println!("cargo:rerun-if-changed=lumen/ops/fill/mps/fill.metal");
    println!("cargo:rerun-if-changed=lumen/stream/mps/mps.mm");
    println!("cargo:rerun-if-changed=lumen/stream/mps/mps.h");

    // The compute kernels live in fill.metal (an actual Metal source file,
    // not a string literal). Embed its text as a C++ raw string so
    // mps_fill.mm can hand it to newLibraryWithSource: at runtime, without
    // needing Xcode's metal/metallib tools at build time.
    embed_metal_source();

    cc::Build::new()
        .file("lumen/allocator/mps_shim.mm")
        .file("lumen/ops/fill/mps/mps_fill.mm")
        .file("lumen/stream/mps/mps.mm")
        .include(PathBuf::from(
            env::var_os("OUT_DIR").expect("OUT_DIR not set"),
        ))
        .cpp(true)
        .flag("-std=c++17")
        .flag("-fobjc-arc")
        .compile("lumen_mps_shim");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-cfg=lumen_mps_linked");
}

/// Write the text of `lumen/ops/fill/mps/fill.metal` into a header in
/// `OUT_DIR` as a C++ raw string literal (`kMpsFillSource`), which
/// `mps_fill.mm` includes. Keeps the kernels in a real `.metal` file while
/// still compiling them at runtime, so no Xcode `metal`/`metallib` tools are
/// needed to build.
fn embed_metal_source() {
    const SOURCE: &str = "lumen/ops/fill/mps/fill.metal";
    let text =
        std::fs::read_to_string(SOURCE).unwrap_or_else(|e| panic!("cannot read {SOURCE}: {e}"));
    // A delimiter that cannot occur in the source, so the raw string is safe.
    assert!(
        !text.contains(")LUMEN_METAL"),
        "{SOURCE} may not contain the raw-string delimiter )LUMEN_METAL"
    );
    let header = format!(
        "// Generated by build.rs from {SOURCE}; do not edit.\n\
         static NSString *const kMpsFillSource = @R\"LUMEN_METAL({text})LUMEN_METAL\";\n"
    );
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set"));
    std::fs::write(out.join("mps_fill_source.h"), header).expect("cannot write mps_fill_source.h");
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
