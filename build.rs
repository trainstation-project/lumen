//! Build script: links CUDA/CUPTI when found, and compiles the Metal/CoreML and PJRT C shims
//! for the `mps` and `tpu` features.
use std::env;
use std::path::{Path, PathBuf};

/// Emits cfg declarations and dispatches to the per-backend setup for each enabled feature.
fn main() {
    println!("cargo:rustc-check-cfg=cfg(lumen_cuda_linked)");
    println!("cargo:rustc-check-cfg=cfg(lumen_mps_linked)");
    println!("cargo:rustc-check-cfg=cfg(lumen_cupti_linked)");
    println!("cargo:rustc-check-cfg=cfg(lumen_tpu_linked)");

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

    if env::var_os("CARGO_FEATURE_TPU").is_some() {
        build_tpu_shim();
    }
}

/// Compiles the PJRT C shim on Linux and sets `lumen_tpu_linked`.
fn build_tpu_shim() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }

    println!("cargo:rerun-if-changed=lumen/device/tpu/pjrt.c");
    println!("cargo:rerun-if-changed=lumen/device/tpu/pjrt_c_api.h");
    println!("cargo:rerun-if-changed=lumen/device/tpu/profiler_c_api.h");

    cc::Build::new()
        .file("lumen/device/tpu/pjrt.c")
        .flag("-std=c11")
        .compile("lumen_tpu_shim");

    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-cfg=lumen_tpu_linked");
}

/// Links cudart (and CUPTI when present) and sets the matching `lumen_*_linked` cfgs,
/// warning instead of failing when the libraries are missing.
fn detect_cuda() {
    let Some((lib_dir, _)) = find_cudart() else {
        println!(
            "cargo:warning=CUDA feature enabled but cudart was not found; \
             building without CUDA (set CUDA_HOME or \
             CUDART_LIB_DIR to link the real backend)"
        );

        return;
    };

    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    println!(
        "cargo:rustc-link-search=native={}",
        lib_dir.join("stubs").display()
    );

    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-cfg=lumen_cuda_linked");

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

/// Finds a directory holding the CUPTI library, searching `CUPTI_LIB_DIR`, the cudart directory
/// and the usual `extras/CUPTI/lib64` locations.
fn find_cupti(cudart_dir: &Path) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = env::var_os("CUPTI_LIB_DIR")
        .map(PathBuf::from)
        .into_iter()
        .collect();

    candidates.push(cudart_dir.to_path_buf());

    for home in ["CUDA_HOME", "CUDA_PATH"].iter().filter_map(env::var_os) {
        candidates.push(PathBuf::from(home).join("extras/CUPTI/lib64"));
    }

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

/// Compiles the Objective-C++ Metal/CoreML shims on macOS with the primitive kernels embedded,
/// and links the Apple frameworks.
fn build_mps_shim() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    println!("cargo:rerun-if-changed=lumen/allocator/mps/mps_shim.mm");
    println!("cargo:rerun-if-changed=lumen/stream/mps/mps.mm");
    println!("cargo:rerun-if-changed=lumen/stream/mps/mps.h");
    println!("cargo:rerun-if-changed=lumen/kernels/mps/shim.mm");
    println!("cargo:rerun-if-changed=lumen/graph_compiler/mps/ane/coreml.mm");

    for source in PRIMITIVE_KERNELS {
        println!("cargo:rerun-if-changed={source}");
    }

    embed_metal_source(
        PRIMITIVE_KERNELS,
        "kPrimitiveKernels",
        "primitive_kernels.h",
    );

    cc::Build::new()
        .file("lumen/allocator/mps/mps_shim.mm")
        .file("lumen/stream/mps/mps.mm")
        .file("lumen/kernels/mps/shim.mm")
        .file("lumen/graph_compiler/mps/ane/coreml.mm")
        .include(PathBuf::from(
            env::var_os("OUT_DIR").expect("OUT_DIR not set"),
        ))
        .cpp(true)
        .flag("-std=c++17")
        .flag("-fobjc-arc")
        .compile("lumen_mps_shim");

    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=CoreML");
    println!("cargo:rustc-cfg=lumen_mps_linked");
}

/// Metal sources concatenated into the shim's `kPrimitiveKernels` string.
const PRIMITIVE_KERNELS: &[&str] = &[
    "lumen/kernels/mps/kernels.metal",
    "lumen/codegen/random/mps/philox.metal",
    "lumen/codegen/random/mps/kernel.metal",
    "lumen/codegen/reduction/mps/kernel.metal",
    "lumen/codegen/scan/mps/kernel.metal",
    "lumen/codegen/dot_general/mps/integer/matmul_kernel.metal",
    "lumen/codegen/dot_general/mps/integer/matmul_wide_kernel.metal",
    "lumen/codegen/dot_general/mps/tiled/matmul_sg_kernel.metal",
    "lumen/codegen/dot_general/mps/tiled/matmul_atomic_kernel.metal",
    "lumen/codegen/dot_general/mps/skinny/kernel.metal",
    "lumen/codegen/dot_general/mps/linear_cross_entropy/kernel.metal",
    "lumen/codegen/dynamic_slice/mps/kernel.metal",
    "lumen/codegen/scatter_add/mps/kernel.metal",
    "lumen/codegen/factory/mps/kernel.metal",
];

/// Concatenates `sources` into an Objective-C raw string constant `name` written to `OUT_DIR/header`.
fn embed_metal_source(sources: &[&str], name: &str, header: &str) {
    let text: String = sources
        .iter()
        .map(|source| {
            std::fs::read_to_string(source).unwrap_or_else(|e| panic!("cannot read {source}: {e}"))
        })
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        !text.contains(")LUMEN_METAL"),
        "{sources:?} may not contain the raw-string delimiter )LUMEN_METAL"
    );

    let contents = format!(
        "// Generated by build.rs from {}; do not edit.\n\
         static NSString *const {name} = @R\"LUMEN_METAL({text})LUMEN_METAL\";\n",
        sources.join(", ")
    );

    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR not set"));

    std::fs::write(out.join(header), contents)
        .unwrap_or_else(|e| panic!("cannot write {header}: {e}"));
}

/// Locates the cudart library directory, returning it with the variable or path that found it.
fn find_cudart() -> Option<(PathBuf, String)> {
    if let Some(dir) = env::var_os("CUDART_LIB_DIR") {
        let dir = PathBuf::from(dir);

        if has_cudart(&dir) {
            return Some((dir, "CUDART_LIB_DIR".to_owned()));
        }
    }

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

/// Reports whether `dir` contains a shared or static cudart library.
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
            || name == "cudart.lib"
    })
}
