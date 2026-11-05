//! Triton elementwise kernels: every output of an elementwise fusion, as the `BODY` of the fixed
//! `elementwise_kernel` (`kernel.py`); and the fill and copy kernels tensors run on CUDA.

mod config;

pub(crate) use config::*;

use std::collections::HashMap;
use std::fmt::Write;
use std::sync::{Arc, Mutex, PoisonError};

use super::ElementwiseCodegen;
use crate::codegen::emitter::Emit;
use crate::codegen::emitter::triton::Emitter;
use crate::codegen::store::triton::store;
use crate::codegen::triton::{Kernel, Template};
use crate::codegen::utils::triton as codegen;
use crate::codegen::utils::triton::*;
use crate::graph::{Graph, TensorType};
use crate::graph_compiler::passes::horizontal::segments;
use crate::runtime::kernel::triton::{Function, Launches, launch};
use crate::{DType, Scalar, Tensor};

/// Triton elementwise fusions: the fixed `elementwise_kernel` (lumen.codegen.elementwise.triton.kernel) runs
/// a generated `BODY` over blocks of each grid segment; the body loads and stores through helpers for its
/// tensors' `layouts` (so `views` goes unread). Its block size is tuned at launch.
impl ElementwiseCodegen for Triton {
    /// The `BODY` function's name and Python source.
    type Code = (String, String);
    type Schedule = Tensors;

    /// Generates `BODY(EXTRA, BLOCK, MASK, SEGMENT)`, computing every output of grid segment `SEGMENT` at
    /// the flat indices `BLOCK`, masked by `MASK`; inputs flagged in `by_value` arrive as scalars. Outputs
    /// of different element counts are segments of their own (`segments`), in its order: each stores only
    /// its outputs. The first output is `out` stored through `STORE`; output `k` after it is `out{k}`,
    /// stored through `STORE{k}`.
    fn elementwise(
        body: &Graph,
        by_value: &[bool],
        _: &HashMap<usize, (usize, Vec<usize>)>,
        layouts: &Tensors,
    ) -> Result<(String, String), String> {
        let segments = segments(body);
        let mut emitter = Emitter::new(body, by_value);
        emitter.mask = "MASK".into();

        // Statements computing and storing the outputs at `k`, then the ones computing them.
        let stored = |emitter: &mut Emitter, k: &[usize]| {
            let stores: String = k
                .iter()
                .map(|&k| {
                    let result = emitter.value(body.outputs()[k], "BLOCK".into());

                    match k {
                        0 => format!("    STORE(out, BLOCK, {result}, MASK)\n"),
                        k => format!("    STORE{k}(out{k}, BLOCK, {result}, MASK)\n"),
                    }
                })
                .collect::<Vec<_>>()
                .concat();

            std::mem::take(&mut emitter.lines) + &stores
        };

        let code = match segments.as_slice() {
            [(_, k)] => stored(&mut emitter, k),
            _ => {
                let mut code = String::new();

                for (s, (_, k)) in segments.iter().enumerate() {
                    let branch = match s {
                        0 => "if SEGMENT == 0".to_string(),
                        s => format!("elif SEGMENT == {s}"),
                    };

                    // Values computed in another branch aren't defined in this one.
                    emitter.values.clear();
                    emitter.indices.clear();

                    let statements: String = stored(&mut emitter, k)
                        .lines()
                        .map(|line| format!("    {line}\n"))
                        .collect();

                    write!(code, "    {branch}:\n{statements}").unwrap();
                }

                code
            }
        };

        Ok(body_function(
            "elementwise",
            layouts,
            "BLOCK, MASK, SEGMENT: tl.constexpr",
            &code,
        ))
    }
}

/// A fill or copy kernel with its helpers for one layout of its tensors, and the kernels compiled for it.
struct Utility {
    kernel: Kernel,
    functions: Vec<Function>,
    launches: Launches,
}

/// The fill or copy kernel `make` builds for the tensors' dtypes, shapes and strides in `layouts`,
/// made once per layout.
fn utility(layouts: &[&Tensor], make: impl FnOnce() -> (Kernel, Vec<Function>)) -> Arc<Utility> {
    type Layout = (DType, Vec<usize>, Vec<usize>);
    static UTILITIES: Mutex<Option<HashMap<Vec<Layout>, Arc<Utility>>>> = Mutex::new(None);

    let key: Vec<Layout> = layouts
        .iter()
        .map(|t| (t.dtype(), t.shape().to_vec(), t.strides().to_vec()))
        .collect();

    let mut cache = UTILITIES.lock().unwrap_or_else(PoisonError::into_inner);
    let cache = cache.get_or_insert_with(HashMap::new);

    cache
        .entry(key)
        .or_insert_with(|| {
            let (kernel, functions) = make();

            Arc::new(Utility {
                kernel,
                functions,
                launches: Launches::default(),
            })
        })
        .clone()
}

/// Fills `t` with `value` using a Triton kernel.
pub(crate) fn fill(t: &Tensor, value: Scalar) -> Result<(), String> {
    let fill = utility(&[t], || {
        let (name, source) = codegen::fill(t.dtype());

        let kernel = Kernel {
            name,
            source,
            by_value: vec![true],
            lanes: 1,
            splits: 1,
            segments: Vec::new(),
            template: None,
        };

        let store = Function::new("STORE", store(t.shape(), t.strides()));
        (kernel, vec![store])
    });

    let bits = Tensor::from_slice(&[codegen::bits(t.dtype(), value)], DType::U64);

    launch(
        &fill.kernel,
        &LOOP_CONFIGS[UNTUNED_LOOP],
        &fill.functions,
        &fill.launches,
        &[bits],
        t,
        t.numel(),
    )
}

/// Copies `src` into `dst`, which may have different strides, using a Triton kernel.
pub(crate) fn copy(dst: &Tensor, src: &Tensor) -> Result<(), String> {
    let copy = utility(&[dst, src], || {
        let mut body = Graph::new();
        let x = body.input(TensorType::new(src.dtype(), src.shape()));
        body.set_outputs(&[x]).expect("the copied value");

        let layouts = codegen::Tensors {
            inputs: vec![Some((src.shape().to_vec(), src.strides().to_vec()))],
            outputs: vec![(dst.shape().to_vec(), dst.strides().to_vec())],
        };

        let generated = codegen::Triton::elementwise(&body, &[false], &HashMap::new(), &layouts)
            .expect("a Triton loop of a copy");

        let kernel = Kernel {
            name: generated.0.clone(),
            source: String::new(),
            by_value: vec![false],
            lanes: 1,
            splits: 1,
            segments: Vec::new(),
            template: Some(Template::Elementwise),
        };

        let functions = vec![Function::new("BODY", generated)];

        (kernel, functions)
    });

    launch(
        &copy.kernel,
        &LOOP_CONFIGS[UNTUNED_LOOP],
        &copy.functions,
        &copy.launches,
        std::slice::from_ref(src),
        dst,
        dst.numel(),
    )
}
