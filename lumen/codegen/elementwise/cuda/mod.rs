//! CUDA elementwise kernels: every output of an elementwise fusion as the `BODY` of the fixed
//! `ElementwiseKernel` (`kernel.py`), each tensor read and written at its own strides.

use std::collections::HashMap;
use std::fmt::Write;
use std::hash::{DefaultHasher, Hash, Hasher};

use super::ElementwiseCodegen;
use crate::codegen::emitter::Emit;
use crate::codegen::emitter::cuda::Emitter;
use crate::codegen::store::triton::gather_index;
use crate::codegen::utils::cuda::Cuda;
use crate::graph::Graph;
use crate::graph_compiler::common::horizontal::segments;

/// Threads per block and elements per thread per step of `ElementwiseKernel` (`kernel.py`'s
/// `THREADS` and `UNROLL`), which size its grid.
const THREADS: usize = 256;
const UNROLL: usize = 4;

/// The strides (in elements, from the first) a CUDA elementwise kernel reads each input and writes
/// each output at: what its generated loads and stores index with.
pub(crate) struct Layouts {
    pub inputs: Vec<Vec<usize>>,
    pub outputs: Vec<Vec<usize>>,
}

/// CUDA elementwise fusions: the fixed `ElementwiseKernel` (lumen.codegen.elementwise.cuda.kernel) walks
/// the outputs and runs a generated `BODY` at each index; `lumen.codegen.utils.cuda` compiles and runs
/// the generated module's `launch`.
impl ElementwiseCodegen for Cuda {
    /// The kernel's name and its generated module's Python source.
    type Code = (String, String);
    type Schedule = Layouts;

    /// Generates `BODY(ins, outs, j, SEGMENT)`, computing and storing every output of segment `SEGMENT` (the
    /// outputs of one element count, in `segments`' order) at flat index `j`, each output at its `layouts`
    /// strides, and `launch(in0, …, out0, …, blocks, stream)` running it in `ElementwiseKernel`. Inputs
    /// flagged in `by_value` arrive as scalars.
    fn elementwise(
        body: &Graph,
        by_value: &[bool],
        _: &HashMap<usize, (usize, Vec<usize>)>,
        layouts: &Layouts,
    ) -> Result<(String, String), String> {
        let mut emitter = Emitter::new(body, by_value);
        emitter.strides = layouts.inputs.clone();
        emitter.indent = " ".repeat(8);

        let segments: Vec<(usize, Vec<usize>)> =
            segments(body).into_iter().filter(|(n, _)| *n > 0).collect();

        let mut code = String::new();

        for (s, (_, outputs)) in segments.iter().enumerate() {
            // Values computed in another segment's branch aren't defined in this one.
            emitter.values.clear();
            emitter.indices.clear();
            emitter.lines.clear();

            let mut stores = String::new();

            for &k in outputs {
                let v = body.outputs()[k];
                let value = emitter.value(v, "j".into());
                let at = gather_index("j", &body.type_of(v).shape, &layouts.outputs[k]);
                writeln!(stores, "        out{k}[{at}] = {value}").unwrap();
            }

            write!(
                code,
                "    if cutlass.const_expr(SEGMENT == {s}):\n{}{stores}",
                emitter.lines
            )
            .unwrap();
        }

        let unpack = |prefix: &str, count: usize| -> String {
            (0..count)
                .map(|k| format!("    {prefix}{k} = {prefix}s[{k}]\n"))
                .collect::<String>()
        };

        let names = |prefix: &str, count: usize| -> Vec<String> {
            (0..count).map(|k| format!("{prefix}{k}")).collect()
        };

        let (ins, outs) = (
            names("in", body.inputs().len()),
            names("out", body.outputs().len()),
        );
        let counts: Vec<String> = segments.iter().map(|(n, _)| format!("{n}, ")).collect();

        let source = format!(
            r#"import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32

from lumen.codegen.elementwise.cuda.kernel import ElementwiseKernel


@cute.jit
def BODY(ins, outs, j, SEGMENT: cutlass.Constexpr):
    """Every output of segment `SEGMENT` at flat index `j`."""
{unpack_ins}{unpack_outs}{code}

@cute.jit
def launch({tensors}, blocks: Int32, stream: cuda.CUstream):
    """Runs the fusion's `ElementwiseKernel` on `blocks` blocks on `stream`."""
    ElementwiseKernel(BODY, ({counts})).launch([{ins}], [{outs}], blocks, stream)
"#,
            unpack_ins = unpack("in", ins.len()),
            unpack_outs = unpack("out", outs.len()),
            tensors = [ins.clone(), outs.clone()].concat().join(", "),
            counts = counts.concat(),
            ins = ins.join(", "),
            outs = outs.join(", "),
        );

        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);

        Ok((format!("cuda_elementwise_{:016x}", hasher.finish()), source))
    }
}

/// Blocks a kernel over `n` elements launches: enough for every thread's first step, at most 4096
/// (the walk is grid-strided, so fewer cover any `n`).
pub(crate) fn blocks(n: usize) -> usize {
    n.div_ceil(THREADS * UNROLL).clamp(1, 4096)
}
