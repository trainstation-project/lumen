//! CUDA elementwise kernels: every output of an elementwise fusion as the `BODY` of the fixed
//! `ElementwiseKernel` (`kernel.py`), each tensor read and written at its own strides.

use std::collections::HashMap;
use std::fmt::Write;
use std::hash::{DefaultHasher, Hash, Hasher};

use super::ElementwiseCodegen;
use crate::DType;
use crate::codegen::emitter::Emit;
use crate::codegen::emitter::cuda::Emitter;
use crate::codegen::store::triton::gather_index;
use crate::codegen::utils::cuda::{Cuda, cute_type, unsigned};
use crate::graph::Graph;
use crate::graph_compiler::passes::horizontal::segments;

/// The launch shape of an `ElementwiseKernel` (`kernel.py`): threads per block, and the bytes of its
/// widest tensor each thread moves a step.
#[cfg_attr(not(lumen_cuda_linked), allow(dead_code))]
pub(crate) type Shape = (usize, usize);

/// The shape kernels run with untuned (`kernel.py`'s `THREADS` and `STEP_BYTES`).
#[cfg_attr(not(lumen_cuda_linked), allow(dead_code))]
pub(crate) const UNTUNED: Shape = (256, 16);

/// The shapes autotuning times: 128 to 1024 threads, 16 to 64 bytes a thread.
#[cfg_attr(not(lumen_cuda_linked), allow(dead_code))]
pub(crate) fn shapes() -> Vec<Shape> {
    let threads = (1..=8).map(|k| 128 * k);
    threads.flat_map(|t| [16, 32, 64].map(|b| (t, b))).collect()
}

/// The strides (in elements, from the first) a CUDA elementwise kernel reads each input and writes
/// each output at: what its generated loads and stores index with; and the elements each tensor spans,
/// inputs then outputs, the flat views its kernel builds from their pointers.
pub(crate) struct Layouts {
    pub inputs: Vec<Vec<usize>>,
    pub outputs: Vec<Vec<usize>>,
    pub spans: Vec<usize>,
}

/// CUDA elementwise fusions: the fixed `ElementwiseKernel` (lumen.codegen.elementwise.cuda.kernel) walks
/// the outputs and runs a generated `BODY` at each index; `lumen.codegen.utils.cuda` compiles and runs
/// the generated module's `launch`.
impl ElementwiseCodegen for Cuda {
    /// The kernel's name and its generated module's Python source.
    type Code = (String, String);
    type Schedule = Layouts;

    /// Generates `BODY(ins, outs, fins, fouts, j, v, SEGMENT)`, computing and storing every output of
    /// segment `SEGMENT` (the outputs of one element count, in `segments`' order) at flat index `j`, each
    /// output at its `layouts` strides, and `launch(in0, …, out0, …, blocks, stream, vectorized, threads, step_bytes)` running
    /// it in `ElementwiseKernel`. Tensors read or written at `j` itself go through `fins`/`fouts` at `v`
    /// (a thread's vector of consecutive elements, or the tensors themselves with `v = j`); each
    /// segment's are listed for the kernel, with whether the segment reads every input in row-major
    /// order (vectors pay only then). Inputs flagged in `by_value` arrive as scalars.
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
        let (mut fragments_in, mut fragments_out) = (String::new(), String::new());
        let mut contiguous = String::new();

        for (s, (_, outputs)) in segments.iter().enumerate() {
            // Values computed in another segment's branch aren't defined in this one.
            emitter.values.clear();
            emitter.indices.clear();
            emitter.lines.clear();
            emitter.fragments.clear();
            emitter.scattered = false;

            let mut stores = String::new();
            let mut written = Vec::new();

            for &k in outputs {
                let v = body.outputs()[k];
                let value = emitter.value(v, "j".into());

                // A bool is stored as a byte.
                let value = match body.type_of(v).dtype {
                    DType::Bool => format!("cutlass.Boolean({value}).to(cutlass.Uint8)"),
                    _ => value,
                };

                let at = gather_index("j", &body.type_of(v).shape, &layouts.outputs[k]);

                match at == "j" {
                    true => {
                        written.push(k);
                        writeln!(stores, "        fout{k}[v] = {value}").unwrap();
                    }
                    false => writeln!(stores, "        out{k}[{at}] = {value}").unwrap(),
                }
            }

            let listed = |ks: &mut dyn Iterator<Item = usize>| -> String {
                format!("({}), ", ks.map(|k| format!("{k}, ")).collect::<String>())
            };

            fragments_in += &listed(&mut emitter.fragments.iter().copied());
            fragments_out += &listed(&mut written.into_iter());
            contiguous += if emitter.scattered {
                "False, "
            } else {
                "True, "
            };

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

from lumen.codegen.elementwise.cuda.kernel import FAST_MATH, ElementwiseKernel

# The math functions BODY calls: CuTe's, or their fast approximations under config.compiler.fast_math.
math = {math}


@cute.jit
def BODY(ins, outs, fins, fouts, j, v, SEGMENT: cutlass.Constexpr):
    """Every output of segment `SEGMENT` at flat index `j`, the tensors read and written at `j` through
    `fins` and `fouts` at `v`."""
{unpack_ins}{unpack_outs}{unpack_fins}{unpack_fouts}{code}

@cute.jit
def launch(
    {tensors},
    blocks: Int32,
    stream: cuda.CUstream,
    vectorized: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    step_bytes: cutlass.Constexpr,
):
    """Runs the fusion's `ElementwiseKernel` on `blocks` blocks on `stream`, with vector loads and stores
    when `vectorized` (every tensor 16-byte aligned), its blocks of `threads` threads each moving `step_bytes`
    of its widest tensor a step."""
    kernel = ElementwiseKernel(BODY, ({counts}), ({types}), ({spans}), ({fragments_in}), ({fragments_out}), ({contiguous}), vectorized, threads, step_bytes)
    kernel.launch([{ins}], [{outs}], blocks, stream)
"#,
            math = match crate::graph_compiler::config::config().fast_math {
                true => "FAST_MATH",
                false => "cute.math",
            },
            unpack_ins = unpack("in", ins.len()),
            unpack_outs = unpack("out", outs.len()),
            unpack_fins = unpack("fin", ins.len()),
            unpack_fouts = unpack("fout", outs.len()),
            fragments_in = fragments_in,
            fragments_out = fragments_out,
            contiguous = contiguous,
            types = body
                .inputs()
                .iter()
                .chain(body.outputs())
                .map(|&v| match body.type_of(v).dtype {
                    DType::Bool => format!("{}, ", unsigned(8)),
                    d => format!("{}, ", cute_type(d).expect("a CuTe dtype")),
                })
                .collect::<String>(),
            spans = layouts
                .spans
                .iter()
                .map(|n| format!("{n}, "))
                .collect::<String>(),
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

/// Blocks a kernel of launch shape `(threads, step_bytes)` over `n` elements whose widest tensor has
/// `widest`-byte elements launches: one per tile of its threads' steps, so each block makes one pass
/// (faster than fewer blocks looping).
#[cfg_attr(not(lumen_cuda_linked), allow(dead_code))]
pub(crate) fn blocks(n: usize, widest: usize, (threads, step_bytes): Shape) -> usize {
    let per_thread = (step_bytes / widest).max(1);
    n.div_ceil(threads * per_thread).max(1)
}
