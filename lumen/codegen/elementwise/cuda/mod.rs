//! CUDA elementwise kernels: every output of an elementwise fusion as the `BODY` of the fixed
//! `ElementwiseKernel` (`kernel.py`), each tensor read and written at its own strides.

use std::collections::HashMap;
use std::fmt::Write;
use std::hash::{DefaultHasher, Hash, Hasher};

use super::ElementwiseCodegen;
use crate::codegen::dot_general::GemmCodegen;
use crate::codegen::emitter::Emit;
use crate::codegen::emitter::cuda::Emitter;
use crate::codegen::kernel::Kernel;
use crate::codegen::kinds::gemm_dot;
use crate::codegen::store::triton::gather_index;
use crate::codegen::utils::cuda::{Cuda, cute_type, unsigned};
use crate::graph::Graph;
use crate::graph::{Primitive, TensorType};
use crate::graph_compiler::passes::horizontal::segments;
use crate::plan::CudaStep;
use crate::plan::StepBase;
use crate::plan::step::Tensors;
use crate::runtime::kernel::cuda::LAUNCH;
use crate::{DType, Tensor};
use std::sync::Arc;

/// The launch shape of an `ElementwiseKernel` (`kernel.py`): threads per block, and the bytes of its
/// widest tensor each thread moves a step.
#[cfg_attr(not(all(feature = "python", lumen_cuda_linked)), allow(dead_code))]
pub(crate) type Shape = (usize, usize);

/// The shape kernels run with untuned (`kernel.py`'s `THREADS` and `STEP_BYTES`).
#[cfg_attr(not(all(feature = "python", lumen_cuda_linked)), allow(dead_code))]
pub(crate) const UNTUNED: Shape = (256, 16);

/// The shapes autotuning times: 128 to 1024 threads, 16 to 64 bytes a thread.
#[cfg_attr(not(all(feature = "python", lumen_cuda_linked)), allow(dead_code))]
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
#[cfg_attr(not(all(feature = "python", lumen_cuda_linked)), allow(dead_code))]
pub(crate) fn blocks(n: usize, widest: usize, (threads, step_bytes): Shape) -> usize {
    let per_thread = (step_bytes / widest).max(1);
    n.div_ceil(threads * per_thread).max(1)
}

/// The kernel generated for an elementwise fusion or an output's copy (a reshape), for its tensors' layouts.
pub(crate) struct Elementwise;

impl Kernel for Elementwise {
    type Step = CudaStep;
    type Buffers = Tensors;
    type Config = ();

    /// Runs the kernel generated for the tensors' layouts (kept in the step's memo), launched by Python, over the
    /// largest output's elements.
    fn run(
        &self,
        step: &CudaStep,
        _: Option<&()>,
        (inputs, out, outs): &Tensors,
    ) -> Result<(), String> {
        let launch = LAUNCH.get().ok_or(
            "the CUDA backend launches from Python (lumen.codegen.utils.cuda): none is installed",
        )?;

        let outputs: Vec<&Tensor> = std::iter::once(out).chain(outs).collect();
        let n = outputs.iter().map(|t| t.numel()).max().unwrap_or(0);

        if n == 0 {
            return Ok(());
        }

        let tensors: Vec<Tensor> = inputs
            .iter()
            .chain(outputs.iter().copied())
            .map(Tensor::flat)
            .collect();

        let layouts = Layouts {
            inputs: inputs.iter().map(|t| t.strides().to_vec()).collect(),
            outputs: outputs.iter().map(|t| t.strides().to_vec()).collect(),
            spans: tensors.iter().map(Tensor::numel).collect(),
        };

        let code = crate::codegen::elementwise::cuda::generated(step, layouts)?;
        let (name, source) = code.as_ref();
        launch(name, source, &tensors, outputs.len(), n)
    }
}

/// The kernel `step` runs as for tensors laid out as `layouts` (a GEMM fusion's epilogue), generated once per layout
/// and kept in the step's memo.
pub(crate) fn generated(
    step: &StepBase,
    layouts: Layouts,
) -> Result<Arc<(String, String)>, String> {
    let key = layouts
        .inputs
        .iter()
        .chain(&layouts.outputs)
        .chain(std::iter::once(&layouts.spans))
        .fold(0u64, |h, s| {
            use std::hash::{Hash, Hasher};

            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            (h, s).hash(&mut hasher);
            hasher.finish()
        });

    step.memo.get(key, || {
        let by_value = vec![false; layouts.inputs.len()];

        let body = match &step.primitive {
            Primitive::Fusion { body, .. } => match gemm_dot(body) {
                Some(dot) => return Cuda::gemm(body, dot, &by_value, &layouts),
                None => body.clone(),
            },
            Primitive::Reshape { new_sizes } => copy_body(&step.inputs[0].1, new_sizes),
            p => {
                return Err(format!(
                    "{}: the CUDA backend has no kernel for it yet",
                    p.name()
                ));
            }
        };

        Cuda::elementwise(&body, &by_value, &Default::default(), &layouts)
    })
}

/// Elements a tensor of `shape` laid out at `strides` spans, from its first to its last.
pub(crate) fn extent(shape: &[usize], strides: &[usize]) -> usize {
    match shape.contains(&0) {
        true => 0,
        false => {
            shape
                .iter()
                .zip(strides)
                .map(|(&n, &s)| (n - 1) * s)
                .sum::<usize>()
                + 1
        }
    }
}

/// The body of a copy of `x` reshaped to `shape`: its input, reshaped, its output.
pub(crate) fn copy_body(x: &TensorType, shape: &[usize]) -> Graph {
    let mut body = Graph::new();
    let input = body.input(x.clone());

    let output = body
        .apply(
            Primitive::Reshape {
                new_sizes: shape.to_vec(),
            },
            &[input],
        )
        .expect("a reshape of the copied value");

    body.set_outputs(&[output]).expect("the copied value");
    body
}
