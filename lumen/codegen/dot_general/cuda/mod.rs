//! The CUDA (CuTe DSL) GEMM: which matmuls it runs, its launch configs, and the epilogue it runs fused after one. The
//! kernel and its launcher are Python (`ampere.py`, `launch.py`).

mod config;
pub(crate) use config::*;

use std::fmt::Write;
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::DType;
use crate::codegen::dot_general::GemmCodegen;
use crate::codegen::elementwise::cuda::Layouts;
use crate::codegen::emitter::Emit;
use crate::codegen::emitter::cuda::Emitter;
use crate::codegen::regions::{Region, gemm_regions};
use crate::codegen::store::triton::gather_index;
use crate::codegen::utils::cuda::{Cuda, cute_type, unsigned};
use crate::graph::{Graph, Node, Primitive, Var};

/// Bytes of the vectors the GEMM's epilogue moves the product by, from shared memory to its outputs.
const VECTOR_BYTES: usize = 16;

/// Whether `node` is a matmul the GEMM runs: float16 or bfloat16 operands of rank 2, or 3 with a leading batch dim,
/// contracting one dim each, accumulated in float32 and written as float16, bfloat16 or float32, every extent a
/// multiple of 8 (its tiles move 16-byte vectors along whichever dim is contiguous).
pub(crate) fn matmul(graph: &Graph, node: &Node) -> bool {
    use crate::DType::{BF16, F16, F32};

    let Primitive::DotGeneral {
        lhs_contracting,
        rhs_contracting,
        lhs_batch,
        rhs_batch,
        accum_dtype,
        ..
    } = &node.primitive
    else {
        return false;
    };

    let (lhs, rhs) = (graph.type_of(node.inputs[0]), graph.type_of(node.inputs[1]));
    let out = graph.type_of(node.output);
    let rank = lhs.shape.len();

    let batched = match rank {
        2 => lhs_batch.is_empty() && rhs_batch.is_empty(),
        3 => *lhs_batch == [0] && *rhs_batch == [0],
        _ => false,
    };

    let contracting = matches!((lhs_contracting.as_slice(), rhs_contracting.as_slice()), ([l], [r]) if *l >= rank - 2 && *r >= rank - 2);
    let typed = lhs.dtype == rhs.dtype && matches!(lhs.dtype, F16 | BF16) && *accum_dtype == F32;
    let extents = lhs.shape[rank - 2..].iter().chain(&rhs.shape[rank - 2..]);

    batched
        && contracting
        && rhs.shape.len() == rank
        && typed
        && matches!(out.dtype, F16 | BF16 | F32)
        && extents.into_iter().all(|&d| d > 0 && d % 8 == 0)
}

/// Elements of `dot`'s product (of `dtype`) in one of the epilogue's vectors.
fn lanes(dtype: DType) -> usize {
    VECTOR_BYTES / dtype.size_of()
}

/// Whether the GEMM runs an epilogue after `dot` in `graph`: a matmul it runs whose readers, when they split its
/// product into column regions, each take whole vectors of it: regions along its last axis, each a plain slice or
/// the interleaved members of a reshape (as many as divide a vector), not the two halves of the product (which would
/// need both halves' tiles in one CTA).
pub(crate) fn epilogue(graph: &Graph, dot: &Node) -> bool {
    if !matmul(graph, dot) {
        return false;
    }

    let product = graph.type_of(dot.output);
    let (rank, lanes) = (product.shape.len(), lanes(product.dtype));

    gemm_regions(graph, dot).is_none_or(|regions| {
        regions.iter().all(|c| {
            let fits = match &c.region {
                Region::Plain(_) => true,
                Region::Group(members) => {
                    members[0].inputs[0] != dot.output && lanes.is_multiple_of(members.len())
                }
            };

            c.axis + 1 == rank
                && c.offset.is_multiple_of(lanes)
                && c.width.is_multiple_of(lanes)
                && fits
        })
    })
}

/// Whether `v` is one of `on` or is computed from any of them in `body`.
fn depends(body: &Graph, v: Var, on: &[Var]) -> bool {
    let producer = body.nodes().iter().find(|n| n.output == v);
    on.contains(&v) || producer.is_some_and(|n| n.inputs.iter().any(|&u| depends(body, u, on)))
}

/// `value`, of `dtype`, as it is stored: a bool as a byte.
fn stored(value: String, dtype: DType) -> String {
    match dtype {
        DType::Bool => format!("cutlass.Boolean({value}).to(cutlass.Uint8)"),
        _ => value,
    }
}

/// CUDA GEMM epilogues: the Ampere GEMM computes the product and stages each subtile of it in shared memory, then
/// each thread hands a generated `EPILOGUE` its vectors of it, which computes the fusion's outputs from them and
/// stores each at its strides.
impl GemmCodegen for Cuda {
    /// The generated module's name and Python source.
    type Code = (String, String);
    type Options = Layouts;

    /// Generates `EPILOGUE(ins, outs, acc, l, m, n, aligned)`: the fusion's outputs at the `VECTOR` consecutive
    /// product elements `acc` of batch `l`, row `m` and columns `n` on, read and written in the fusion's tensors
    /// `ins` (its inputs) and `outs` (its outputs) at their `layouts`. Without regions, its outputs are the
    /// product's shape: each read or written where the product is moves by one vector when `aligned` (every tensor
    /// 16-byte aligned). With regions, each vector lies in one; each of its member tuples' values computes the
    /// outputs reading the region at their own index. The module also names each tensor's CuTe type and span
    /// (`TYPES`, `SPANS`, its inputs first: `INPUTS`) and the product's elements a vector (`VECTOR`).
    fn gemm(
        body: &Graph,
        dot: &Node,
        by_value: &[bool],
        layouts: &Layouts,
    ) -> Result<(String, String), String> {
        if !epilogue(body, dot) {
            return Err(format!(
                "{}: the CUDA GEMM runs no epilogue after this matmul",
                dot.primitive.name()
            ));
        }

        let product = body.type_of(dot.output);
        let shape = &product.shape;
        let rank = shape.len();
        let (rows, cols) = (shape[rank - 2], shape[rank - 1]);
        let lanes = lanes(product.dtype);
        let wide = layouts.spans.iter().any(|&n| n >= 1 << 31) || product.numel() >= 1 << 31;
        let at = |i: &str| match wide {
            true => format!("cutlass.Int64({i})"),
            false => i.to_string(),
        };

        let outputs = body.outputs();
        let mut code = String::new();
        let mut locals = 0;

        match gemm_regions(body, dot) {
            None => {
                let mut emitter = Emitter::new(body, by_value);
                emitter.strides = layouts.inputs.clone();
                emitter.indent = " ".repeat(8);
                emitter.values.insert((dot.output, "j".into()), "a".into());
                let mut stores = String::new();
                let mut written = Vec::new();

                for (k, &v) in outputs.iter().enumerate() {
                    let ty = body.type_of(v);
                    let value = stored(emitter.value(v, "j".into()), ty.dtype);

                    match gather_index("j", &ty.shape, &layouts.outputs[k]).as_str() {
                        "j" => {
                            written.push(k);
                            writeln!(stores, "        fout{k}[v] = {value}").unwrap();
                        }
                        at => writeln!(stores, "        out{k}[{at}] = {value}").unwrap(),
                    }
                }

                // The tensors read or written where the product is: a vector each, moved whole when aligned.
                let read: Vec<usize> = emitter.fragments.iter().copied().collect();
                writeln!(code, "    j0 = ({} * {rows} + m) * {cols} + n", at("l")).unwrap();
                writeln!(code, "    if cutlass.const_expr(aligned):").unwrap();

                for &k in &read {
                    writeln!(
                        code,
                        "        fin{k} = cute.make_rmem_tensor((VECTOR,), in{k}.element_type)\n        fin{k}.store(vectors(in{k}, VECTOR)[(None, j0 // VECTOR)].load())"
                    )
                    .unwrap();
                }

                for &k in &written {
                    writeln!(
                        code,
                        "        fout{k} = cute.make_rmem_tensor((VECTOR,), out{k}.element_type)"
                    )
                    .unwrap();
                }

                writeln!(code, "        pass\n    else:").unwrap();

                for &k in &read {
                    writeln!(code, "        fin{k} = in{k}").unwrap();
                }

                for &k in &written {
                    writeln!(code, "        fout{k} = out{k}").unwrap();
                }

                write!(
                    code,
                    "        pass\n\n    for lane in cutlass.range_constexpr(VECTOR):\n        j = j0 + lane\n        v = lane if cutlass.const_expr(aligned) else j\n        a = acc[lane]\n{}{stores}\n    if cutlass.const_expr(aligned):\n",
                    emitter.lines
                )
                .unwrap();

                for &k in &written {
                    writeln!(
                        code,
                        "        vectors(out{k}, VECTOR)[(None, j0 // VECTOR)].store(fout{k}.load())"
                    )
                    .unwrap();
                }

                writeln!(code, "        pass").unwrap();
            }
            Some(regions) => {
                // Each region's block, nested as `if n < end: … else: …` so a vector runs the one it lies in.
                let last = regions.len() - 1;

                for (r, c) in regions.iter().enumerate() {
                    let depth = 4 * (1 + r + usize::from(r < last));
                    let pad = " ".repeat(depth);
                    let members = c.members();
                    let (k, size) = (members.len(), c.size());

                    if r < last {
                        writeln!(
                            code,
                            "{}if n < {}:",
                            " ".repeat(depth - 4),
                            c.offset + c.width
                        )
                        .unwrap();
                    }

                    let mut emitter = Emitter::new(body, by_value);
                    emitter.strides = layouts.inputs.clone();
                    emitter.indent = " ".repeat(depth + 4);
                    emitter.locals = locals;
                    let index = format!("r{r}_j");

                    writeln!(
                        code,
                        "{pad}for t in cutlass.range_constexpr(VECTOR // {k}):\n{pad}    {index} = {} * {size} + (n - {}) // {k} + t",
                        at("m"),
                        c.offset
                    )
                    .unwrap();

                    for (member, &v) in members.iter().enumerate() {
                        let name = format!("r{r}_a{member}");
                        writeln!(code, "{pad}    {name} = acc[t * {k} + {member}]").unwrap();
                        emitter.values.insert((v, index.clone()), name);
                    }

                    let mut stores = String::new();

                    for (o, &v) in outputs.iter().enumerate() {
                        if !depends(body, v, &members) {
                            continue;
                        }

                        let ty = body.type_of(v);
                        let value = stored(emitter.value(v, index.clone()), ty.dtype);
                        let at = gather_index(&index, &ty.shape, &layouts.outputs[o]);
                        writeln!(stores, "{pad}    out{o}[{at}] = {value}").unwrap();
                    }

                    locals = emitter.locals;
                    write!(code, "{}{stores}", emitter.lines).unwrap();

                    if r < last {
                        writeln!(code, "{}else:", " ".repeat(depth - 4)).unwrap();
                    }
                }
            }
        }

        let tensors = || body.inputs().iter().chain(outputs);

        let unpack = |prefix: &str, count: usize| -> String {
            (0..count)
                .map(|k| format!("    {prefix}{k} = {prefix}s[{k}]\n"))
                .collect()
        };

        let source = format!(
            r#"import cutlass
import cutlass.cute as cute

from lumen.codegen.elementwise.cuda.kernel import FAST_MATH, vectors

# The math functions EPILOGUE calls: CuTe's, or their fast approximations under config.compiler.fast_math.
math = {math}

# The CuTe type each tensor is stored as (a bool's a byte) and the elements it spans: the fusion's inputs, then its
# outputs.
TYPES = ({types})
SPANS = ({spans})
INPUTS = {inputs}

# Elements of the product in a vector EPILOGUE takes.
VECTOR = {lanes}


@cute.jit
def EPILOGUE(ins, outs, acc, l, m, n, aligned: cutlass.Constexpr):
    """The fusion's outputs at the product's `VECTOR` elements `acc`, of batch `l`, row `m` and columns `n` on, its
    tensors `ins` and `outs` moved by vectors where they are read or written as the product when `aligned`."""
{unpack_ins}{unpack_outs}{code}"#,
            math = match crate::graph_compiler::config::config().fast_math {
                true => "FAST_MATH",
                false => "cute.math",
            },
            types = tensors()
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
            inputs = body.inputs().len(),
            unpack_ins = unpack("in", body.inputs().len()),
            unpack_outs = unpack("out", outputs.len()),
        );

        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);

        Ok((format!("cuda_gemm_{:016x}", hasher.finish()), source))
    }
}
