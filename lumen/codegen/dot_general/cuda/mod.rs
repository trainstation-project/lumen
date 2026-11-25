//! The CUDA (CuTe DSL) GEMM: which matmuls it runs, its launch configs, and the epilogue it runs fused after one. The
//! kernel and its launcher are Python (`ampere.py`, `launch.py`).
//!
//! The CUDA (CuTe DSL) GEMM kernel: its autotuning candidates, the tile it launches with untuned, and its launch on
//! matmul steps and GEMM fusions (their epilogue generated for the tensors' layouts).

use std::fmt::Write;
use std::hash::{DefaultHasher, Hash, Hasher};

use crate::DType;
use crate::Tensor;
use crate::codegen::devices::cuda::{Cuda, cute_type, unsigned};
use crate::codegen::dot_general::GemmCodegen;
use crate::codegen::elementwise::cuda::Layouts;
use crate::codegen::emitter::Emit;
use crate::codegen::emitter::cuda::Emitter;
use crate::codegen::kernel::Kernel;
use crate::codegen::kinds::gemm_dot;
use crate::codegen::regions::{Region, gemm_expanded, gemm_regions};
use crate::codegen::store::triton::gather_index;
use crate::graph::{Graph, Node, Primitive, Var};
use crate::plan::CudaStep;
use crate::plan::step::Tensors;
use crate::runtime::kernel::cuda::{GEMM, Output};

/// Bytes of the vectors the GEMM's epilogue moves the product by, from shared memory to its outputs.
const VECTOR_BYTES: usize = 16;

/// Whether `node` is a matmul the GEMM runs: float16 or bfloat16 operands of rank 2, or 3 with a leading batch dim,
/// contracting one dim each, accumulated in float32 and written as float16, bfloat16 or float32, of any extents (its
/// tiles move the widest vectors each tensor's address, contiguous extent and strides allow, 16 bytes down to one
/// element).
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
        && extents.into_iter().all(|&d| d > 0)
}

/// Elements of `dot`'s product (of `dtype`) in one of the epilogue's vectors.
fn lanes(dtype: DType) -> usize {
    VECTOR_BYTES / dtype.size_of()
}

/// Whether the GEMM runs an epilogue after `dot` in `graph`: a matmul it runs whose readers, when they split its
/// product into column regions, each take whole vectors of it (the epilogue reads the product a 16-byte vector at a
/// time): regions along its last axis, each a plain slice or the interleaved members of a reshape (as many as divide a
/// vector), not the two halves of the product (which would need both halves' tiles in one CTA), and a product whose
/// rows aren't whole vectors has none (its epilogue skips the lanes past each row instead).
pub(crate) fn epilogue(graph: &Graph, dot: &Node) -> bool {
    if !matmul(graph, dot) {
        return false;
    }

    let product = graph.type_of(dot.output);
    let (rank, lanes) = (product.shape.len(), lanes(product.dtype));

    if !product.shape[rank - 1].is_multiple_of(lanes) && gemm_regions(graph, dot).is_some() {
        return false;
    }

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
    /// 16-byte aligned) and the product's rows are whole vectors; a row that isn't skips the last vector's lanes past
    /// it. With regions, each vector lies in one; each of its member tuples' values computes the
    /// outputs reading the region at their own index. An expansion (the product's shape twice, concatenated) stores
    /// its two halves' values into their slots of it. The module also names each tensor's CuTe type and span
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

                    // An expansion (as on Triton) stores its two dot-shaped operands into their slots of the
                    // concatenation, `width` elements apart in runs of `width`.
                    if let (0, Some((parts, _))) = (k, gemm_expanded(body, dot)) {
                        let concat = body.nodes().iter().find(|n| n.output == v);

                        let Some(Primitive::Concatenate { dimension, .. }) =
                            concat.map(|n| &n.primitive)
                        else {
                            unreachable!("an expansion's concatenation")
                        };

                        let width: usize = shape[(*dimension).min(rank)..].iter().product();

                        writeln!(
                            stores,
                            "        jx0 = (j // {width}) * {} + j % {width}\n        jx1 = jx0 + {width}",
                            2 * width
                        )
                        .unwrap();

                        for (part, jx) in parts.into_iter().zip(["jx0", "jx1"]) {
                            let value = stored(emitter.value(part, "j".into()), ty.dtype);
                            let at = gather_index(jx, &ty.shape, &layouts.outputs[k]);
                            writeln!(stores, "        out{k}[{at}] = {value}").unwrap();
                        }

                        continue;
                    }

                    let value = stored(emitter.value(v, "j".into()), ty.dtype);

                    match gather_index("j", &ty.shape, &layouts.outputs[k]).as_str() {
                        "j" => {
                            written.push(k);
                            writeln!(stores, "        fout{k}[v] = {value}").unwrap();
                        }
                        at => writeln!(stores, "        out{k}[{at}] = {value}").unwrap(),
                    }
                }

                // A row of the product not whole vectors: its rows don't start on a vector, so nothing moves by vectors,
                // and the last vector's lanes past the row are skipped.
                let partial = !cols.is_multiple_of(lanes);

                let moved = match partial {
                    true => "cutlass.const_expr(False)",
                    false => "cutlass.const_expr(aligned)",
                };

                // The tensors read or written where the product is: a vector each, moved whole when aligned.
                let read: Vec<usize> = emitter.fragments.iter().copied().collect();
                writeln!(code, "    j0 = ({} * {rows} + m) * {cols} + n", at("l")).unwrap();
                writeln!(code, "    if {moved}:").unwrap();

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

                let lane = format!("{}{stores}", emitter.lines);

                let lane = match partial {
                    true => {
                        let indented: String = lane.lines().map(|l| format!("    {l}\n")).collect();
                        format!("        if n + lane < {cols}:\n{indented}")
                    }
                    false => lane,
                };

                write!(
                    code,
                    "        pass\n\n    for lane in cutlass.range_constexpr(VECTOR):\n        j = j0 + lane\n        v = lane if {moved} else j\n        a = acc[lane]\n{lane}\n    if {moved}:\n"
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

/// CTA tile, warps and pipeline stages of one Ampere GEMM (`ampere.py`'s `AmpereGemm`) candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Output rows per CTA.
    pub tile_m: usize,
    /// Output columns per CTA.
    pub tile_n: usize,
    /// Reduction-axis slice each pipeline stage holds.
    pub tile_k: usize,
    /// Warps per CTA (4 or 8), laid out 2 along M by `warps / 2` along N.
    pub warps: usize,
    /// A/B pipeline stages in shared memory, 0 for as many as fit at the CTAs per SM `warps` allows (two with 4
    /// warps, one with 8); a count only one CTA per SM fits runs one.
    pub stages: usize,
}

/// GEMM candidates the autotuner benchmarks, largest tiles first, each with 3 to 5 stages where they fit (a 64-deep
/// k-tile's A/B stage is twice a 32-deep one's), and the 32-deep ones also with as many as fit (6 to 9, the stages
/// they ran with before stages were tuned). On an H100 (bf16) 128 x 256 of 64 x 64 warp tiles leads from 4096³
/// up (457 vs 405 TFLOPS for 128 x 128), and the 64-row tiles at 128 x 4096 x 4096 (142 vs 70), where bigger tiles
/// leave SMs idle; on an A100 128 x 128 x 64 at 1000 x 4096 x 1000, where 64 CTAs leave SMs idle (80 vs 86 us for
/// 128 x 128 x 32).
pub(crate) const CONFIGS: [Config; 24] = [
    Config {
        tile_m: 128,
        tile_n: 256,
        tile_k: 32,
        warps: 8,
        stages: 3,
    },
    Config {
        tile_m: 128,
        tile_n: 256,
        tile_k: 32,
        warps: 8,
        stages: 4,
    },
    Config {
        tile_m: 128,
        tile_n: 256,
        tile_k: 32,
        warps: 8,
        stages: 5,
    },
    Config {
        tile_m: 128,
        tile_n: 256,
        tile_k: 32,
        warps: 8,
        stages: 0,
    },
    Config {
        tile_m: 128,
        tile_n: 256,
        tile_k: 64,
        warps: 8,
        stages: 3,
    },
    Config {
        tile_m: 128,
        tile_n: 128,
        tile_k: 32,
        warps: 4,
        stages: 3,
    },
    Config {
        tile_m: 128,
        tile_n: 128,
        tile_k: 32,
        warps: 4,
        stages: 4,
    },
    Config {
        tile_m: 128,
        tile_n: 128,
        tile_k: 32,
        warps: 4,
        stages: 5,
    },
    Config {
        tile_m: 128,
        tile_n: 128,
        tile_k: 64,
        warps: 4,
        stages: 3,
    },
    Config {
        tile_m: 128,
        tile_n: 128,
        tile_k: 64,
        warps: 4,
        stages: 4,
    },
    Config {
        tile_m: 128,
        tile_n: 64,
        tile_k: 32,
        warps: 4,
        stages: 3,
    },
    Config {
        tile_m: 128,
        tile_n: 64,
        tile_k: 32,
        warps: 4,
        stages: 4,
    },
    Config {
        tile_m: 128,
        tile_n: 64,
        tile_k: 32,
        warps: 4,
        stages: 5,
    },
    Config {
        tile_m: 128,
        tile_n: 64,
        tile_k: 32,
        warps: 4,
        stages: 0,
    },
    Config {
        tile_m: 64,
        tile_n: 128,
        tile_k: 32,
        warps: 4,
        stages: 3,
    },
    Config {
        tile_m: 64,
        tile_n: 128,
        tile_k: 32,
        warps: 4,
        stages: 4,
    },
    Config {
        tile_m: 64,
        tile_n: 128,
        tile_k: 32,
        warps: 4,
        stages: 5,
    },
    Config {
        tile_m: 64,
        tile_n: 128,
        tile_k: 32,
        warps: 4,
        stages: 0,
    },
    Config {
        tile_m: 64,
        tile_n: 64,
        tile_k: 32,
        warps: 4,
        stages: 3,
    },
    Config {
        tile_m: 64,
        tile_n: 64,
        tile_k: 32,
        warps: 4,
        stages: 4,
    },
    Config {
        tile_m: 64,
        tile_n: 64,
        tile_k: 32,
        warps: 4,
        stages: 5,
    },
    Config {
        tile_m: 64,
        tile_n: 64,
        tile_k: 32,
        warps: 4,
        stages: 0,
    },
    Config {
        tile_m: 64,
        tile_n: 64,
        tile_k: 64,
        warps: 4,
        stages: 3,
    },
    Config {
        tile_m: 64,
        tile_n: 64,
        tile_k: 64,
        warps: 4,
        stages: 4,
    },
];

/// Fewest multiply-adds a matmul is autotuned at (1024³): below, the GEMM takes tens of microseconds whatever its
/// tile, while timing every candidate costs a CuTe compile each.
pub(crate) const TUNED_MACS: usize = 1 << 30;

/// The untuned tile for outputs it fills the GPU with: 128 x 256, 8 warps, as many stages as fit.
const WIDE: Config = Config {
    tile_m: 128,
    tile_n: 256,
    tile_k: 32,
    warps: 8,
    stages: 0,
};

/// The untuned tile otherwise: 128 x 128, 4 warps, two CTAs an SM.
const NARROW: Config = Config {
    tile_m: 128,
    tile_n: 128,
    tile_k: 32,
    warps: 4,
    stages: 0,
};

/// The config used untuned for a `batch` x (`m`, `n`) output on a GPU of `sms` SMs: the wide tile once its CTAs
/// alone make a wave, else the narrow one so smaller outputs still spread over every SM.
#[cfg_attr(not(all(feature = "python", lumen_cuda_linked)), allow(dead_code))]
pub(crate) fn untuned(batch: usize, m: usize, n: usize, sms: usize) -> Config {
    let tiles = batch * m.div_ceil(WIDE.tile_m) * n.div_ceil(WIDE.tile_n);

    match tiles >= sms {
        true => WIDE,
        false => NARROW,
    }
}

#[cfg(feature = "python")]
crate::autotune::python::config_class! {
    /// `lumen.autotune.CudaGemmConfig`: a tile of the CUDA backend's SM80 GEMM, one of its autotuning candidates or
    /// any other (`stages` 0 for as many as fit).
    PyCudaGemmConfig(Config) = "CudaGemmConfig" {
        tile_m: usize,
        tile_n: usize,
        tile_k: usize,
        warps: usize,
        stages: usize = 0,
    }

    /// Every config the autotuner times, in its order.
    #[staticmethod]
    fn candidates() -> Vec<Self> {
        CONFIGS.iter().copied().map(PyCudaGemmConfig).collect()
    }
}

/// The Ampere GEMM, running a matmul step or a GEMM fusion (its epilogue writing the fusion's outputs from the
/// product).
pub(crate) struct Gemm;

impl Kernel for Gemm {
    type Step = CudaStep;
    type Buffers = Tensors;
    type Config = Config;

    /// Every GEMM tile for a step of at least `TUNED_MACS` multiply-adds, none otherwise.
    fn configs(&self, step: &CudaStep) -> Vec<Config> {
        match macs(step) {
            Some(macs) if macs >= TUNED_MACS => CONFIGS.to_vec(),
            _ => Vec::new(),
        }
    }

    /// Runs the matmul on the GEMM with `config`'s tile, A as (.., m, k) and B as (.., k, n) whichever of their dims
    /// contract; a GEMM fusion's epilogue, generated for its tensors' layouts (kept in the step's memo), writes its
    /// outputs from the product.
    fn run(
        &self,
        step: &CudaStep,
        config: Option<&Config>,
        (inputs, out, outs): &Tensors,
    ) -> Result<(), String> {
        // A GEMM fusion's matmul, and the body it runs in.
        let fused = match &step.primitive {
            Primitive::Fusion { body, .. } => gemm_dot(body).map(|dot| (body, dot)),
            _ => None,
        };

        let (dot, [lhs, rhs]) = match (&step.primitive, fused) {
            (p @ Primitive::DotGeneral { .. }, _) => (p, [0, 1]),
            (_, Some((body, dot))) => {
                let position = |v| {
                    body.inputs()
                        .iter()
                        .position(|&i| i == v)
                        .expect("an input")
                };
                (
                    &dot.primitive,
                    [position(dot.inputs[0]), position(dot.inputs[1])],
                )
            }
            _ => unreachable!("a matmul step or a GEMM fusion"),
        };

        let Primitive::DotGeneral {
            lhs_contracting,
            rhs_contracting,
            ..
        } = dot
        else {
            unreachable!("a matmul")
        };

        let gemm = GEMM.get().ok_or(
            "the CUDA backend's GEMM runs from Python (lumen.codegen.dot_general.cuda): none is installed",
        )?;

        let outputs: Vec<Tensor> = std::iter::once(out).chain(outs).cloned().collect();

        if outputs.iter().all(|t| t.numel() == 0) {
            return Ok(());
        }

        let rank = inputs[lhs].ndim();

        let a = match lhs_contracting[0] == rank - 1 {
            true => inputs[lhs].clone(),
            false => inputs[lhs].transpose(rank - 2, rank - 1),
        };

        let b = match rhs_contracting[0] == rank - 2 {
            true => inputs[rhs].clone(),
            false => inputs[rhs].transpose(rank - 2, rank - 1),
        };

        let Some((body, dot)) = fused else {
            return gemm(&a, &b, Output::Product(out), config);
        };

        let tensors: Vec<Tensor> = inputs.iter().chain(&outputs).map(Tensor::flat).collect();
        let (ins, outs) = tensors.split_at(inputs.len());

        let layouts = Layouts {
            inputs: inputs.iter().map(|t| t.strides().to_vec()).collect(),
            outputs: outputs.iter().map(|t| t.strides().to_vec()).collect(),
            spans: tensors.iter().map(Tensor::numel).collect(),
        };

        let code = crate::codegen::elementwise::cuda::generated(step, layouts)?;

        let output = Output::Epilogue {
            code: code.as_ref(),
            dtype: body.type_of(dot.output).dtype,
            inputs: ins,
            outputs: outs,
        };

        gemm(&a, &b, output, config)
    }
}

/// The multiply-adds of the matmul `step` runs on the GEMM: a matmul step's, or a GEMM fusion's.
fn macs(step: &CudaStep) -> Option<usize> {
    let (dot, lhs, product) = match &step.primitive {
        p @ Primitive::DotGeneral { .. } => (p, &step.inputs[0].1, step.output.1.numel()),
        Primitive::Fusion { body, .. } => {
            let dot = gemm_dot(body)?;
            (
                &dot.primitive,
                body.type_of(dot.inputs[0]),
                body.type_of(dot.output).numel(),
            )
        }
        _ => return None,
    };

    let Primitive::DotGeneral {
        lhs_contracting, ..
    } = dot
    else {
        return None;
    };

    Some(product * lhs.shape[lhs_contracting[0]])
}
