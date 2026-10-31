//! Triton code generation: the pieces every op's `@triton.jit` source shares (types, constants, the fill
//! kernel, template `BODY` functions), re-exporting the emitter and the load/store helpers.

use std::fmt::Write;
use std::hash::{DefaultHasher, Hash, Hasher};

use indoc::formatdoc;

use crate::graph::{Graph, Node, Primitive, Var};
use crate::tensor::dtype::dispatch_dtype;
use crate::tensor::storage::as_bytes;
use crate::{DType, Element, Scalar};

pub(crate) use crate::codegen::attention::triton::{
    ATTENTION_CONFIGS, AttentionConfig, untuned_attention,
};
pub(crate) use crate::codegen::dot_general::triton::row::{
    dot_row_fits, partials_epilogue, row_epilogue,
};
pub(crate) use crate::codegen::dot_general::triton::{
    CONFIGS, Config, UNTUNED, partial_tile, row_gemm_configs, untuned_row_gemm,
};
pub(crate) use crate::codegen::dot_general::triton::{
    EpilogueFlags, Layouts, gemm, gemm_dot, gemm_flags,
};
pub(crate) use crate::codegen::elementwise::triton::{
    LOOP_CONFIGS, LoopConfig, UNTUNED_LOOP, loop_configs, untuned_loop,
};
pub(crate) use crate::codegen::emitter::Emit;
pub(crate) use crate::codegen::emitter::triton::Emitter;
pub(crate) use crate::codegen::kinds::reduction_root;
pub(crate) use crate::codegen::reduction::triton::scan;
pub(crate) use crate::codegen::row::triton::{ROW_PARTIALS, row_count, row_groups, row_looped};
pub(crate) use crate::codegen::store::triton::{gather_index, load, store, stored_at};

/// The Triton code generator.
pub(crate) struct Triton;

/// Triton element type for `dtype`, or `None` when Triton can't hold it.
pub(crate) fn triton_type(dtype: DType) -> Option<&'static str> {
    use DType::*;

    Some(match dtype {
        Bool => "tl.int1",
        U8 => "tl.uint8",
        U16 => "tl.uint16",
        U32 => "tl.uint32",
        U64 => "tl.uint64",
        I8 => "tl.int8",
        I16 => "tl.int16",
        I32 => "tl.int32",
        I64 => "tl.int64",
        F16 => "tl.float16",
        BF16 => "tl.bfloat16",
        F32 => "tl.float32",
        F64 => "tl.float64",
        _ => return None,
    })
}

/// Whether `node` can be emitted inside a Triton fusion kernel: a supported primitive whose
/// inputs and output all have Triton dtypes.
pub(crate) fn fusible(graph: &Graph, node: &Node) -> bool {
    use Primitive::*;

    let op = matches!(
        node.primitive,
        Add | Sub
            | Mul
            | Div
            | Max
            | Eq
            | Lt
            | Neg
            | Exp
            | Log
            | Sqrt
            | Tanh
            | Logistic
            | Cast { .. }
            | Select
            | Unary(_)
            | Binary(_)
            | Reshape { .. }
            | BroadcastInDim { .. }
            | Transpose { .. }
            | Slice { .. }
            | Full { .. }
            | Iota { .. }
            | RandomBits { .. }
            | ReduceSum { .. }
            | ReduceMax { .. }
            | DynamicSlice { .. }
            | Gather { .. }
            | Concatenate { .. }
    );

    let typed = |v: &Var| triton_type(graph.type_of(*v).dtype).is_some();
    op && node.inputs.iter().chain([&node.output]).all(typed)
}

/// Kernel parameter list: inputs (`in{k}`), optional `extra`, `out`, `n`, a `LOAD{k}` helper per
/// by-reference input, `stores` store helpers (`STORE`, then `STORE{k}` from 1) and `BLOCK_SIZE`,
/// one per line.
pub(crate) fn params(by_value: &[bool], extra: Option<&str>, stores: usize) -> String {
    let inputs = (0..by_value.len())
        .map(|k| format!("in{k}"))
        .chain(extra.map(String::from));

    let loads = (0..by_value.len())
        .filter(|&k| !by_value[k])
        .map(|k| format!("LOAD{k}: tl.constexpr"));

    inputs
        .chain(["out".into(), "n".into()])
        .chain(loads)
        .chain((0..stores).map(|k| match k {
            0 => "STORE: tl.constexpr".into(),
            k => format!("STORE{k}: tl.constexpr"),
        }))
        .chain(["BLOCK_SIZE: tl.constexpr".into()])
        .map(|p| format!("    {p},\n"))
        .collect()
}

/// Generates a kernel filling `out` with a constant passed as its raw bits (int64) at launch.
pub(crate) fn fill(dtype: DType) -> (String, String) {
    let value = match dtype {
        DType::Bool => "bits != 0".to_string(),
        _ => format!(
            "tl.full([], bits, {}).to({}, bitcast=True)",
            raw_type(dtype),
            triton_type(dtype).expect("a Triton dtype")
        ),
    };

    let source = formatdoc!(
        "
        @triton.jit
        def NAME(bits, out, n, STORE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            j = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            STORE(out, j, {value}, j < n)
        "
    );

    named("fill", source)
}

/// A tensor's shape and strides, in elements.
pub(crate) type Owned = (Vec<usize>, Vec<usize>);

/// The layouts a template kernel's `BODY` is generated for: each input's (`None` when it is passed by
/// value), then each output's.
pub(crate) struct Tensors {
    pub(crate) inputs: Vec<Option<Owned>>,
    pub(crate) outputs: Vec<Owned>,
}

/// A template kernel's generated `BODY(EXTRA, {params})` running `code`: it binds the step's inputs `in{k}`
/// and outputs `out` and `out{k}` from `EXTRA` (the inputs, then the outputs after the first, then the
/// first), and defines the `LOAD{k}` and `STORE`/`STORE{k}` helpers `code` calls for their `layouts`.
/// Returns `(name, source)`, the name `kind` plus a hash of the source.
pub(crate) fn body_function(
    kind: &str,
    layouts: &Tensors,
    params: &str,
    code: &str,
) -> (String, String) {
    let (inputs, outputs) = (layouts.inputs.len(), layouts.outputs.len());
    let mut helpers = String::new();

    for (k, layout) in layouts.inputs.iter().enumerate() {
        if let Some((shape, strides)) = layout {
            let (name, source) = load(shape, strides);
            helpers += &source.replace(&format!("def {name}("), &format!("def LOAD{k}("));
            helpers += "\n";
        }
    }

    for (k, (shape, strides)) in layouts.outputs.iter().enumerate() {
        let (name, source) = store(shape, strides);

        let param = match k {
            0 => "STORE".to_string(),
            k => format!("STORE{k}"),
        };

        helpers += &source.replace(&format!("def {name}("), &format!("def {param}("));
        helpers += "\n";
    }

    let mut bind: String = (0..inputs)
        .map(|k| format!("    in{k} = EXTRA[{k}]\n"))
        .collect();

    for k in 1..outputs {
        writeln!(bind, "    out{k} = EXTRA[{}]", inputs + k - 1).unwrap();
    }

    writeln!(bind, "    out = EXTRA[{}]", inputs + outputs - 1).unwrap();

    named(
        kind,
        format!("{helpers}@triton.jit\ndef NAME(EXTRA, {params}):\n{bind}{code}"),
    )
}

/// Bit pattern of `value` cast to `dtype`, zero-extended into a `u64` (little-endian bytes).
pub(crate) fn bits(dtype: DType, value: Scalar) -> u64 {
    let bytes = dispatch_dtype!(dtype, T => as_bytes(std::slice::from_ref(&T::from_scalar(value))).to_vec());

    bytes
        .iter()
        .rev()
        .fold(0u64, |acc, &b| (acc << 8) | b as u64)
}

/// Substitutes `NAME` in `source` with `kind` plus a hash of the source, so identical kernels share
/// a name and compile cache entry. Returns `(name, source)`.
pub(crate) fn named(kind: &str, source: String) -> (String, String) {
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    let name = format!("{kind}_{:016x}", hasher.finish());
    (name.clone(), source.replace("NAME", &name))
}

/// Unsigned Triton type as wide as `dtype`, used to bitcast constants.
fn raw_type(dtype: DType) -> &'static str {
    match dtype.size_of() {
        1 => "tl.uint8",
        2 => "tl.uint16",
        4 => "tl.uint32",
        _ => "tl.uint64",
    }
}

/// Triton expression for the scalar `value` of `dtype`, built bit-exactly from its raw bits.
pub(crate) fn constant(dtype: DType, value: Scalar) -> String {
    let (bits, raw) = (bits(dtype, value), raw_type(dtype));

    match dtype {
        DType::Bool => format!("tl.full([], {}, tl.int1)", (bits != 0) as u8),
        _ => format!(
            "tl.full([], {bits:#x}, {raw}).to({}, bitcast=True)",
            triton_type(dtype).expect("a Triton dtype")
        ),
    }
}
