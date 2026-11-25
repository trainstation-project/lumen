//! Slices read only by GEMMs, turned into strided views of their operand (`Slice { view: true }`) instead of copies,
//! when the GEMM kernel can read the operand at the parent's strides. It runs after `fusion` and `wide_rows` and before
//! `alone`, so the lone layout ops `alone` turns into elementwise fusions no longer include these slices.
//!
//! Devices: it runs on each device implementing `StridedOperands` (its GEMM's rule for the operand layouts it reads in
//! place, `reads_slice`, in `<device>.rs`): MPS (`dot_general::mps::reads_strided`), Triton (any strides) and TPU
//! (`dot_general::tpu::slice_view`: slices of whole tiles, in its GEMM kernel). CPU and CUDA don't run it.
//!
//! The rule: a `Slice` of one value is marked a view when its output is not a graph output, its input is not itself a
//! slice already marked a view, and it has at least one reader, every reader being a `DotGeneral` whose rule accepts
//! each operand slot reading the slice (from its starts, at the sliced input's contiguous strides), or a GEMM fusion
//! where `fusion_reads_strided` holds: fusion input `k` read only by the body's GEMM dot (`codegen::kinds::gemm_dot`), as
//! exactly one of its operands, in a layout the rule accepts. MPS codegen calls `fusion_reads_strided` too
//! (`codegen::fusion::mps::fusion_reads_strided`, planning MPS steps).

#[cfg(lumen_mps_linked)]
mod mps;
mod tpu;
mod triton;

use crate::codegen::kinds::gemm_dot;
use crate::graph::{Graph, Primitive, TensorType, Var};
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::DeviceModule;
use crate::graph_compiler::passes::{CompilerPass, Module};
use crate::tensor::contiguous_strides;

/// Whether a step running a kernel (its fusion's name, else its dot's) reads its dot's (`Primitive`, its two operands'
/// types) operand `k` as the slice from `starts` (the last slice) of a value laid out with `strides`.
pub(crate) type ReadsSlice =
    fn(&str, &Primitive, [&TensorType; 2], usize, &[usize], &[usize]) -> bool;

/// Marks each slice `dot_views` accepts as a view.
pub(crate) fn place_views(graph: &mut Graph, reads: ReadsSlice) {
    for v in dot_views(graph, reads) {
        let node = graph.nodes.iter_mut().find(|n| n.output == v);

        if let Some(Primitive::Slice { view, .. }) = node.map(|n| &mut n.primitive) {
            *view = true;
        }
    }
}

/// Outputs of slices whose readers are all dots or GEMM fusions able to read the slice of the sliced input.
fn dot_views(graph: &Graph, reads: ReadsSlice) -> Vec<Var> {
    let nodes = graph.nodes();
    let mut views = Vec::new();

    for node in nodes {
        let (Primitive::Slice { start_indices, .. }, &[x]) =
            (&node.primitive, node.inputs.as_slice())
        else {
            continue;
        };

        let v = node.output;

        if graph.outputs().contains(&v) || views.contains(&x) {
            continue;
        }

        let strides = contiguous_strides(&graph.type_of(x).shape);
        let mut readers = nodes.iter().filter(|n| n.inputs.contains(&v)).peekable();
        let read = readers.peek().is_some();

        let in_place = readers.all(|n| match &n.primitive {
            Primitive::DotGeneral { .. } => {
                let operands = [graph.type_of(n.inputs[0]), graph.type_of(n.inputs[1])];
                let kernel = n.primitive.name();

                (0..2).all(|k| {
                    n.inputs[k] != v
                        || reads(kernel, &n.primitive, operands, k, &strides, start_indices)
                })
            }
            Primitive::Fusion { name, body, .. } => (0..n.inputs.len()).all(|k| {
                let read =
                    |dot: &Primitive, operands: [&TensorType; 2], side, strides: &[usize]| {
                        reads(name, dot, operands, side, strides, start_indices)
                    };

                n.inputs[k] != v || fusion_reads_strided(body, k, &strides, read)
            }),
            _ => false,
        });

        if read && in_place {
            views.push(v);
        }
    }

    views
}

/// Whether GEMM fusion input `k` is read only by the dot, as one operand, in a layout it can stride through.
pub(crate) fn fusion_reads_strided(
    body: &Graph,
    k: usize,
    strides: &[usize],
    reads: impl Fn(&Primitive, [&TensorType; 2], usize, &[usize]) -> bool,
) -> bool {
    let (Some(dot), Some(&v)) = (gemm_dot(body), body.inputs().get(k)) else {
        return false;
    };

    let side = dot.inputs.iter().position(|&u| u == v);

    let only = body
        .nodes()
        .iter()
        .all(|n| std::ptr::eq(n, dot) || !n.inputs.contains(&v));

    let operands = [body.type_of(dot.inputs[0]), body.type_of(dot.inputs[1])];

    match side {
        Some(side) if only && dot.inputs[0] != dot.inputs[1] => {
            reads(&dot.primitive, operands, side, strides)
        }
        _ => false,
    }
}

/// A backend's GEMM telling the `place_views` pass which operand layouts it reads in place: slices only such GEMMs read
/// become strided views instead of copies.
pub(crate) trait StridedOperands {
    /// Whether the GEMM reads `dot`'s operand `k` (of the two `operands`' types) laid out with `strides`.
    fn reads_strided(
        dot: &crate::graph::Primitive,
        operands: [&crate::graph::TensorType; 2],
        k: usize,
        strides: &[usize],
    ) -> bool;

    /// Whether a step running `kernel` (its fusion's name, else its dot's) reads `dot`'s operand `k` as the slice
    /// from `starts` of a value laid out with `strides`; by default, from any start in any kernel, as
    /// `reads_strided` accepts.
    fn reads_slice(
        _kernel: &str,
        dot: &crate::graph::Primitive,
        operands: [&crate::graph::TensorType; 2],
        k: usize,
        strides: &[usize],
        _starts: &[usize],
    ) -> bool {
        Self::reads_strided(dot, operands, k, strides)
    }
}

/// Compiler pass marking each slice read only by GEMMs that stride through it (`reads`, the backend's
/// matmul kernels' rule) a view.
pub(crate) struct PlaceViews {
    pub reads: ReadsSlice,
}

impl<D: StridedOperands> CompilerPass<DeviceModule<'_, D>> for PlaceViews {
    /// Always, reading strided operands as the device's GEMMs can.
    fn configured(_: &CompilerConfig) -> Option<Self> {
        Some(PlaceViews {
            reads: D::reads_slice,
        })
    }

    /// Pass name used in profiler ranges and dumps.
    fn name(&self) -> &'static str {
        "place_views"
    }

    /// Marks the slices `place_views` accepts.
    fn apply(&mut self, m: &mut DeviceModule<'_, D>) -> Result<(), String> {
        place_views(m.graph_mut(), self.reads);
        Ok(())
    }
}
