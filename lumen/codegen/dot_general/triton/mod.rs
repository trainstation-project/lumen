//! Triton GEMMs: operand prologues, the output epilogue, and fused elementwise epilogues that consume
//! column regions of the GEMM output.
//!
//! Triton GEMM autotuning candidates and launch defaults, plain and with a row epilogue.
//!
//! Autotuning the Triton GEMM: its configs for GEMM steps, whole-row tiles for a row epilogue.

pub(crate) mod cross_entropy;
pub(crate) mod launch;
pub(crate) mod row;

use crate::codegen::dot_general::triton::launch::{DOT_ROW, is_gemm};
use crate::plan::TritonStep;
use crate::plan::step::Tensors;
use std::fmt::Write;

use indoc::formatdoc;

use crate::codegen::devices::triton::{Triton, indented, named, triton_type};
use crate::codegen::dot_general::GemmCodegen;
use crate::codegen::emitter::Emit;
use crate::codegen::emitter::triton::Emitter;
use crate::codegen::regions::{Columns, Region, gemm_expanded, gemm_regions};
use crate::codegen::store::triton::{gather_index, load, stored_at};
use crate::graph::{Graph, Node, Primitive, TensorType, Var};
use crate::tensor::contiguous_strides;

/// Generated pieces of one GEMM kernel.
pub(crate) struct GemmCode {
    /// `[batch, m, k, n]` extents of the dot.
    pub(crate) sizes: [usize; 4],
    /// `(name, source)` of the jit function loading an A tile at `(l, m, k)`.
    pub(crate) prologue_a: (String, String),
    /// `(name, source)` of the jit function loading a B tile at `(l, k, n)`.
    pub(crate) prologue_b: (String, String),
    /// `(name, source)` of the jit function storing the output tile.
    pub(crate) epilogue: (String, String),
}

/// A tensor's `(shape, strides)` in elements.
type Layout<'a> = (&'a [usize], &'a [usize]);

/// A tensor's shape and strides, owned.
type Owned = (Vec<usize>, Vec<usize>);

/// The layouts a GEMM fusion's code is generated for: its dot's operands', its inputs' and its outputs'.
pub(crate) struct Layouts {
    pub(crate) lhs: Owned,
    pub(crate) rhs: Owned,
    pub(crate) inputs: Vec<Owned>,
    pub(crate) outputs: Vec<Owned>,
}

/// An owned layout, borrowed.
fn view((shape, strides): &Owned) -> Layout<'_> {
    (shape, strides)
}

/// Triton GEMMs: the fixed GEMM kernel reading the dot's operands through generated prologues for their
/// layouts, the product computed into a contiguous tile and stored by a generated epilogue applying the
/// fusion's elementwise ops.
impl GemmCodegen for Triton {
    /// The prologues and the epilogue.
    type Code = GemmCode;
    type Options = Layouts;

    /// The prologues for the operands' layouts and the epilogue writing the fusion's outputs at theirs.
    fn gemm(
        body: &Graph,
        dot: &Node,
        by_value: &[bool],
        layouts: &Layouts,
    ) -> Result<GemmCode, String> {
        let product = body.type_of(dot.output);
        let contiguous = contiguous_strides(&product.shape);

        let mut code = gemm(
            &dot.primitive,
            view(&layouts.lhs),
            view(&layouts.rhs),
            (&product.shape, &contiguous),
        );

        let [_, rows, _, cols] = code.sizes;
        let inputs: Vec<Layout> = layouts.inputs.iter().map(view).collect();
        let outputs: Vec<Layout> = layouts.outputs.iter().map(view).collect();
        code.epilogue = epilogue(body, by_value, dot, &inputs, &outputs, [rows, cols]);
        Ok(code)
    }
}

/// Lowers a `dot_general` with arbitrary batch/free/contracting dims and strided operands to a
/// `[l, m, k] x [l, k, n]` GEMM by generating strided load and store helpers.
pub(crate) fn gemm(dot: &Primitive, lhs: Layout, rhs: Layout, out: Layout) -> GemmCode {
    let Primitive::DotGeneral {
        lhs_contracting,
        rhs_contracting,
        lhs_batch,
        rhs_batch,
        ..
    } = dot
    else {
        unreachable!("a dot_general")
    };

    // Dims that are neither contracting nor batch, in operand order; they flatten into m or n.
    let free = |rank: usize, contracting: &[usize], batch: &[usize]| -> Vec<usize> {
        (0..rank)
            .filter(|d| !contracting.contains(d) && !batch.contains(d))
            .collect()
    };

    let lhs_free = free(lhs.0.len(), lhs_contracting, lhs_batch);
    let rhs_free = free(rhs.0.len(), rhs_contracting, rhs_batch);
    let (batch, rows) = (lhs_batch.len(), lhs_free.len());
    // The output is laid out as batch dims, then lhs free dims, then rhs free dims.
    let out_dims: Vec<usize> = (0..out.0.len()).collect();

    let (out_batch, out_rows, out_cols) = (
        &out_dims[..batch],
        &out_dims[batch..batch + rows],
        &out_dims[batch + rows..],
    );

    let size = |shape: &[usize], dims: &[usize]| dims.iter().map(|&d| shape[d]).product();

    // Flattened index offset of each logical coordinate (`l`, `m`/`n`, `k`) through the operand's strides.
    let address = |terms: [(&str, Layout, &[usize]); 3]| -> String {
        let offsets: Vec<String> = terms
            .iter()
            .filter_map(|&(idx, (shape, strides), dims)| {
                let shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
                let strides: Vec<usize> = dims.iter().map(|&d| strides[d]).collect();

                // A coordinate of one element is read at 0, its own index there; a broadcast one keeps
                // its tile's shape with zeros. The batch `l` is a scalar with no shape to keep, so a zero
                // offset is left out: added, it hides the pointer's alignment from Triton (a 4096
                // bfloat16 GEMM's loads are no longer vectorized: 744 vs 201 µs on an H100).
                match gather_index(idx, &shape, &strides).as_str() {
                    "0" if idx == "l" => None,
                    "0" if shape.iter().product::<usize>() == 1 => Some(idx.to_string()),
                    "0" => Some(format!("tl.zeros_like({idx})")),
                    offset => Some(offset.to_string()),
                }
            })
            .collect();
        offsets.join(" + ")
    };

    let prologue = |rows: &str, cols: &str, at: String| {
        named(
            "prologue",
            formatdoc!(
                "
                @triton.jit
                def NAME(ptr, l, {rows}, {cols}, mask):
                    return tl.load(ptr + {at}, mask=mask, other=0)
                "
            ),
        )
    };

    let a = address([
        ("l", lhs, lhs_batch),
        ("m", lhs, &lhs_free),
        ("k", lhs, lhs_contracting),
    ]);

    let b = address([
        ("l", rhs, rhs_batch),
        ("k", rhs, rhs_contracting),
        ("n", rhs, &rhs_free),
    ]);

    let d = address([
        ("l", out, out_batch),
        ("m", out, out_rows),
        ("n", out, out_cols),
    ]);

    GemmCode {
        sizes: [
            size(lhs.0, lhs_batch),
            size(lhs.0, &lhs_free),
            size(lhs.0, lhs_contracting),
            size(rhs.0, &rhs_free),
        ],
        prologue_a: prologue("m", "k", a),
        prologue_b: prologue("k", "n", b),
        epilogue: named(
            "epilogue",
            formatdoc!(
                "
                @triton.jit
                def NAME(ptr, EXTRA, l, m, n, value, mask):
                    tl.store(ptr + {d}, value.to(ptr.dtype.element_ty), mask=mask)
                "
            ),
        ),
    }
}

pub(crate) use crate::codegen::kinds::gemm_dot;

/// How the GEMM kernel shapes its accumulator for a fused epilogue.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct EpilogueFlags {
    /// Region members are interleaved along the axis rather than laid out as contiguous parts.
    pub(crate) interleaved: bool,
    /// Number of separate accumulators (contiguous parts) the kernel produces; 1 for a single one.
    pub(crate) group: usize,
    /// The regions split the row axis instead of the last (column) axis.
    pub(crate) rows: bool,
}

impl EpilogueFlags {
    /// Whether the kernel keeps one accumulator per contiguous part of the output.
    fn halves(&self) -> bool {
        self.group > 1 && !self.interleaved
    }
}

/// Chooses the accumulator layout: one accumulator per member when a single group region reads
/// contiguous parts of the dot output, otherwise one interleaved accumulator.
pub(crate) fn gemm_flags(body: &Graph, dot: &Node) -> EpilogueFlags {
    let rank = body.type_of(dot.output).shape.len();

    let Some(regions) = gemm_regions(body, dot) else {
        return EpilogueFlags {
            group: 1,
            ..EpilogueFlags::default()
        };
    };

    let halves =
        |c: &Columns| matches!(&c.region, Region::Group(m) if m[0].inputs[0] == dot.output);

    match regions.as_slice() {
        [c] if halves(c) => EpilogueFlags {
            interleaved: false,
            group: c.k(),
            rows: c.axis + 1 != rank,
        },
        _ => EpilogueFlags {
            interleaved: true,
            group: 1,
            rows: regions[0].axis + 1 != rank,
        },
    }
}

/// The `dynamic_update_slice` making `v` in `body` from an operand that is a body input, if any: an update an
/// epilogue writes into the operand's slot.
pub(crate) fn updated(body: &Graph, v: Var) -> Option<&Node> {
    body.nodes().iter().find(|n| {
        n.output == v
            && matches!(n.primitive, Primitive::DynamicUpdateSlice)
            && body.inputs().contains(&n.inputs[0])
    })
}

/// The value `update` writes at the region's flat index `j` (of its update operand) and where in its operand:
/// `j`'s coordinates plus the start indices, each clamped so the update stays in bounds, as a flat index of the
/// operand.
fn slot(emitter: &mut Emitter, body: &Graph, update: &Node) -> (String, String) {
    let (x, written) = (
        body.type_of(update.inputs[0]),
        body.type_of(update.inputs[1]),
    );
    let (spans, strides) = (
        contiguous_strides(&written.shape),
        contiguous_strides(&x.shape),
    );
    let value = emitter.value(update.inputs[1], "j".into());

    let terms: Vec<String> = (0..x.shape.len())
        .filter(|&d| x.shape[d] != 1)
        .map(|d| {
            let start = emitter.value(update.inputs[2 + d], "0".into());
            let most = x.shape[d] - written.shape[d];

            format!(
                "((j // {}) % {} + tl.minimum(tl.maximum(({start}).to(tl.int64), 0), {most})) * {}",
                spans[d], written.shape[d], strides[d]
            )
        })
        .collect();

    let at = match terms.is_empty() {
        true => "j".to_string(),
        false => terms.join(" + "),
    };

    (value, at)
}

/// Whether `v` is one of `on` or is computed from any of them in `body`.
fn depends(body: &Graph, v: Var, on: &[Var]) -> bool {
    let producer = body.nodes().iter().find(|n| n.output == v);
    on.contains(&v) || producer.is_some_and(|n| n.inputs.iter().any(|&u| depends(body, u, on)))
}

/// Emits code splitting tile `x` into `k` (a power of two) interleaved parts along columns or rows
/// via reshape and repeated `tl.split`; returns the part names ordered by member index.
fn split(code: &mut String, x: &str, k: usize, columns: bool, name: &str) -> Vec<String> {
    let bits = k.ilog2() as usize;
    let twos = vec!["2"; bits].join(", ");

    match columns {
        true => writeln!(
            code,
            "    {name} = tl.reshape({x}, ({x}.shape[0], {x}.shape[1] // {k}, {twos}))"
        ),
        // Rows: move the k-sized split axes after the column axis so `tl.split` peels them off.
        false => {
            let order: Vec<String> = std::iter::once(0)
                .chain(std::iter::once(bits + 1))
                .chain(1..=bits)
                .map(|d| d.to_string())
                .collect();

            writeln!(
                code,
                "    {name} = tl.permute(tl.reshape({x}, ({x}.shape[0] // {k}, {twos}, {x}.shape[1])), ({}))",
                order.join(", ")
            )
        }
    }
    .unwrap();

    // Each split level halves every part; the bit set at `level` becomes part of the member index.
    let mut parts = vec![(name.to_string(), 0)];

    for level in 0..bits {
        parts = parts
            .into_iter()
            .flat_map(|(p, member)| {
                writeln!(code, "    {p}_0, {p}_1 = tl.split({p})").unwrap();

                [
                    (format!("{p}_0"), member),
                    (format!("{p}_1"), member + (1 << level)),
                ]
            })
            .collect();
    }

    parts.sort_by_key(|&(_, member)| member);
    parts.into_iter().map(|(p, _)| p).collect()
}

/// Generates the GEMM epilogue jit function that computes the fused body's outputs from the
/// accumulator tile `value` and stores them, one guarded block per output column region.
pub(crate) fn epilogue(
    body: &Graph,
    by_value: &[bool],
    dot: &Node,
    inputs: &[Layout],
    outputs: &[Layout],
    [rows, cols]: [usize; 2],
) -> (String, String) {
    let loads: String = inputs
        .iter()
        .enumerate()
        .filter(|&(k, _)| !by_value[k])
        .map(|(k, &(shape, strides))| {
            let (name, source) = load(shape, strides);
            source.replace(&format!("def {name}("), &format!("def LOAD{k}(")) + "\n\n"
        })
        .collect();

    // Extra inputs and secondary output pointers arrive packed in the `EXTRA` tuple.
    let mut code: String = (0..inputs.len())
        .map(|k| format!("    in{k} = EXTRA[{k}]\n"))
        .collect();

    let pointers: Vec<String> = (0..outputs.len())
        .map(|o| match o {
            0 => "ptr".to_string(),
            _ => {
                writeln!(code, "    out{o} = EXTRA[{}]", inputs.len() + o - 1).unwrap();
                format!("out{o}")
            }
        })
        .collect();

    let store = |o: usize, index: &str, value: &str, mask: &str| {
        let (shape, strides) = outputs[o];
        format!(
            "tl.store({} + {}, {value}, mask={mask})",
            pointers[o],
            stored_at(index, shape, strides)
        )
    };

    let product = Product {
        value: dot.output,
        regions: gemm_regions(body, dot),
        flags: gemm_flags(body, dot),
        expanded: gemm_expanded(body, dot),
    };

    code += &product.stores(body, by_value, [rows, cols], &store);

    named(
        "epilogue",
        formatdoc!(
            "
            {loads}@triton.jit
            def NAME(ptr, EXTRA, l, m, n, value, mask):
            {code}"
        ),
    )
}

/// What a GEMM-style epilogue starts from: the `value` its accumulator tile holds (a dot's product, or a
/// split-K combine's sum through its rounding cast), the column `regions` its readers split it into, how the
/// accumulator is shaped for them (`flags`), and an expansion's two concatenated operands.
pub(crate) struct Product<'a> {
    pub(crate) value: Var,
    pub(crate) regions: Option<Vec<Columns<'a>>>,
    pub(crate) flags: EpilogueFlags,
    pub(crate) expanded: Option<([Var; 2], crate::codegen::regions::Expansion)>,
}

impl Product<'_> {
    /// Lines computing `body`'s outputs from the accumulator tile `value` (rows `m`, columns `n` of batch `l`
    /// of a `[rows, cols]` product, `mask` its valid elements, `in{k}` the inputs) and storing each through
    /// `store(output, index, value, mask)`, one guarded block per output column region.
    pub(crate) fn stores(
        &self,
        body: &Graph,
        by_value: &[bool],
        [rows, cols]: [usize; 2],
        store: &dyn Fn(usize, &str, &str, &str) -> String,
    ) -> String {
        let t = triton_type(body.type_of(self.value).dtype).expect("a Triton dtype");
        let rank = body.type_of(self.value).shape.len();
        let flags = self.flags;
        let mut code = String::new();

        let regions: Vec<(usize, usize, usize, Vec<Var>, bool)> = match &self.regions {
            Some(regions) => regions
                .iter()
                .map(|c| {
                    let interleaved = c.k() > 1 && !flags.halves();
                    (c.axis, c.offset, c.width, c.members(), interleaved)
                })
                .collect(),
            None => vec![(rank - 1, 0, cols, vec![self.value], false)],
        };

        // The value itself, when an output besides its regions (a split-K combine's sum read whole), stored once.
        if self.regions.is_some() && !flags.halves() {
            for (o, _) in body
                .outputs()
                .iter()
                .enumerate()
                .filter(|&(_, &v)| v == self.value)
            {
                let index = format!("l * {} + m * {cols} + n", rows * cols);
                let value = format!("value.to({t})");
                writeln!(code, "    {}", store(o, &index, &value, "mask")).unwrap();
            }
        }

        let mut locals = 0;

        for (r, (axis, offset, width, members, interleaved)) in regions.into_iter().enumerate() {
            let columns = axis + 1 == rank;
            let k = members.len();
            let (coord, end) = (if columns { "n" } else { "m" }, offset + width);
            let size = width / k;
            let mut seeds = Vec::new();

            if flags.halves() {
                writeln!(code, "    r{r}_c = {coord}\n    r{r}_keep = mask").unwrap();

                for (member, &v) in members.iter().enumerate() {
                    writeln!(code, "    r{r}_a{member} = value[{member}].to({t})").unwrap();
                    seeds.push((v, format!("r{r}_a{member}")));
                }

                writeln!(code, "    r{r}_local = r{r}_c").unwrap();
            } else if interleaved {
                let tiles = split(&mut code, "value", k, columns, &format!("r{r}_value"));
                let c = split(&mut code, coord, k, columns, &format!("r{r}_coord"));
                let keep = split(&mut code, "mask", k, columns, &format!("r{r}_mask"));

                writeln!(
                    code,
                    "    r{r}_c = {}\n    r{r}_keep = {} & (r{r}_c >= {offset}) & (r{r}_c < {end})\n    r{r}_local = (r{r}_c - {offset}) // {k}",
                    c[0], keep[0]
                )
                .unwrap();

                for (member, &v) in members.iter().enumerate() {
                    writeln!(code, "    r{r}_a{member} = {}.to({t})", tiles[member]).unwrap();
                    seeds.push((v, format!("r{r}_a{member}")));
                }
            } else {
                writeln!(
                    code,
                    "    r{r}_c = {coord}\n    r{r}_keep = mask & (r{r}_c >= {offset}) & (r{r}_c < {end})\n    r{r}_local = r{r}_c - {offset}\n    r{r}_a0 = value.to({t})"
                )
                .unwrap();

                seeds.push((members[0], format!("r{r}_a0")));
            }

            // Flat index of the tile within the region's own (narrower) output.
            let index = match columns {
                true => format!("l * {} + m * {size} + r{r}_local", rows * size),
                false => format!("l * {} + r{r}_local * {cols} + n", size * cols),
            };

            let mut emitter = Emitter::new(body, by_value);
            emitter.locals = locals;
            emitter.mask = format!("r{r}_keep");

            for (v, tile) in seeds {
                emitter.registers.insert(v);
                emitter.values.insert((v, "j".into()), tile);
            }

            writeln!(emitter.lines, "    j = {index}").unwrap();
            let mut stores = String::new();

            for (o, &v) in body.outputs().iter().enumerate() {
                if !depends(body, v, &members) {
                    continue;
                }

                // An update writes the region's values into its operand's slot (`pos` of a KV cache), the
                // operand's other elements left as they are (the output holds the operand).
                if let Some(update) = updated(body, v) {
                    let (value, at) = slot(&mut emitter, body, update);
                    let keep = format!("r{r}_keep");
                    writeln!(stores, "    {}", store(o, &at, &value, &keep)).unwrap();
                    continue;
                }

                // An expansion (as on MPS) stores its two dot-shaped operands into their slots of the
                // concatenation, `width` elements apart in runs of `width`.
                let parts = match (o, self.expanded) {
                    (0, Some(([a, b], _))) => {
                        let concat = body.nodes().iter().find(|n| n.output == v);

                        let Some(Primitive::Concatenate { dimension, .. }) =
                            concat.map(|n| &n.primitive)
                        else {
                            unreachable!("an expansion's concatenation")
                        };

                        let product = &body.type_of(self.value).shape;
                        let width: usize =
                            product[(*dimension).min(product.len())..].iter().product();

                        writeln!(
                            stores,
                            "    jx0 = (j // {width}) * {} + j % {width}\n    jx1 = jx0 + {width}",
                            2 * width
                        )
                        .unwrap();

                        vec![(a, "jx0"), (b, "jx1")]
                    }
                    _ => vec![(v, "j")],
                };

                for (v, index) in parts {
                    let value = emitter.value(v, "j".into());
                    let keep = format!("r{r}_keep");
                    writeln!(stores, "    {}", store(o, index, &value, &keep)).unwrap();
                }
            }

            locals = emitter.locals;
            let block = format!("{}{stores}", emitter.lines);

            // Skip regions the tile doesn't overlap; in halves mode each tile covers every part.
            match flags.halves() {
                false => write!(
                    code,
                    "    if (tl.max(r{r}_c) >= {offset}) & (tl.min(r{r}_c) < {end}):\n{}",
                    indented(&block)
                ),
                true => write!(code, "{block}"),
            }
            .unwrap();
        }

        code
    }
}

/// Tile shape and launch parameters for one Triton GEMM candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub struct Config {
    /// Output rows per program.
    pub block_m: usize,
    /// Output columns per program.
    pub block_n: usize,
    /// Reduction-axis slice loaded per loop iteration.
    pub block_k: usize,
    /// Rows of program tiles grouped together so neighbouring programs share operand tiles in L2.
    pub group_m: usize,
    /// Software pipelining depth (`num_stages`).
    pub stages: usize,
    /// Warps per program (`num_warps`).
    pub warps: usize,
}

/// GEMM candidates the autotuner benchmarks, largest tiles first.
pub(crate) const CONFIGS: [Config; 7] = [
    Config {
        block_m: 128,
        block_n: 256,
        block_k: 64,
        group_m: 8,
        stages: 3,
        warps: 8,
    },
    Config {
        block_m: 64,
        block_n: 256,
        block_k: 32,
        group_m: 8,
        stages: 4,
        warps: 4,
    },
    Config {
        block_m: 128,
        block_n: 128,
        block_k: 32,
        group_m: 8,
        stages: 4,
        warps: 4,
    },
    Config {
        block_m: 128,
        block_n: 64,
        block_k: 32,
        group_m: 8,
        stages: 4,
        warps: 4,
    },
    Config {
        block_m: 64,
        block_n: 128,
        block_k: 32,
        group_m: 8,
        stages: 4,
        warps: 4,
    },
    Config {
        block_m: 128,
        block_n: 32,
        block_k: 32,
        group_m: 8,
        stages: 4,
        warps: 4,
    },
    Config {
        block_m: 64,
        block_n: 32,
        block_k: 32,
        group_m: 8,
        stages: 5,
        warps: 2,
    },
];

/// Index into `CONFIGS` used when autotuning is off.
pub(crate) const UNTUNED: usize = 2;

/// Most rows a matrix-vector product (GEMV) has: as MPS's `GEMV_VECTORS`.
pub(crate) const GEMV_ROWS: usize = 8;

/// Whether `dot` of `lhs` and `rhs` is a matrix-vector product the GEMV kernel (`gemv_kernel.py`) runs, by MPS's
/// rule (`dot_general::mps::gemv`, its vector-lhs side): float accumulation, at most `GEMV_ROWS` rows, and a
/// matrix contracted along its last (contiguous) dim, as a decode step's `x @ W.t()`.
pub(crate) fn gemv(dot: &Primitive, lhs: &TensorType, rhs: &TensorType) -> bool {
    let Primitive::DotGeneral {
        lhs_contracting,
        rhs_contracting,
        lhs_batch,
        accum_dtype,
        ..
    } = dot
    else {
        return false;
    };

    let free: usize = (0..lhs.shape.len())
        .filter(|d| !lhs_contracting.contains(d) && !lhs_batch.contains(d))
        .map(|d| lhs.shape[d])
        .product();

    accum_dtype.is_float()
        && free <= GEMV_ROWS
        && rhs_contracting.as_slice() == [rhs.shape.len().saturating_sub(1)]
}

/// Whether GEMM body `body`'s dot is a matrix-vector product (`gemv`), which runs on the GEMV kernel and so
/// joins no row fusion (as MPS keeps its GEMVs out of row reductions with a matrix lhs).
pub(crate) fn gemv_body(body: &Graph) -> bool {
    gemm_dot(body).is_some_and(|d| {
        gemv(
            &d.primitive,
            body.type_of(d.inputs[0]),
            body.type_of(d.inputs[1]),
        )
    })
}

/// GEMV candidates for `m` rows: every program holds the rows (`block_m`, a power of two) and `block_n` matrix rows,
/// streaming `block_k` of K at a time, at most `GEMV_ELEMENTS` products a step.
pub(crate) fn gemv_configs(m: usize) -> Vec<Config> {
    let block_m = m.next_power_of_two();
    let mut out = Vec::new();

    for block_n in [8, 16, 32, 64] {
        for block_k in [64, 128, 256, 512] {
            for warps in [2, 4, 8] {
                if block_m * block_n * block_k <= GEMV_ELEMENTS
                    && block_n * block_k >= 32 * warps * 8
                {
                    out.push(Config {
                        block_m,
                        block_n,
                        block_k,
                        group_m: 1,
                        stages: 1,
                        warps,
                    });
                }
            }
        }
    }

    out
}

/// The rows of a GEMV step's product: its dot's lhs elements outside its contracting and batch dims.
fn gemv_rows(step: &TritonStep) -> Option<usize> {
    let dot = match &step.primitive {
        Primitive::Fusion { body, .. } => {
            let dot = gemm_dot(body)?;
            (dot.primitive.clone(), body.type_of(dot.inputs[0]).clone())
        }
        p => (p.clone(), step.inputs.first()?.1.clone()),
    };

    let (
        Primitive::DotGeneral {
            lhs_contracting,
            lhs_batch,
            ..
        },
        lhs,
    ) = &dot
    else {
        return None;
    };

    Some(
        (0..lhs.shape.len())
            .filter(|d| !lhs_contracting.contains(d) && !lhs_batch.contains(d))
            .map(|d| lhs.shape[d])
            .product(),
    )
}

/// Most products a GEMV program accumulates per step (`block_m x block_n x block_k`).
const GEMV_ELEMENTS: usize = 16 << 10;

/// The GEMV config used when autotuning is off: 16 rows of the matrix, up to 256 of K a step (fewer with more
/// rows, within `GEMV_ELEMENTS`), 4 warps.
pub(crate) fn untuned_gemv(m: usize) -> Config {
    let (block_m, block_n) = (m.next_power_of_two(), 16);

    Config {
        block_m,
        block_n,
        block_k: (GEMV_ELEMENTS / (block_m * block_n)).min(256),
        group_m: 1,
        stages: 1,
        warps: 4,
    }
}

/// Widest row a GEMM's tile holds whole for a row epilogue (`BLOCK_SIZE_N`); wider products are written
/// and normalized by a row kernel. On an H100 (bf16, autotuned) the fused kernel matches a GEMM and a
/// row kernel at 256 but is slower at 512 (121 vs 112 µs) and 1024 (252 vs 132 µs): a tile of whole
/// wide rows leaves too few programs.
pub(crate) const ROW_TILE: usize = 256;

/// Most accumulator elements a row-epilogue GEMM tile holds (`BLOCK_SIZE_M x BLOCK_SIZE_N`).
const ROW_TILE_ELEMENTS: usize = 32 << 10;

/// GEMM candidates for a row epilogue over rows `n` wide: the tile's `BLOCK_SIZE_N` covers the row, each
/// program owning `block_m` whole rows, as many as the accumulator budget allows.
pub(crate) fn row_gemm_configs(n: usize) -> Vec<Config> {
    let block_n = n.next_power_of_two().max(16);
    let mut configs = Vec::new();

    for block_m in [16, 32, 64] {
        if block_m > 16 && block_m * block_n > ROW_TILE_ELEMENTS {
            continue;
        }

        for block_k in [32, 64] {
            for warps in [4, 8] {
                configs.push(Config {
                    block_m,
                    block_n,
                    block_k,
                    group_m: 1,
                    stages: 2,
                    warps,
                });
            }
        }
    }

    configs
}

/// The row-epilogue GEMM config used when autotuning is off: the most rows the accumulator budget
/// allows (at least 16), `BLOCK_SIZE_K` 32, two stages, 8 warps.
pub(crate) fn untuned_row_gemm(n: usize) -> Config {
    let block_n = n.next_power_of_two().max(16);

    Config {
        block_m: (ROW_TILE_ELEMENTS / block_n).clamp(16, 64),
        block_n,
        block_k: 32,
        group_m: 1,
        stages: 2,
        warps: 8,
    }
}

/// Columns of a wide row each partial covers: the narrowest GEMM tile's `BLOCK_SIZE_N`, so every
/// candidate tile (each a multiple of it) writes whole partials and the autotuner picks any.
const PARTIAL_WIDTH: usize = 32;

/// Narrowest partial: a row width no power of two this wide divides (an odd one) stays whole.
const LEAST_PARTIAL_WIDTH: usize = 8;

/// The partial width for a GEMM whose product rows are `n` wide and normalized after it, rows too wide
/// for one tile (past `ROW_TILE`): the widest power of two up to `PARTIAL_WIDTH` dividing `n`, so
/// rows of any width MPS splits are split too (1000 in partials of 8).
pub(crate) fn partial_tile(n: usize) -> Option<usize> {
    let tile = (1 << n.trailing_zeros()).min(PARTIAL_WIDTH);
    (n > ROW_TILE && tile >= LEAST_PARTIAL_WIDTH).then_some(tile)
}

#[cfg(feature = "python")]
crate::autotune::python::config_class! {
    /// `lumen.autotune.TritonGemmConfig`: a tile of the Triton backend's GEMM.
    PyTritonGemmConfig(Config) = "TritonGemmConfig" {
        block_m: usize,
        block_n: usize,
        block_k: usize,
        stages: usize,
        warps: usize,
        group_m: usize = 8,
    }

    /// Every config the autotuner times for a plain GEMM, or for one with a row epilogue over rows `row` wide.
    #[staticmethod]
    #[pyo3(signature = (row=None))]
    fn candidates(row: Option<usize>) -> Vec<Self> {
        match row {
            Some(n) => row_gemm_configs(n),
            None => CONFIGS.to_vec(),
        }
        .into_iter()
        .map(PyTritonGemmConfig)
        .collect()
    }
}

/// The Triton GEMM, running a matmul, a GEMM fusion (with its epilogue, or a row epilogue) or a linear
/// cross-entropy.
pub(crate) struct Gemm;

impl crate::codegen::kernel::Kernel for Gemm {
    type Step = TritonStep;
    type Buffers = Tensors;
    type Config = Config;

    /// Every GEMM config for a GEMM step (whole-row tiles for a row epilogue).
    fn configs(&self, step: &TritonStep) -> Vec<Config> {
        match &step.primitive {
            Primitive::Fusion { name, .. } if name == DOT_ROW => {
                row_gemm_configs(step.output.1.shape.last().copied().unwrap_or(1))
            }
            _ if launch::runs_gemv(step) => gemv_configs(gemv_rows(step).unwrap_or(1)),
            _ if is_gemm(step) => CONFIGS.to_vec(),
            _ => Vec::new(),
        }
    }

    /// Launches the GEMM with `config`, its untuned one when `None`.
    fn run(
        &self,
        step: &TritonStep,
        config: Option<&Config>,
        (inputs, out, outs): &Tensors,
    ) -> Result<(), String> {
        launch::gemm(step, inputs, out, outs, config)
    }
}
