//! Running passes: the one order every device's passes run in, each device filling the slots its kernels
//! support (`Passes`) and leaving the others `Empty`; the named pipeline running them, each in a profiler range,
//! optionally dumping the graph after each one; and the plain module the frontend and CPU pipelines run on.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::graph::Graph;
use crate::graph_compiler::passes::empty::Empty;
use crate::graph_compiler::passes::{CompilerPass, Module};
use crate::graph_compiler::{CompilerConfig, Options};

/// A graph being compiled with the caller's options.
pub(crate) struct GraphModule<'a> {
    pub graph: Graph,
    pub options: &'a Options,
}

impl Module for GraphModule<'_> {
    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn graph_mut(&mut self) -> &mut Graph {
        &mut self.graph
    }

    fn options(&self) -> &Options {
        self.options
    }

    fn config(&self) -> &CompilerConfig {
        &self.options.config
    }
}

/// An ordered list of passes run under one name.
pub(crate) struct Pipeline<M> {
    name: &'static str,
    passes: Vec<Box<dyn CompilerPass<M>>>,
}

/// Counter numbering pipeline runs so each run's dump files are distinct.
static DUMPED: AtomicUsize = AtomicUsize::new(0);

impl<M: Module> Pipeline<M> {
    /// An empty pipeline called `name`.
    pub(crate) fn new(name: &'static str) -> Self {
        Pipeline {
            name,
            passes: Vec::new(),
        }
    }

    /// Appends `pass` to run after those already added, unless it is `Empty`.
    pub(crate) fn add(&mut self, pass: impl CompilerPass<M> + 'static) -> &mut Self {
        self.push(Box::new(pass))
    }

    /// Appends the boxed `pass` to run after those already added, unless it is `Empty`.
    pub(crate) fn push(&mut self, pass: Pass<M>) -> &mut Self {
        if !pass.is_empty() {
            self.passes.push(pass);
        }

        self
    }

    /// Runs every pass on `module` in order, each in a profiler range; when the config sets a dump
    /// directory, writes the graph before the first pass and after each one.
    pub(crate) fn run(&mut self, module: &mut M) -> Result<(), String> {
        let name = self.name;

        let dump = module.config().dump_to().map(str::to_owned).map(|dir| {
            let id = DUMPED.fetch_add(1, Ordering::Relaxed);

            move |step: usize, pass: &str, graph: &Graph| {
                let file = format!("{dir}/{id:04}.{name}.{step:02}.{pass}.txt");

                std::fs::create_dir_all(&dir)
                    .and_then(|()| std::fs::write(&file, graph.to_string()))
                    .map_err(|e| format!("dumping {file}: {e}"))
            }
        });

        if let Some(dump) = &dump {
            dump(0, "input", module.graph())?;
        }

        for (step, pass) in self.passes.iter_mut().enumerate() {
            let range = format!("{name}::{}", pass.name());
            let _range = crate::profiler::record_function(range);
            pass.apply(module)?;

            if let Some(dump) = &dump {
                dump(step + 1, pass.name(), module.graph())?;
            }
        }

        Ok(())
    }
}

/// A boxed pass for a module of type `M`.
pub(crate) type Pass<M> = Box<dyn CompilerPass<M>>;

/// `pass` when `on`, else `Empty`.
pub(crate) fn when<M>(on: bool, pass: impl CompilerPass<M> + 'static) -> Pass<M> {
    match on {
        true => Box::new(pass),
        false => Box::new(Empty),
    }
}

/// A device's passes, one per slot of the order every device's passes run in: its version of the pass, or `Empty`
/// (the default) where its kernels don't take what the pass makes or its config turns the pass off.
pub(crate) trait Passes: Module + Sized {
    /// The pipeline's name, in profiler ranges and dump files.
    const NAME: &'static str;

    /// Lowers `take_along_axis` to a flat `gather`.
    fn take_along_axis(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Computes a cross entropy over logits the graph already computes chunk by chunk, per chunk.
    fn chunked_cross_entropy(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Merges the backward dots of weights sharing a forward input.
    fn merge_dots_backward(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Merges dots sharing an operand into one dot against a packed block, before attention is matched.
    fn merge_dots(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Replaces each forward attention with a `flash_attention` fusion.
    fn flash_attention(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Replaces attention's gradients with `flash_attention_backward` fusions.
    fn flash_attention_backward(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Folds output layouts into the attention fusions, for the device's attention kernels.
    fn attention_layouts(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Replaces `cross_entropy(h @ w, target)` with one `LinearCrossEntropy` op.
    fn linear_cross_entropy(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Clamps gather, scatter-add and linear-cross-entropy indices into range.
    fn clamp_indices(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Folds a reshape flattening leading dims into a 2-D dot's lhs.
    fn fold_dot_reshapes(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Simplifies the graph algebraically.
    fn simplify(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Turns dots between vectors into elementwise multiplies and reduce-sums.
    fn dot_strength(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Merges dots sharing an operand, once the graph is simplified.
    fn merge_dots_late(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Offloads GEMM regions to the Apple Neural Engine.
    fn neural_engine(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Moves a factor of a matmul's operand to its output.
    fn linear_reordering(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Splits long-K matmuls with too few output tiles.
    fn split_k(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Transposes dot operands the device's matmul kernels can't read as laid out.
    fn canonicalize_dots(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Fuses normalization rows into `row` fusions.
    fn fused_diamonds(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Fuses a matmul into the `row` fusion reading its whole output rows.
    fn dot_rows(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Fuses chains of constant-offset `dynamic_update_slice`s into one `updates` fusion.
    fn updates(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Finds normalization rows for fusion.
    fn diamonds(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Fuses matmuls with their slices, stacked lhs and epilogues.
    fn fuse_dots(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Places concatenations whose operands can be written straight into their slots.
    fn place_concats(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Fuses the graph into kernels.
    fn fusion(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Splits rows too wide for one GEMM tile.
    fn wide_rows(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Turns slices read only by GEMMs into strided views.
    fn place_views(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Wraps the nodes left outside any fusion that need a kernel into single-node fusions.
    fn alone(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Copies the concatenations fusion left unable to be written in place.
    fn concats(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }

    /// Merges independent kernels of one kind into one launch.
    fn horizontal_fusion(_: &CompilerConfig) -> Pass<Self> {
        Box::new(Empty)
    }
}

/// The device's passes for `config`, in the shared order.
pub(crate) fn pipeline<M: Passes>(config: &CompilerConfig) -> Pipeline<M> {
    let mut pipeline = Pipeline::new(M::NAME);

    pipeline
        .push(M::take_along_axis(config))
        .push(M::chunked_cross_entropy(config))
        .push(M::merge_dots_backward(config))
        .push(M::merge_dots(config))
        .push(M::flash_attention(config))
        .push(M::flash_attention_backward(config))
        .push(M::attention_layouts(config))
        .push(M::linear_cross_entropy(config))
        .push(M::clamp_indices(config))
        .push(M::fold_dot_reshapes(config))
        .push(M::simplify(config))
        .push(M::dot_strength(config))
        .push(M::merge_dots_late(config))
        .push(M::neural_engine(config))
        .push(M::linear_reordering(config))
        .push(M::split_k(config))
        .push(M::canonicalize_dots(config))
        .push(M::fused_diamonds(config))
        .push(M::dot_rows(config))
        .push(M::updates(config))
        .push(M::diamonds(config))
        .push(M::fuse_dots(config))
        .push(M::place_concats(config))
        .push(M::fusion(config))
        .push(M::wide_rows(config))
        .push(M::place_views(config))
        .push(M::alone(config))
        .push(M::concats(config))
        .push(M::horizontal_fusion(config));

    pipeline
}
