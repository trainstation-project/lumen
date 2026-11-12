//! Running passes: the one pass list every device runs (`pipeline`), each pass the device's own version or the
//! identity pass; the named pipeline running them, each in a profiler range, optionally dumping the graph after
//! each one; and the plain module the frontend and CPU pipelines run on.

use crate::graph::Graph;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{Cpu, Device, DeviceModule};
use crate::graph_compiler::passes::identity::Identity;
use crate::graph_compiler::passes::simplify::Simplify;
use crate::graph_compiler::passes::{CompilerPass, Module};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A graph being compiled with the caller's options and no device's passes (the frontend's, the CPU's).
pub(crate) type GraphModule<'a> = DeviceModule<'a, Cpu>;

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

    /// Appends `pass` to run after those already added, unless it is the identity pass.
    pub(crate) fn add(&mut self, pass: impl CompilerPass<M> + 'static) -> &mut Self {
        self.push(Box::new(pass))
    }

    /// Appends the boxed `pass` to run after those already added, unless it is the identity pass.
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

/// `graph` as `pass` leaves it, run on a plain module with default options.
#[cfg(test)]
pub(crate) fn rewritten<P>(graph: &Graph, mut pass: P) -> Graph
where
    P: for<'a> CompilerPass<GraphModule<'a>>,
{
    let options = crate::graph_compiler::Options::default();

    let mut module = GraphModule::new(graph, &options);

    pass.apply(&mut module).expect("a rewritten graph");
    module.graph
}

/// A boxed pass for a module of type `M`.
pub(crate) type Pass<M> = Box<dyn CompilerPass<M>>;

/// `pass` when `on`, else the identity pass.
pub(crate) fn when<M>(on: bool, pass: impl CompilerPass<M> + 'static) -> Pass<M> {
    match on {
        true => Box::new(pass),
        false => Box::new(Identity),
    }
}

/// The passes for device `D` under `config`, the one list every device runs: each pass the device's own version
/// (`passes/<pass>/<device>.rs`), or the identity pass where it doesn't run.
pub(crate) fn pipeline<'a, D: Device>(config: &CompilerConfig) -> Pipeline<DeviceModule<'a, D>> {
    let mut pipeline = Pipeline::new(D::NAME);

    pipeline
        .push(D::unsupported(config))
        .push(D::take_along_axis(config))
        .push(D::chunked_cross_entropy(config))
        .push(D::merge_dots_backward(config))
        .push(D::merge_dots(config))
        .push(D::flash_attention(config))
        .push(D::flash_attention_backward(config))
        .push(D::attention_layouts(config))
        .push(D::linear_cross_entropy(config))
        .push(D::fold_dot_reshapes(config))
        .add(Simplify { matched: true })
        .push(D::dot_strength(config))
        .push(D::merge_dots_late(config))
        .push(D::neural_engine(config))
        .push(D::linear_reordering(config))
        .push(D::split_k(config))
        .push(D::canonicalize_dots(config))
        .push(D::fused_diamonds(config))
        .push(D::dot_rows(config))
        .push(D::updates(config))
        .push(D::diamonds(config))
        .push(D::fuse_dots(config))
        .push(D::place_concats(config))
        .push(D::fusion(config))
        .push(D::wide_rows(config))
        .push(D::place_views(config))
        .push(D::alone(config))
        .push(D::concats(config))
        .push(D::horizontal_fusion(config));

    pipeline
}
