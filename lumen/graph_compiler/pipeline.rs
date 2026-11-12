//! Running passes: the one pass list every device runs (`pipeline`), each pass the device's own version or the
//! identity pass; the named pipeline running them, each in a profiler range, optionally dumping the graph after
//! each one; and the plain module the frontend and CPU pipelines run on.

use crate::graph::Graph;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{Cpu, DeviceModule};
use crate::graph_compiler::passes::identity::Identity;
use crate::graph_compiler::passes::{CompilerPass, DevicePass, Module};
use std::marker::PhantomData;
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

    /// The names of the passes it runs, in order.
    #[cfg(test)]
    pub(crate) fn names(&self) -> Vec<&'static str> {
        self.passes.iter().map(|p| p.name()).collect()
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

/// Device `D`'s version of pass `P`, picked by method resolution: through `(&Pick::<P, D>::new()).pass(config)`,
/// written for a concrete `D`, the impl on `Pick<P, D>` (taken when `P: DevicePass<D>`) wins over the one on
/// `&Pick<P, D>` (the identity pass), so a device implements only the passes it runs.
pub(crate) struct Pick<P, D>(PhantomData<(P, D)>);

impl<P, D> Pick<P, D> {
    /// The picker of pass `P` for device `D`.
    pub(crate) fn new() -> Self {
        Pick(PhantomData)
    }
}

/// A device's version of a pass.
pub(crate) trait Picked<'a, D> {
    /// The pass under `config`.
    fn pass(&self, config: &CompilerConfig) -> Pass<DeviceModule<'a, D>>;
}

impl<'a, P: DevicePass<D>, D> Picked<'a, D> for Pick<P, D> {
    /// The device's own version.
    fn pass(&self, config: &CompilerConfig) -> Pass<DeviceModule<'a, D>> {
        P::pass(config)
    }
}

impl<'a, P, D> Picked<'a, D> for &Pick<P, D> {
    /// The identity pass: the device doesn't run this one.
    fn pass(&self, _: &CompilerConfig) -> Pass<DeviceModule<'a, D>> {
        Box::new(Identity)
    }
}

/// Concrete device `$device`'s version of pass `$pass` under `$config`: its own when the pass implements
/// `DevicePass<$device>`, else the identity pass.
macro_rules! pick {
    ($device:ty, $pass:ty, $config:expr) => {{
        use $crate::graph_compiler::pipeline::Picked as _;
        (&$crate::graph_compiler::pipeline::Pick::<$pass, $device>::new()).pass($config)
    }};
}

/// The one pass list, for concrete device `$device` under `$config` (expanded per device, which is how each slot
/// picks the device's own pass or the identity pass): every device runs these, in this order.
macro_rules! pipeline {
    ($device:ty, $config:expr) => {{
        use $crate::graph_compiler::device::Device as _;
        use $crate::graph_compiler::passes::simplify::Simplify;

        let config: &$crate::graph_compiler::CompilerConfig = $config;
        let mut pipeline = $crate::graph_compiler::pipeline::Pipeline::new(<$device>::NAME);

        pipeline
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::unsupported::Unsupported,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::take_along_axis::TakeAlongAxis,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::chunked_cross_entropy::ChunkedCrossEntropy,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::merge_dots::MergeDotsBackward,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::merge_dots::MergeDots,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::attention::FlashAttention,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::attention::FlashAttentionBackward,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::attention::tpu::AttentionLayouts,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::linear_cross_entropy::LinearCrossEntropy,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::dot::fold_reshapes::FoldDotReshapes,
                config
            ))
            .add(Simplify { matched: true })
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::dot::strength::DotStrength,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::merge_dots::MergeDotsLate,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::ane::NeuralEngine,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::linear_reordering::LinearReordering,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::dot::split_k::SplitK,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::dot::canonicalize::CanonicalizeDots,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::diamonds::FusedDiamonds,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::dot::rows::tpu::DotRows,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::updates::Updates,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::diamonds::Diamonds,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::dot::fuse::tpu::FuseDots,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::place_concats::PlaceConcats,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::fusion::Fusion,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::dot::rows::triton::WideRows,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::place_views::PlaceViews,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::alone::Alone,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::place_concats::Concats,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                $crate::graph_compiler::passes::horizontal::HorizontalFusion,
                config
            ));

        pipeline
    }};
}

pub(crate) use {pick, pipeline};
