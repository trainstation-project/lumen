//! Running passes: the one pass list every device runs (`pipeline`), each pass the device's own version or the
//! identity pass; the named pipeline running them, each in a profiler range, optionally dumping the graph after
//! each one; and the plain module the frontend and CPU pipelines run on.

use crate::graph::Graph;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{Cpu, DeviceModule};
use crate::graph_compiler::passes::identity::Identity;
use crate::graph_compiler::passes::{CompilerPass, Module};
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

/// Device `D`'s version of a pass, picked by method resolution: through `(&Pick::<D>::new()).pass(config)`, written
/// for a concrete `D`, the impl on `Pick<D>` (taken when `D` implements the pass's device trait) wins over the one on
/// `&Pick<D>` (the identity pass), so a device implements only the passes it runs.
pub(crate) struct Pick<D>(PhantomData<D>);

impl<D> Pick<D> {
    /// The picker for device `D`.
    pub(crate) fn new() -> Self {
        Pick(PhantomData)
    }
}

/// For each slot of the pass list, a module `slot` with trait `Slot`: implemented on `Pick<D>` as device `D`'s pass
/// when `D` implements the slot's device trait, and on `&Pick<D>` as the identity pass.
macro_rules! slots {
    ($($slot:ident: $device:path),* $(,)?) => {
        $(
            #[doc = concat!("The `", stringify!($slot), "` slot of the pass list.")]
            pub(crate) mod $slot {
                use super::{CompilerConfig, DeviceModule, Identity, Pass, Pick};
                use crate::graph_compiler::passes::*;

                /// A device's pass for this slot.
                pub(crate) trait Slot<'a, D> {
                    /// The pass under `config`.
                    fn pass(&self, config: &CompilerConfig) -> Pass<DeviceModule<'a, D>>;
                }

                impl<'a, D: $device> Slot<'a, D> for Pick<D> {
                    /// The device's own pass.
                    fn pass(&self, config: &CompilerConfig) -> Pass<DeviceModule<'a, D>> {
                        D::$slot(config)
                    }
                }

                impl<'a, D> Slot<'a, D> for &Pick<D> {
                    /// The identity pass: the device doesn't implement this one.
                    fn pass(&self, _: &CompilerConfig) -> Pass<DeviceModule<'a, D>> {
                        Box::new(Identity)
                    }
                }
            }
        )*
    };
}

slots! {
    unsupported: unsupported::UnsupportedDevice,
    take_along_axis: take_along_axis::TakeAlongAxisDevice,
    chunked_cross_entropy: chunked_cross_entropy::ChunkedCrossEntropyDevice,
    merge_dots_backward: merge_dots::MergeDotsDevice,
    merge_dots: merge_dots::MergeDotsDevice,
    flash_attention: attention::AttentionDevice,
    flash_attention_backward: attention::AttentionDevice,
    attention_layouts: attention::AttentionDevice,
    linear_cross_entropy: linear_cross_entropy::LinearCrossEntropyDevice,
    fold_dot_reshapes: dot::fold_reshapes::FoldDotReshapesDevice,
    dot_strength: dot::strength::DotStrengthDevice,
    merge_dots_late: merge_dots::MergeDotsDevice,
    neural_engine: ane::NeuralEngineDevice,
    linear_reordering: linear_reordering::LinearReorderingDevice,
    split_k: dot::split_k::SplitKDevice,
    canonicalize_dots: dot::canonicalize::CanonicalizeDotsDevice,
    fused_diamonds: diamonds::DiamondsDevice,
    dot_rows: dot::rows::DotRowsDevice,
    updates: updates::UpdatesDevice,
    diamonds: diamonds::DiamondsDevice,
    fuse_dots: dot::fuse::FuseDotsDevice,
    place_concats: place_concats::PlaceConcatsDevice,
    fusion: fusion::FusionDevice,
    wide_rows: dot::rows::DotRowsDevice,
    place_views: place_views::PlaceViewsDevice,
    alone: alone::AloneDevice,
    concats: place_concats::PlaceConcatsDevice,
    horizontal_fusion: horizontal::HorizontalFusionDevice,
}

/// Concrete device `$device`'s pass for slot `$slot` under `$config`: its own when it implements the slot's device
/// trait, else the identity pass.
macro_rules! pick {
    ($device:ty, $slot:ident, $config:expr) => {{
        use $crate::graph_compiler::pipeline::$slot::Slot as _;
        (&$crate::graph_compiler::pipeline::Pick::<$device>::new()).pass($config)
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
                unsupported,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                take_along_axis,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                chunked_cross_entropy,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                merge_dots_backward,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device, merge_dots, config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                flash_attention,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                flash_attention_backward,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                attention_layouts,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                linear_cross_entropy,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                fold_dot_reshapes,
                config
            ))
            .add(Simplify { matched: true })
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                dot_strength,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                merge_dots_late,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                neural_engine,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                linear_reordering,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device, split_k, config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                canonicalize_dots,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                fused_diamonds,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device, dot_rows, config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device, updates, config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device, diamonds, config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device, fuse_dots, config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                place_concats,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device, fusion, config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device, wide_rows, config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                place_views,
                config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device, alone, config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device, concats, config
            ))
            .push($crate::graph_compiler::pipeline::pick!(
                $device,
                horizontal_fusion,
                config
            ));

        pipeline
    }};
}

pub(crate) use {pick, pipeline};
