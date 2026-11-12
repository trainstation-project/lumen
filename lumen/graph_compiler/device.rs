//! The devices the graph compiler targets and the one module every device's passes rewrite: each device is a type
//! implementing [`Device`], its answers to each pass living beside the pass (`passes/<pass>/<device>.rs`), so the
//! compiler is one pass list (`pipeline::pipeline!`) and one `compile!` for every device.

use super::passes::Module;
use super::passes::diamonds::{Row, RowsModule};
use super::passes::merge_dots::{Blocks, Packs};
use super::{CompilerConfig, Options};
pub(crate) use crate::codegen::utils::{cuda::Cuda, tpu::Tpu, triton::Triton};
use crate::graph::Graph;
use crate::graph_compiler::positions;
use crate::plan::{CpuPlan, Plan, PlanOptions};
use std::marker::PhantomData;

// Each kernel backend's codegen type is the device it compiles for: CUDA through CuTe DSL kernels (CUDA's default),
// CUDA through Triton (`config.compiler.use_triton_backend`), Apple GPUs through Metal, TPUs through Pallas.
#[cfg(lumen_mps_linked)]
pub(crate) use crate::codegen::utils::mps::Mps;

/// The host, interpreting the simplified graph.
pub(crate) struct Cpu;

/// A graph being compiled for device `D`, with the state passes hand to later ones.
pub(crate) struct DeviceModule<'a, D> {
    pub(crate) graph: Graph,
    pub(crate) options: &'a Options,
    /// Packed weight blocks dot merging appends as the graph's last inputs: member input indices and the packed
    /// axis.
    pub(crate) packed: Packs,
    /// Per packed block, how many members each interleaved group spans.
    pub(crate) interleaved: Vec<Vec<usize>>,
    /// Normalization rows the diamonds pass finds, which fusion consumes.
    pub(crate) rows: Vec<Row>,
    /// Graph input indices passed to kernels by value.
    #[cfg_attr(not(lumen_mps_linked), allow(dead_code))]
    pub(crate) scalars: Vec<usize>,
    device: PhantomData<D>,
}

impl<'a, D> DeviceModule<'a, D> {
    /// `graph` to compile under `options`, its by-value inputs those `options` marks.
    pub(crate) fn new(graph: &Graph, options: &'a Options) -> Self {
        DeviceModule {
            graph: graph.clone(),
            options,
            packed: Vec::new(),
            interleaved: Vec::new(),
            rows: Vec::new(),
            scalars: super::positions(&options.scalars),
            device: PhantomData,
        }
    }
}

impl<D> Module for DeviceModule<'_, D> {
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

impl<D> Blocks for DeviceModule<'_, D> {
    /// The packed and interleaved weight block tables the dot merge passes fill.
    fn blocks(&mut self) -> (&mut Packs, &mut Vec<Vec<usize>>) {
        (&mut self.packed, &mut self.interleaved)
    }
}

impl<D> RowsModule for DeviceModule<'_, D> {
    fn rows_mut(&mut self) -> &mut Vec<Row> {
        &mut self.rows
    }
}

/// A device the graph compiler targets: its name and how its passed graph becomes a plan (its passes are the ones
/// it implements beside each pass, `passes/<pass>/<device>.rs`; `pipeline::pipeline!` picks them).
pub(crate) trait Device: Sized + 'static {
    /// The device's name, in profiler ranges and dump files.
    const NAME: &'static str;

    /// Plans `module`'s passed graph on `device`, generates its kernels and autotunes them when enabled.
    fn lower(
        module: DeviceModule<'_, Self>,
        device: crate::Device,
    ) -> Result<Box<dyn Plan>, String>;
}

/// Compiles `graph` for concrete device `$device` (a `crate::Device`, `$at`) under `$options`: runs the one pass
/// list, then lowers the result to a plan.
macro_rules! compile {
    ($device:ty, $graph:expr, $at:expr, $options:expr) => {{
        use $crate::graph_compiler::device::Device as _;

        let options: &$crate::graph_compiler::Options = $options;
        let mut module =
            $crate::graph_compiler::device::DeviceModule::<$device>::new($graph, options);
        $crate::graph_compiler::pipeline::pipeline!($device, &options.config).run(&mut module)?;
        <$device>::lower(module, $at)
    }};
}

pub(crate) use compile;

/// The CPU plans the simplified graph as given.
impl Device for Cpu {
    const NAME: &'static str = "cpu";

    /// Plans `module`'s graph for the host.
    fn lower(module: DeviceModule<'_, Self>, _: crate::Device) -> Result<Box<dyn Plan>, String> {
        Ok(Box::new(plan(&module.graph, module.options)))
    }
}

/// Plans `graph` as given, with no scratch region, kernel blocks or device layout.
pub(crate) fn plan(graph: &Graph, options: &Options) -> CpuPlan {
    let plan = PlanOptions {
        scratch: None,
        donate: options.donate.clone(),
        donate_into: options.donate_into.clone(),
        parameters: options.parameters.clone(),
        scalars: positions(&options.scalars),
        memory_limit: options.config.memory_limit(),
        blocks: Vec::new(),
        layout: None,
    };

    CpuPlan::compile_with(graph, &plan)
}
