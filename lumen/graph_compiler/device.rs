//! The devices the graph compiler targets and the one module every device's passes rewrite: each device is a type
//! implementing [`Device`], its answers to each pass living beside the pass (`passes/<pass>/<device>.rs`), so the
//! compiler is one pass list (`pipeline::pipeline`) and one `compile` for every device.

use super::passes::Module;
use super::passes::alone::AloneDevice;
use super::passes::ane::NeuralEngineDevice;
use super::passes::attention::AttentionDevice;
use super::passes::chunked_cross_entropy::ChunkedCrossEntropyDevice;
use super::passes::diamonds::{DiamondsDevice, Row, RowsModule};
use super::passes::dot::canonicalize::CanonicalizeDotsDevice;
use super::passes::dot::fold_reshapes::FoldDotReshapesDevice;
use super::passes::dot::fuse::FuseDotsDevice;
use super::passes::dot::rows::DotRowsDevice;
use super::passes::dot::split_k::SplitKDevice;
use super::passes::dot::strength::DotStrengthDevice;
use super::passes::fusion::FusionDevice;
use super::passes::horizontal::HorizontalFusionDevice;
use super::passes::linear_cross_entropy::LinearCrossEntropyDevice;
use super::passes::linear_reordering::LinearReorderingDevice;
use super::passes::merge_dots::{Blocks, MergeDotsDevice, Packs};
use super::passes::place_concats::PlaceConcatsDevice;
use super::passes::place_views::PlaceViewsDevice;
use super::passes::take_along_axis::TakeAlongAxisDevice;
use super::passes::unsupported::UnsupportedDevice;
use super::passes::updates::UpdatesDevice;
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

/// A device the graph compiler targets: its answer to every pass (each pass's own trait, implemented beside the
/// pass), and how its passed graph becomes a plan.
pub(crate) trait Device:
    TakeAlongAxisDevice
    + ChunkedCrossEntropyDevice
    + MergeDotsDevice
    + AttentionDevice
    + LinearCrossEntropyDevice
    + FoldDotReshapesDevice
    + DotStrengthDevice
    + NeuralEngineDevice
    + LinearReorderingDevice
    + SplitKDevice
    + CanonicalizeDotsDevice
    + DiamondsDevice
    + DotRowsDevice
    + UpdatesDevice
    + FuseDotsDevice
    + PlaceConcatsDevice
    + FusionDevice
    + PlaceViewsDevice
    + AloneDevice
    + HorizontalFusionDevice
    + UnsupportedDevice
{
    /// The device's name, in profiler ranges and dump files.
    const NAME: &'static str;

    /// Plans `module`'s passed graph on `device`, generates its kernels and autotunes them when enabled.
    fn lower(
        module: DeviceModule<'_, Self>,
        device: crate::Device,
    ) -> Result<Box<dyn Plan>, String>;
}

/// Compiles `graph` for device `D` (`device`): runs the one pass list, then lowers the result to a plan.
pub(crate) fn compile<D: Device>(
    graph: &Graph,
    device: crate::Device,
    options: &Options,
) -> Result<Box<dyn Plan>, String> {
    let mut module = DeviceModule::<D>::new(graph, options);
    super::pipeline::pipeline::<D>(&options.config).run(&mut module)?;
    D::lower(module, device)
}

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
