//! Each device's passes (`Passes`, its list in `<device>.rs`), running passes in order (`run`: each in a profiler
//! range, optionally dumping the graph after each one), and the plain module tests run single passes on.

mod cpu;
mod cuda;
#[cfg(lumen_mps_linked)]
mod mps;
mod tpu;
mod triton;

use crate::graph::Graph;
use crate::graph_compiler::CompilerConfig;
use crate::graph_compiler::device::{Device, DeviceModule};
use crate::graph_compiler::passes::{CompilerPass, Module};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A graph being compiled with the caller's options and no device's own passes, which tests run single passes on.
#[cfg(test)]
pub(crate) type GraphModule<'a> = DeviceModule<'a, crate::graph_compiler::device::Cpu>;

/// A boxed pass for a module of type `M`.
pub(crate) type Pass<M> = Box<dyn CompilerPass<M>>;

/// A device's passes: the ones the graph compiler runs on its module, in order, under a config, and how they run.
/// Every list starts with the same clean-ups: `preserve_precision` (before CSE shares the casts it drops), `cse` and
/// `simplify` without structural rewrites.
pub(crate) trait Passes: Device {
    /// The device's passes under `config`, in order: each pass its list adds, under the config flags it runs with.
    fn passes<'a>(config: &CompilerConfig) -> Vec<Pass<DeviceModule<'a, Self>>>;

    /// Runs the device's passes on `module` in order, each in a profiler range `<device>::<pass>`; when the config sets
    /// a dump directory, writes the graph before the first pass and after each one (`<run>.<device>.<step>.<pass>.txt`).
    fn run(module: &mut DeviceModule<'_, Self>) -> Result<(), String> {
        let name = Self::NAME;

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

        let passes = Self::passes(module.config());

        for (step, mut pass) in passes.into_iter().enumerate() {
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

/// Counter numbering runs so each run's dump files are distinct.
static DUMPED: AtomicUsize = AtomicUsize::new(0);

/// `graph` as `pass` leaves it, run on a plain module with default options.
#[cfg(test)]
pub(crate) fn rewritten<P>(graph: &Graph, pass: P) -> Graph
where
    P: for<'a> CompilerPass<GraphModule<'a>>,
{
    rewritten_on::<crate::graph_compiler::device::Cpu, P>(graph, pass)
}

/// `graph` as `pass` leaves it, run on device `D`'s module with default options (a pass implemented for the
/// devices whose answers it reads).
#[cfg(test)]
pub(crate) fn rewritten_on<D, P>(graph: &Graph, mut pass: P) -> Graph
where
    P: for<'a> CompilerPass<DeviceModule<'a, D>>,
{
    let options = crate::graph_compiler::Options::default();

    let mut module = DeviceModule::<D>::new(graph, &options);

    pass.apply(&mut module).expect("a rewritten graph");
    module.graph
}
