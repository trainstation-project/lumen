//! The kernels plan steps run: each one's candidate configs and its launch under one, which codegen picks per step
//! and the autotuner times.

use std::fmt;
use std::ops::Deref;

use crate::autotune::forced::forced;
use crate::plan::{StepBase, Tuned};

/// A kernel a plan step of one backend runs: the configs it can be tuned over and its launch under one, compiled
/// for the tensors it runs on at its first launch on them.
pub(crate) trait Kernel: Send + Sync + 'static {
    /// The backend's plan steps.
    type Step: Deref<Target = StepBase>;
    /// What a step runs on: its tensors, or the raw buffers of a backend that launches from pointers.
    type Buffers;
    /// One candidate's launch parameters (`()` for a kernel with none).
    type Config: Clone + fmt::Debug + Send + Sync + 'static;

    /// Candidate configs for `step`; none when it isn't tuned (too small to be worth it, or nothing to tune).
    fn configs(&self, step: &Self::Step) -> Vec<Self::Config> {
        let _ = step;
        Vec::new()
    }

    /// Whether forced `config` can run `step`; one it can't keeps its untuned launch.
    fn accepts(&self, step: &Self::Step, config: &Self::Config) -> bool {
        let _ = (step, config);
        true
    }

    /// Prepares `step`'s launches under each of `configs` before any runs: a kernel generating a variant per config
    /// compiles them together. By default nothing; `run` compiles what it needs on first use.
    fn prepare(&self, step: &Self::Step, configs: &[Self::Config]) -> Result<(), String> {
        let _ = (step, configs);
        Ok(())
    }

    /// Runs `step` on `buffers` under `config`, or its untuned launch when `None`.
    fn run(
        &self,
        step: &Self::Step,
        config: Option<&Self::Config>,
        buffers: &Self::Buffers,
    ) -> Result<(), String>;
}

/// A `Kernel` behind its config type, as a step holds it: its candidates and a forced config as `Tuned`s, and its
/// launch under one.
pub(crate) trait Runner<S, B>: Send + Sync {
    /// The kernel's name, keying its tuning results.
    fn name(&self) -> &'static str;
    /// Every candidate config for `step`.
    fn candidates(&self, step: &S) -> Vec<Tuned>;
    /// The config forced for this kernel (`lumen.autotune.use`), if one is and it runs `step`.
    fn forced(&self, step: &S) -> Option<Tuned>;
    /// Prepares `step`'s launches under each of `tuned`'s configs before any runs.
    fn prepare(&self, step: &S, tuned: &[Tuned]) -> Result<(), String>;
    /// Runs `step` on `buffers` under `tuned`'s config, or its untuned launch when `None`.
    fn run(&self, step: &S, tuned: Option<&Tuned>, buffers: &B) -> Result<(), String>;
}

/// `config` of kernel `K` as `step`'s tuned launch, named by its `Debug`, for the views `step` reads.
pub(crate) fn tuned<K: Kernel>(step: &K::Step, config: K::Config) -> Tuned {
    Tuned::new(format!("{config:?}"), config).for_views(step.views.clone())
}

/// Every kernel runs behind its config type.
impl<K: Kernel> Runner<K::Step, K::Buffers> for K {
    /// The kernel's type name.
    fn name(&self) -> &'static str {
        std::any::type_name::<K>()
    }

    /// The kernel's configs for `step`, each named by its `Debug`.
    fn candidates(&self, step: &K::Step) -> Vec<Tuned> {
        self.configs(step)
            .into_iter()
            .map(|config| tuned::<K>(step, config))
            .collect()
    }

    /// The forced config of the kernel's config type, if it can run `step`.
    fn forced(&self, step: &K::Step) -> Option<Tuned> {
        forced::<K::Config>()
            .filter(|config| self.accepts(step, config))
            .map(|config| tuned::<K>(step, config))
    }

    /// Prepares the kernel under each config of its type among `tuned`.
    fn prepare(&self, step: &K::Step, tuned: &[Tuned]) -> Result<(), String> {
        let configs: Vec<K::Config> = tuned
            .iter()
            .filter_map(Tuned::launch::<K::Config>)
            .cloned()
            .collect();

        Kernel::prepare(self, step, &configs)
    }

    /// Runs the kernel under `tuned`'s config, if it holds one of the kernel's config type.
    fn run(
        &self,
        step: &K::Step,
        tuned: Option<&Tuned>,
        buffers: &K::Buffers,
    ) -> Result<(), String> {
        let config = tuned.and_then(Tuned::launch::<K::Config>);
        Kernel::run(self, step, config, buffers)
    }
}

/// A step's kernel shows as its name.
impl<S, B> fmt::Debug for dyn Runner<S, B> {
    /// The kernel's name.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
