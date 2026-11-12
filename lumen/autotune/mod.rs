//! Backend-independent kernel autotuning: benchmarks candidate configs of each plan step and keeps the fastest.

pub(crate) mod cuda;
pub(crate) mod forced;
#[cfg(lumen_mps_linked)]
pub(crate) mod mps;
#[cfg(feature = "python")]
pub(crate) mod python;
pub(crate) mod triton;

pub(crate) use forced::forcing;

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, PoisonError};

use crate::device::Device;
use crate::graph::Primitive;
use crate::plan::{DevicePlan, Step, StepBase, Tuned};
use crate::runtime::event::event_for;
use crate::runtime::stream::hold;

/// A backend's source of kernel candidates for plan steps.
pub(crate) trait Kernels {
    /// The backend's plan steps.
    type Step;
    /// Backend-specific tuning parameters of one candidate.
    type Config: Clone + 'static;

    /// Candidate configs for `step`; empty when the step isn't tuned.
    fn configs(&self, step: &Self::Step) -> Vec<Self::Config>;
    /// Whether `step` runs on this family's kernels, so a forced config applies to it: by default, whether it
    /// has candidates (a family whose candidates depend on the step's size overrides it).
    fn handles(&self, step: &Self::Step) -> bool {
        !self.configs(step).is_empty()
    }
    /// Whether forced `config` can run `step` (a step it can't run keeps its untuned kernel): by default, whether
    /// this family handles `step`.
    fn accepts(&self, step: &Self::Step, config: &Self::Config) -> bool {
        let _ = config;
        self.handles(step)
    }
    /// Builds the kernel for `step` under `config`.
    fn build(&mut self, step: &Self::Step, config: Self::Config) -> Tuned;
    /// Compiles every candidate built so far, before any is timed.
    fn compile(&mut self) -> Result<(), String> {
        Ok(())
    }
}

/// Runs plan steps on a device so they can be timed.
pub(crate) trait Timer {
    /// The backend's plan steps.
    type Step: Step + Clone;
    /// Scratch inputs and outputs a step runs on.
    type Buffers;

    /// Device the steps run on.
    fn device(&self) -> Device;
    /// Allocates buffers to run `step` on, or `None` when it can't be timed.
    fn buffers(&self, step: &Self::Step) -> Result<Option<Self::Buffers>, String>;
    /// Enqueues one run of `step`; `rep` numbers the repetition.
    fn run(&self, buffers: &Self::Buffers, step: &Self::Step, rep: usize) -> Result<(), String>;
}

/// Average seconds per run of `step` over `reps` runs after `WARMUP_RUNS` untimed ones.
fn time<T: Timer>(
    timer: &T,
    buffers: &T::Buffers,
    step: &T::Step,
    reps: usize,
) -> Result<f64, String> {
    for rep in 0..WARMUP_RUNS {
        timer.run(buffers, step, rep)?;
    }

    let device = timer.device();
    let mut start = event_for(device, true)?;
    let mut end = event_for(device, true)?;
    // Stall the queue while enqueuing so the timed runs execute back to back between the events.
    let held = hold(device)?;
    start.record()?;

    for rep in 0..reps {
        timer.run(buffers, step, rep)?;
    }

    end.record()?;
    drop(held);
    end.synchronize()?;
    Ok(start.elapsed_time(end.as_ref())? / 1e3 / reps as f64)
}

/// Runs `step` back to back for at least `WARMUP_SECONDS`, bringing the device to its sustained clocks.
fn warm_up<T: Timer>(timer: &T, buffers: &T::Buffers, step: &T::Step) -> Result<(), String> {
    let mut elapsed = 0.0;

    while elapsed < WARMUP_SECONDS {
        elapsed += time(timer, buffers, step, TIMED_RUNS)? * TIMED_RUNS as f64;
    }

    Ok(())
}

/// Picks the fastest config for every tunable step of a plan, timing with `timer`.
pub(crate) struct Autotuner<T> {
    pub timer: T,
    /// Whether to time candidates; without it, only forced configs (`forced`) are applied.
    pub autotune: bool,
}

/// Untimed runs of a candidate before it is timed.
const WARMUP_RUNS: usize = 5;
/// Runs a candidate's time averages over.
const TIMED_RUNS: usize = 10;
/// Time the device runs a candidate after a step's candidates compile and before any is timed: a GPU takes about
/// this long under load to reach its sustained clocks, which the candidates timed first would otherwise run below
/// (as QuACK's autotuner warms up before benchmarking).
const WARMUP_SECONDS: f64 = 0.2;

/// Steps with the same kernel, operand types and views share one tuning result.
fn key(step: &StepBase) -> String {
    let name = match &step.primitive {
        Primitive::Fusion { name, .. } => name.clone(),
        p => p.to_string(),
    };

    let types: Vec<String> = step.inputs.iter().map(|(_, ty)| ty.to_string()).collect();
    format!(
        "{name} {} {:?} -> {}",
        types.join(" "),
        step.views,
        step.output.1
    )
}

static PICKS: LazyLock<Mutex<HashMap<String, usize>>> = LazyLock::new(Default::default);

fn pick<K>(device: Device, step: &StepBase) -> String {
    let types: Vec<String> = step.inputs.iter().map(|(_, ty)| ty.to_string()).collect();

    format!(
        "{} {device} {} {} {:?} -> {}",
        std::any::type_name::<K>(),
        step.primitive,
        types.join(" "),
        step.views,
        step.output.1
    )
}

impl<T: Timer> Autotuner<T> {
    /// Sets the `tuned` kernel of each tunable step in `plan` to its fastest candidate, outside the profiler.
    pub(crate) fn tune<K: Kernels<Step = T::Step>>(
        &self,
        kernels: &mut K,
        plan: &mut DevicePlan<T::Step>,
    ) -> Result<(), String> {
        crate::profiler::untraced(|| self.tune_untraced(kernels, plan))
    }

    /// Body of [`Self::tune`].
    fn tune_untraced<K: Kernels<Step = T::Step>>(
        &self,
        kernels: &mut K,
        plan: &mut DevicePlan<T::Step>,
    ) -> Result<(), String> {
        // A config forced for this family: every step it handles (and can run) is built with it, untimed.
        if let Some(config) = forced::forced::<K::Config>() {
            for i in 0..plan.steps().len() {
                if kernels.accepts(&plan.steps()[i], &config) {
                    let tuned = kernels.build(&plan.steps()[i], config.clone());
                    plan.steps_mut()[i].tuned = Some(tuned);
                }
            }

            return kernels.compile();
        }

        if !self.autotune {
            return Ok(());
        }

        let mut same: HashMap<String, usize> = HashMap::new();
        let mut jobs: Vec<(Vec<usize>, Vec<Tuned>)> = Vec::new();

        for (i, step) in plan.steps().iter().enumerate() {
            if let Some(&job) = same.get(&key(step)) {
                jobs[job].0.push(i);
                continue;
            }

            let candidates: Vec<Tuned> = kernels
                .configs(step)
                .into_iter()
                .map(|config| kernels.build(step, config))
                .collect();

            if !candidates.is_empty() {
                same.insert(key(step), jobs.len());
                jobs.push((vec![i], candidates));
            }
        }

        if jobs.is_empty() {
            return Ok(());
        }

        kernels.compile()?;
        let picks = || PICKS.lock().unwrap_or_else(PoisonError::into_inner);

        for (steps, candidates) in jobs {
            let picked = pick::<K>(self.timer.device(), &plan.steps()[steps[0]]);

            if let Some(&k) = picks().get(&picked).filter(|&&k| k < candidates.len()) {
                for i in steps {
                    plan.steps_mut()[i].tuned = Some(candidates[k].clone());
                }

                continue;
            }

            let mut trial = plan.steps()[steps[0]].clone();

            let Some(buffers) = self.timer.buffers(&trial)? else {
                continue;
            };

            // A lone candidate has nothing to be timed against: one run checks that it works.
            if let [only] = &candidates[..] {
                trial.tuned = Some(only.clone());

                if time(&self.timer, &buffers, &trial, 1).is_ok() {
                    picks().insert(picked, 0);

                    for i in steps {
                        plan.steps_mut()[i].tuned = Some(only.clone());
                    }
                }

                continue;
            }

            eprintln!(
                "{}: autotuning {} ({}) on {}: {} candidates",
                crate::LIBRARY_NAME,
                trial.label,
                key(&trial),
                self.timer.device(),
                candidates.len()
            );

            // Every candidate's first run (its compile) before any is timed: a compile idles the GPU long enough
            // to drop it to base clocks, which would slow whichever candidate ran next. A failing one is dropped.
            let runs: Vec<bool> = candidates
                .iter()
                .map(|candidate| {
                    trial.tuned = Some(candidate.clone());
                    self.timer.run(&buffers, &trial, 0).is_ok()
                })
                .collect();

            // Then back up to sustained clocks, on the first candidate that runs.
            if let Some(k) = runs.iter().position(|&ran| ran) {
                trial.tuned = Some(candidates[k].clone());
                warm_up(&self.timer, &buffers, &trial).ok();
            }

            // Each candidate's average over `TIMED_RUNS` runs.
            let seconds: Vec<f64> = candidates
                .iter()
                .zip(&runs)
                .map(|(candidate, &ran)| {
                    trial.tuned = Some(candidate.clone());

                    match ran {
                        true => {
                            time(&self.timer, &buffers, &trial, TIMED_RUNS).unwrap_or(f64::INFINITY)
                        }
                        false => f64::INFINITY,
                    }
                })
                .collect();

            let fastest = (0..seconds.len()).min_by(|&a, &b| seconds[a].total_cmp(&seconds[b]));

            let Some(k) = fastest.filter(|&k| seconds[k].is_finite()) else {
                eprintln!(
                    "{}: autotuning {}: no candidate ran",
                    crate::LIBRARY_NAME,
                    trial.label
                );

                continue;
            };

            eprintln!(
                "{}: autotuned {}: candidate {} of {}, {:.1} us",
                crate::LIBRARY_NAME,
                trial.label,
                k + 1,
                candidates.len(),
                seconds[k] * 1e6
            );

            picks().insert(picked, k);

            for i in steps {
                plan.steps_mut()[i].tuned = Some(candidates[k].clone());
            }
        }

        Ok(())
    }
}
