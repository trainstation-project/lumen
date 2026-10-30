//! Backend-independent kernel autotuning: benchmarks candidate configs of each plan step and keeps the fastest.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, PoisonError};

use crate::device::Device;
use crate::graph::Primitive;
use crate::graph_compiler::Plan;
use crate::graph_compiler::plan::{Step, Tuned};
use crate::stream::{event_for, hold};

/// A backend's source of kernel candidates for plan steps.
pub(crate) trait Kernels {
    /// Backend-specific tuning parameters of one candidate.
    type Config;

    /// Candidate configs for `step`; empty when the step isn't tuned.
    fn configs(&self, step: &Step) -> Vec<Self::Config>;
    /// Builds the kernel for `step` under `config`.
    fn build(&mut self, step: &Step, config: Self::Config) -> Tuned;
    /// Compiles every candidate built so far, before any is timed.
    fn compile(&mut self) -> Result<(), String> {
        Ok(())
    }
}

/// Runs plan steps on a device so they can be timed.
pub(crate) trait Timer {
    /// Scratch inputs and outputs a step runs on.
    type Buffers;

    /// Device the steps run on.
    fn device(&self) -> Device;
    /// Allocates buffers to run `step` on, or `None` when it can't be timed.
    fn buffers(&self, step: &Step) -> Result<Option<Self::Buffers>, String>;
    /// Enqueues one run of `step`; `rep` numbers the repetition.
    fn run(&self, buffers: &Self::Buffers, step: &Step, rep: usize) -> Result<(), String>;
}

/// Average seconds per run of `step` over `reps` runs after one warm-up run.
fn time<T: Timer>(
    timer: &T,
    buffers: &T::Buffers,
    step: &Step,
    reps: usize,
) -> Result<f64, String> {
    timer.run(buffers, step, 0)?;
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

/// Picks the fastest config for every tunable step of a plan, timing with `timer`.
pub(crate) struct Autotuner<T> {
    pub timer: T,
}

/// Timing passes over all candidates; each keeps a candidate's best time.
const PASSES: usize = 2;
/// Time each candidate aims to run for per measurement.
const BUDGET_SECONDS: f64 = 1e-3;
/// Bounds on repetitions per measurement.
const REPS: std::ops::RangeInclusive<usize> = 2..=256;

/// Steps with the same kernel and operand types share one tuning result.
fn key(step: &Step) -> String {
    let name = match &step.primitive {
        Primitive::Fusion { name, .. } => name.clone(),
        p => p.to_string(),
    };

    let types: Vec<String> = step.inputs.iter().map(|(_, ty)| ty.to_string()).collect();
    format!("{name} {} -> {}", types.join(" "), step.output.1)
}

static PICKS: LazyLock<Mutex<HashMap<String, usize>>> = LazyLock::new(Default::default);

fn pick<K>(device: Device, step: &Step) -> String {
    let types: Vec<String> = step.inputs.iter().map(|(_, ty)| ty.to_string()).collect();

    format!(
        "{} {device} {} {} -> {}",
        std::any::type_name::<K>(),
        step.primitive,
        types.join(" "),
        step.output.1
    )
}

impl<T: Timer> Autotuner<T> {
    /// Sets the `tuned` kernel of each tunable step in `plan` to its fastest candidate, outside the profiler.
    pub(crate) fn tune<K: Kernels>(&self, kernels: &mut K, plan: &mut Plan) -> Result<(), String> {
        crate::profiler::untraced(|| self.tune_untraced(kernels, plan))
    }

    /// Body of [`Self::tune`].
    fn tune_untraced<K: Kernels>(&self, kernels: &mut K, plan: &mut Plan) -> Result<(), String> {
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

            let mut reps = vec![0; candidates.len()];
            let mut seconds = vec![f64::INFINITY; candidates.len()];
            let mut runs = vec![true; candidates.len()];

            for _ in 0..PASSES {
                for (k, candidate) in candidates.iter().enumerate() {
                    trial.tuned = Some(candidate.clone());

                    // The first run sizes the repetitions to the time budget; a failing candidate is dropped.
                    if reps[k] == 0 && runs[k] {
                        match time(&self.timer, &buffers, &trial, 1) {
                            Ok(warm) => {
                                reps[k] = ((BUDGET_SECONDS / warm) as usize)
                                    .clamp(*REPS.start(), *REPS.end())
                            }
                            Err(_) => runs[k] = false,
                        }
                    }

                    if runs[k] {
                        seconds[k] = seconds[k].min(time(&self.timer, &buffers, &trial, reps[k])?);
                    }
                }
            }

            let fastest = (0..seconds.len()).min_by(|&a, &b| seconds[a].total_cmp(&seconds[b]));

            let Some(k) = fastest.filter(|&k| seconds[k].is_finite()) else {
                continue;
            };

            picks().insert(picked, k);

            for i in steps {
                plan.steps_mut()[i].tuned = Some(candidates[k].clone());
            }
        }

        Ok(())
    }
}
