//! Backend-independent kernel autotuning: benchmarks candidate configs of each plan step and keeps the fastest.

pub(crate) mod forced;
#[cfg(feature = "python")]
pub(crate) mod python;

pub(crate) use forced::forcing;

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex, PoisonError};

use crate::codegen::kernel::Runner;
use crate::device::Device;
use crate::graph::Primitive;
use crate::plan::{DevicePlan, Step, StepBase, Tuned};
use crate::runtime::event::event_for;
use crate::runtime::stream::hold;

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

/// What a tuning result is kept under across plans: the kernel `family`'s name, the device and the step's
/// primitive, operand types, views and output type.
fn pick(family: &str, device: Device, step: &StepBase) -> String {
    let types: Vec<String> = step.inputs.iter().map(|(_, ty)| ty.to_string()).collect();

    format!(
        "{family} {device} {} {} {:?} -> {}",
        step.primitive,
        types.join(" "),
        step.views,
        step.output.1
    )
}

/// A plan step that runs a `Kernel` (`codegen::kernel`): its candidates and forced config come from it.
pub(crate) trait KernelStep: Sized {
    /// What the step's kernel runs on.
    type Buffers;

    /// The step's kernel.
    fn kernel(&self) -> &dyn Runner<Self, Self::Buffers>;

    /// The name of the step's kernel, keying its tuning results.
    fn kernel_name(&self) -> &'static str {
        self.kernel().name()
    }

    /// The step's candidate configs.
    fn candidates(&self) -> Vec<Tuned> {
        self.kernel().candidates(self)
    }

    /// The config forced for the step's kernel, if one is and it runs the step.
    fn forced(&self) -> Option<Tuned> {
        self.kernel().forced(self)
    }

    /// Prepares the step's launches under each of `tuned`'s configs.
    fn prepare(&self, tuned: &[Tuned]) -> Result<(), String> {
        self.kernel().prepare(self, tuned)
    }
}

impl<T: Timer> Autotuner<T> {
    /// Sets each step's `tuned` config: a config forced for its kernel, else (when autotuning) its kernel's fastest
    /// candidate, outside the profiler.
    pub(crate) fn tune_steps(&self, plan: &mut DevicePlan<T::Step>) -> Result<(), String>
    where
        T::Step: KernelStep,
    {
        crate::profiler::untraced(|| {
            for step in plan.steps_mut() {
                if let Some(tuned) = step.forced() {
                    step.prepare(std::slice::from_ref(&tuned))?;
                    step.tuned = Some(tuned);
                }
            }

            if !self.autotune {
                return Ok(());
            }

            let jobs = self.jobs(plan, |step| match step.tuned.is_some() {
                true => Vec::new(),
                false => step.candidates(),
            });

            for (steps, candidates) in &jobs {
                plan.steps()[steps[0]].prepare(candidates)?;
            }

            self.time(plan, jobs, |step| step.kernel_name().to_owned())
        })
    }

    /// The steps of `plan` with candidates (from `candidates`), grouped with the later steps of the same kernel,
    /// operand types and views, which take the same pick.
    fn jobs(
        &self,
        plan: &DevicePlan<T::Step>,
        mut candidates: impl FnMut(&T::Step) -> Vec<Tuned>,
    ) -> Vec<(Vec<usize>, Vec<Tuned>)> {
        let mut same: HashMap<String, usize> = HashMap::new();
        let mut jobs: Vec<(Vec<usize>, Vec<Tuned>)> = Vec::new();

        for (i, step) in plan.steps().iter().enumerate() {
            if let Some(&job) = same.get(&key(step)) {
                jobs[job].0.push(i);
                continue;
            }

            let candidates = candidates(step);

            if !candidates.is_empty() {
                same.insert(key(step), jobs.len());
                jobs.push((vec![i], candidates));
            }
        }

        jobs
    }

    /// Times each job's candidates on its first step and sets every step of the job to the fastest (or to the one
    /// picked before for the same `family`'s kernel, device and step).
    fn time(
        &self,
        plan: &mut DevicePlan<T::Step>,
        jobs: Vec<(Vec<usize>, Vec<Tuned>)>,
        family: impl Fn(&T::Step) -> String,
    ) -> Result<(), String> {
        let picks = || PICKS.lock().unwrap_or_else(PoisonError::into_inner);

        for (steps, candidates) in jobs {
            let picked = pick(
                &family(&plan.steps()[steps[0]]),
                self.timer.device(),
                &plan.steps()[steps[0]],
            );

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
