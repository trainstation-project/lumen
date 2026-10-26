use std::collections::HashMap;

use crate::device::Device;
use crate::graph::Primitive;
use crate::graph_compiler::Plan;
use crate::graph_compiler::plan::{Step, Tuned};
use crate::stream::{event_for, hold};

pub(crate) trait Kernels {
    type Config;

    fn configs(&self, step: &Step) -> Vec<Self::Config>;
    fn build(&mut self, step: &Step, config: Self::Config) -> Tuned;
    fn compile(&mut self) -> Result<(), String> {
        Ok(())
    }
}

pub(crate) trait Timer {
    type Buffers;

    fn device(&self) -> Device;
    fn buffers(&self, step: &Step) -> Result<Option<Self::Buffers>, String>;
    fn run(&self, buffers: &Self::Buffers, step: &Step, rep: usize) -> Result<(), String>;
}

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

pub(crate) struct Autotuner<T> {
    pub timer: T,
}

const PASSES: usize = 2;
const BUDGET_SECONDS: f64 = 1e-3;
const REPS: std::ops::RangeInclusive<usize> = 2..=256;

fn key(step: &Step) -> String {
    let name = match &step.primitive {
        Primitive::Fusion { name, .. } => name.clone(),
        p => p.to_string(),
    };
    let types: Vec<String> = step.inputs.iter().map(|(_, ty)| ty.to_string()).collect();
    format!("{name} {} -> {}", types.join(" "), step.output.1)
}

impl<T: Timer> Autotuner<T> {
    pub(crate) fn tune<K: Kernels>(&self, kernels: &mut K, plan: &mut Plan) -> Result<(), String> {
        crate::profiler::untraced(|| self.tune_untraced(kernels, plan))
    }

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
        for (steps, candidates) in jobs {
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
            for i in steps {
                plan.steps_mut()[i].tuned = Some(candidates[k].clone());
            }
        }
        Ok(())
    }
}
