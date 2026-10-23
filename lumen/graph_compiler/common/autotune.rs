use std::collections::HashMap;

use crate::graph::Primitive;
use crate::graph_compiler::Plan;
use crate::graph_compiler::plan::{Step, Tuned};

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

    fn buffers(&self, step: &Step) -> Result<Option<Self::Buffers>, String>;
    fn time(&self, buffers: &Self::Buffers, step: &Step, reps: usize) -> Result<f64, String>;
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
            for _ in 0..PASSES {
                for (k, candidate) in candidates.iter().enumerate() {
                    trial.tuned = Some(candidate.clone());
                    if reps[k] == 0 {
                        let warm = self.timer.time(&buffers, &trial, 1)?;
                        reps[k] = ((BUDGET_SECONDS / warm) as usize).clamp(*REPS.start(), *REPS.end());
                    }
                    seconds[k] = seconds[k].min(self.timer.time(&buffers, &trial, reps[k])?);
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
