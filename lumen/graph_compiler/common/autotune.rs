use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use crate::graph_compiler::Plan;
use crate::graph_compiler::plan::Step;

pub(crate) trait Autotuner {
    type Config: Copy;

    fn device(&self) -> &'static str;
    fn key(&self, step: &Step) -> Option<String>;
    fn candidates(&self, step: &Step) -> Vec<Self::Config>;
    fn generate(&mut self, step: &Step, config: Self::Config) -> String;
    fn compile(&mut self) -> Result<(), String>;
    fn time(&self, step: &Step, kernels: &[String]) -> Result<Vec<f64>, String>;
    fn apply(&self, step: &mut Step, kernel: &str, config: Self::Config);
}

const PASSES: usize = 2;

static PICKED: Mutex<Option<HashMap<String, usize>>> = Mutex::new(None);

fn picked(key: &str) -> Option<usize> {
    let picked = PICKED.lock().unwrap_or_else(PoisonError::into_inner);
    picked.as_ref()?.get(key).copied()
}

pub(crate) fn picked_for(device: &str, key: &str) -> Option<usize> {
    picked(&format!("{device}/{key}"))
}

fn pick(key: String, k: usize) {
    let mut picked = PICKED.lock().unwrap_or_else(PoisonError::into_inner);
    picked.get_or_insert_with(HashMap::new).insert(key, k);
}

struct Job<C> {
    step: usize,
    key: String,
    candidates: Vec<(usize, C)>,
    kernels: Vec<String>,
}

pub(crate) fn autotune<T: Autotuner>(tuner: &mut T, plan: &mut Plan) -> Result<(), String> {
    let mut jobs = Vec::new();
    for (i, step) in plan.steps().iter().enumerate() {
        let Some(key) = tuner.key(step) else {
            continue;
        };
        let key = format!("{}/{key}", tuner.device());
        let all = tuner.candidates(step);
        let candidates: Vec<(usize, T::Config)> = match picked(&key) {
            Some(k) if k < all.len() => vec![(k, all[k])],
            _ => all.into_iter().enumerate().collect(),
        };
        if candidates.is_empty() {
            continue;
        }
        let kernels = candidates
            .iter()
            .map(|&(_, config)| tuner.generate(step, config))
            .collect();
        jobs.push(Job {
            step: i,
            key,
            candidates,
            kernels,
        });
    }
    if jobs.is_empty() {
        return Ok(());
    }
    tuner.compile()?;
    for job in jobs {
        let fastest = match job.kernels.len() {
            1 => 0,
            _ => {
                let mut seconds = vec![f64::INFINITY; job.kernels.len()];
                for _ in 0..PASSES {
                    let timed = tuner.time(&plan.steps()[job.step], &job.kernels)?;
                    for (best, t) in seconds.iter_mut().zip(timed) {
                        *best = best.min(t);
                    }
                }
                let fastest = (0..seconds.len()).min_by(|&a, &b| seconds[a].total_cmp(&seconds[b]));
                match fastest {
                    Some(k) if seconds[k].is_finite() => k,
                    _ => continue,
                }
            }
        };
        let (k, config) = job.candidates[fastest];
        pick(job.key, k);
        tuner.apply(
            &mut plan.steps_mut()[job.step],
            &job.kernels[fastest],
            config,
        );
    }
    Ok(())
}
