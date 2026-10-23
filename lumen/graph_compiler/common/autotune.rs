use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, PoisonError};

use crate::graph::Primitive;
use crate::graph_compiler::Plan;
use crate::graph_compiler::plan::Step;

pub(crate) trait Kernels {
    type Config: Copy + Send + 'static;

    const NAME: &'static str;

    fn configs(&self, step: &Step) -> Vec<Self::Config>;
    fn build(&mut self, step: &Step, config: Self::Config) -> String;
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

type Picked = (String, Box<dyn Any + Send>);

static PICKED: Mutex<Option<HashMap<String, Picked>>> = Mutex::new(None);

thread_local! {
    static TRYING: RefCell<Option<(String, Picked)>> = const { RefCell::new(None) };
}

fn key(family: &str, step: &Step) -> String {
    let name = match &step.primitive {
        Primitive::Fusion { name, .. } => name.clone(),
        p => p.to_string(),
    };
    let types: Vec<String> = step.inputs.iter().map(|(_, ty)| ty.to_string()).collect();
    format!("{family}/{name} {} -> {}", types.join(" "), step.output.1)
}

pub(crate) fn tuned<C: Copy + 'static>(family: &str, step: &Step) -> Option<(String, C)> {
    let key = key(family, step);
    let of = |(kernel, config): &Picked| Some((kernel.clone(), *config.downcast_ref::<C>()?));
    let trying = TRYING.with_borrow(|t| t.as_ref().filter(|(k, _)| *k == key).and_then(|(_, p)| of(p)));
    trying.or_else(|| {
        let picked = PICKED.lock().unwrap_or_else(PoisonError::into_inner);
        picked.as_ref()?.get(&key).and_then(of)
    })
}

fn is_picked(key: &str) -> bool {
    let picked = PICKED.lock().unwrap_or_else(PoisonError::into_inner);
    picked.as_ref().is_some_and(|p| p.contains_key(key))
}

fn trying<C: Send + 'static, R>(key: &str, kernel: &str, config: C, f: impl FnOnce() -> R) -> R {
    TRYING.set(Some((key.to_owned(), (kernel.to_owned(), Box::new(config)))));
    let result = f();
    TRYING.set(None);
    result
}

impl<T: Timer> Autotuner<T> {
    pub(crate) fn tune<K: Kernels>(&self, kernels: &mut K, plan: &Plan) -> Result<(), String> {
        let mut seen = HashSet::new();
        let mut jobs = Vec::new();
        for step in plan.steps() {
            let key = key(K::NAME, step);
            if is_picked(&key) || !seen.insert(key.clone()) {
                continue;
            }
            let built: Vec<(String, K::Config)> = kernels
                .configs(step)
                .into_iter()
                .map(|config| (kernels.build(step, config), config))
                .collect();
            if !built.is_empty() {
                jobs.push((step, key, built));
            }
        }
        if jobs.is_empty() {
            return Ok(());
        }
        kernels.compile()?;
        for (step, key, built) in jobs {
            let Some(buffers) = self.timer.buffers(step)? else {
                continue;
            };
            let mut reps = vec![0; built.len()];
            let mut seconds = vec![f64::INFINITY; built.len()];
            for _ in 0..PASSES {
                for (k, (kernel, config)) in built.iter().enumerate() {
                    let t = trying(&key, kernel, *config, || {
                        if reps[k] == 0 {
                            let warm = self.timer.time(&buffers, step, 1)?;
                            reps[k] = ((BUDGET_SECONDS / warm) as usize).clamp(*REPS.start(), *REPS.end());
                        }
                        self.timer.time(&buffers, step, reps[k])
                    })?;
                    seconds[k] = seconds[k].min(t);
                }
            }
            let fastest = (0..seconds.len()).min_by(|&a, &b| seconds[a].total_cmp(&seconds[b]));
            let Some(k) = fastest.filter(|&k| seconds[k].is_finite()) else {
                continue;
            };
            let (kernel, config) = built.into_iter().nth(k).expect("the fastest config");
            let mut picked = PICKED.lock().unwrap_or_else(PoisonError::into_inner);
            picked
                .get_or_insert_with(HashMap::new)
                .insert(key, (kernel, Box::new(config)));
        }
        Ok(())
    }
}
