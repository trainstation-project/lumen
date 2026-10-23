use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock, PoisonError};

use crate::graph::Primitive;
use crate::graph_compiler::plan::Step;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GemmConfig {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub threads: usize,
    pub cols: usize,
}

pub(crate) const TUNER: &str = "mps-gemm";

const MAX_THREADGROUP_BYTES: usize = 32 << 10;

pub(crate) fn configs() -> &'static [GemmConfig] {
    static CONFIGS: OnceLock<Vec<GemmConfig>> = OnceLock::new();
    CONFIGS.get_or_init(|| {
        let mut out = Vec::new();
        for threads in [128, 256, 512] {
            let simds = threads / 32;
            for cols in [1, 2, 4].into_iter().filter(|c| simds % c == 0) {
                let rows = simds / cols;
                for (bm, bn, bk) in [32, 64, 128]
                    .into_iter()
                    .flat_map(|bm| [32, 64, 128].map(move |bn| (bm, bn)))
                    .flat_map(|(bm, bn)| [16, 32].map(move |bk| (bm, bn, bk)))
                {
                    if bm % (rows * 8) != 0 || bn % (cols * 8) != 0 {
                        continue;
                    }
                    let (fm, fn_) = (bm / rows / 8, bn / cols / 8);
                    let fragments = fm * fn_;
                    let loads = (bm * bk) % threads == 0
                        && (bk * bn) % threads == 0
                        && threads % bk == 0
                        && threads % bn == 0;
                    let bytes = ((bm * bk).max(simds * 64) + bk * bn) * 4;
                    if (4..=16).contains(&fragments)
                        && fm <= 2 * fn_
                        && fn_ <= 2 * fm
                        && loads
                        && bytes <= MAX_THREADGROUP_BYTES
                    {
                        out.push(GemmConfig {
                            bm,
                            bn,
                            bk,
                            threads,
                            cols,
                        });
                    }
                }
            }
        }
        out
    })
}

pub(crate) const DEFAULT_CONFIGS: [(&str, GemmConfig); 3] = [
    (
        "",
        GemmConfig {
            bm: 128,
            bn: 64,
            bk: 16,
            threads: 256,
            cols: 2,
        },
    ),
    (
        "_mid",
        GemmConfig {
            bm: 64,
            bn: 64,
            bk: 16,
            threads: 256,
            cols: 2,
        },
    ),
    (
        "_small",
        GemmConfig {
            bm: 32,
            bn: 32,
            bk: 32,
            threads: 256,
            cols: 2,
        },
    ),
];

thread_local! {
    static TRIED: Cell<Option<usize>> = const { Cell::new(None) };
}

static NAMES: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());

pub(crate) fn trying<T>(config: usize, f: impl FnOnce() -> T) -> T {
    TRIED.set(Some(config));
    let result = f();
    TRIED.set(None);
    result
}

pub(crate) fn tile_key(step: &Step) -> String {
    let name = match &step.primitive {
        Primitive::Fusion { name, .. } => name.clone(),
        p => p.to_string(),
    };
    let types: Vec<String> = step.inputs.iter().map(|(_, ty)| ty.to_string()).collect();
    format!("{name} {} -> {}", types.join(" "), step.output.1)
}

pub(crate) fn name_config(step: &Step, config: usize, kernel: &str) {
    let key = format!("{}#{config}", tile_key(step));
    let mut names = NAMES.lock().unwrap_or_else(PoisonError::into_inner);
    names.insert(key, kernel.to_owned());
}

pub(crate) fn tuned(step: &Step) -> Option<(String, GemmConfig)> {
    let key = tile_key(step);
    let config = TRIED
        .get()
        .or_else(|| crate::graph_compiler::common::autotune::picked_for(TUNER, &key))?;
    let names = NAMES.lock().unwrap_or_else(PoisonError::into_inner);
    let name = names.get(&format!("{key}#{config}"))?.clone();
    Some((name, configs()[config]))
}
