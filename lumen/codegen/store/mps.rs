//! Metal loads' and stores' addressing, shared by the generated kernels: the strided offset of a flat index,
//! each input's load helper and each output's store through its view.

use std::collections::HashMap;
use std::fmt::Write;

use indoc::{formatdoc, writedoc};

use crate::graph::Graph;
use crate::tensor::contiguous_strides;

/// Per-output `NAME_store{k}` helpers and the calls storing `result` and `extra` at index `j`;
/// outputs in `views` are written at their offset and strides instead of contiguously.
pub(crate) fn written(
    body: &Graph,
    views: &HashMap<usize, (usize, Vec<usize>)>,
    result: &str,
    extra: &[String],
) -> (String, String) {
    let (mut stores, mut writes) = (String::new(), String::new());

    for (k, value) in std::iter::once(result)
        .chain(extra.iter().map(String::as_str))
        .enumerate()
    {
        let out = match k {
            0 => "out".to_string(),
            _ => format!("out{k}"),
        };

        let at = match views.get(&k) {
            Some((offset, strides)) => {
                let shape = &body.type_of(body.outputs()[k]).shape;
                let mut at = format!("{offset}ul");
                let mut inner = 1;

                // Decompose the flat index into coordinates innermost-first, skipping size-1 dims; the modulo
                // is dropped on the outermost non-trivial dim.
                for (d, (&n, &stride)) in shape.iter().zip(strides).enumerate().rev() {
                    if n == 1 {
                        continue;
                    }

                    let mut idx = match inner {
                        1 => "i".to_string(),
                        _ => format!("(i / {inner}u)"),
                    };

                    if d > 0 && shape[..d].iter().any(|&m| m != 1) {
                        idx = format!("({idx} % {n}u)");
                    }

                    at.push_str(&format!(" + ulong({idx}) * {stride}ul"));
                    inner *= n;
                }

                at
            }
            None => "i".to_string(),
        };

        writedoc!(
            stores,
            "
            template <typename T, typename I, typename V>
            inline void NAME_store{k}(device T *p, I i, V v) {{
                p[{at}] = v;
            }}
            "
        )
        .unwrap();

        writeln!(writes, "NAME_store{k}({out}, j, {value});").unwrap();
    }

    (stores, writes)
}

/// Metal expression of the strided offset for flat index `idx` into `shape`, merging adjacent
/// dims that stay contiguous under `strides` into one divide/modulo term.
pub(crate) fn gather_index(idx: &str, shape: &[usize], strides: &[usize]) -> String {
    let spans = contiguous_strides(shape);

    struct Run {
        first: usize,
        last: usize,
        size: usize,
        stride: usize,
    }

    let mut runs: Vec<Run> = Vec::new();

    for d in (0..shape.len()).filter(|&d| shape[d] != 1 && strides[d] != 0) {
        match runs.last_mut() {
            Some(run)
                if run.stride == strides[d] * shape[d]
                    && shape[run.last + 1..d].iter().all(|&n| n == 1) =>
            {
                run.last = d;
                run.size *= shape[d];
                run.stride = strides[d];
            }
            _ => runs.push(Run {
                first: d,
                last: d,
                size: shape[d],
                stride: strides[d],
            }),
        }
    }

    let terms: Vec<String> = runs
        .iter()
        .map(|run| {
            let mut c = idx.to_string();

            if spans[run.last] != 1 {
                c = format!("{c} / {}u", spans[run.last]);
            }

            if shape[..run.first].iter().any(|&n| n != 1) {
                c = format!("({c}) % {}u", run.size);
            }

            if run.stride != 1 {
                c = format!("({c}) * {}u", run.stride);
            }

            c
        })
        .collect();

    if terms.is_empty() {
        "0".into()
    } else {
        terms.join(" + ")
    }
}

/// Opening of a call to input `k`'s load helper, to be followed by the index and `)`.
pub(crate) fn load(k: usize) -> String {
    format!("NAME_load{k}(in{k}, ")
}

/// Load helpers for every `NAME_load{k}` that `source` calls, each indexing contiguously.
pub(crate) fn loads(source: &str) -> String {
    let mut ks: Vec<usize> = source
        .match_indices("NAME_load")
        .filter_map(|(at, _)| {
            let digits: String = source[at + "NAME_load".len()..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();

            digits.parse().ok()
        })
        .collect();

    ks.sort_unstable();
    ks.dedup();
    ks.iter().map(|&k| load_helper("NAME", k, "i")).collect()
}

/// Metal helper `{name}_load{k}` that reads `p` at `index` (an expression in `i`).
fn load_helper(name: &str, k: usize, index: &str) -> String {
    formatdoc!(
        "
        template <typename T, typename I>
        inline T {name}_load{k}(device const T *p, I i) {{
            return p[{index}];
        }}
        "
    )
}

/// Rewrites input `k`'s contiguous load helper in `source` to read through `strides`, or `None`
/// if `source` has no such helper.
pub(crate) fn strided_load(
    source: &str,
    name: &str,
    k: usize,
    shape: &[usize],
    strides: &[usize],
) -> Option<String> {
    let index = gather_index("i", shape, strides);
    let plain = load_helper(name, k, "i");
    let strided = load_helper(name, k, &index);

    source
        .contains(&plain)
        .then(|| source.replace(&plain, &strided))
}
