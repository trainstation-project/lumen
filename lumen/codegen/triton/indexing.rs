use super::*;
use crate::graph::TensorType;

fn scalar(k: usize, by_value: &[bool]) -> String {
    match by_value[k] {
        true => format!("tl.full([], in{k}, tl.int64)"),
        false => format!("LOAD{k}(in{k}, 0, None).to(tl.int64)"),
    }
}

pub(crate) fn update(x: &TensorType, update: &TensorType, by_value: &[bool]) -> (String, String) {
    let (spans, strides) = (
        contiguous_strides(&update.shape),
        contiguous_strides(&x.shape),
    );
    let terms: Vec<String> = (0..x.shape.len())
        .filter(|&d| x.shape[d] != 1)
        .map(|d| {
            let start = scalar(2 + d, by_value);
            let most = x.shape[d] - update.shape[d];
            format!(
                "((j // {}) % {} + tl.minimum(tl.maximum({start}, 0), {most})) * {}",
                spans[d], update.shape[d], strides[d]
            )
        })
        .collect();
    let at = match terms.is_empty() {
        true => "j * 0".to_string(),
        false => terms.join(" + "),
    };
    let params = params(by_value, None, true);
    let source = formatdoc!(
        "
        @triton.jit
        def NAME(
        {params}):
            j = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = j < n
            STORE(out, {at}, LOAD1(in1, j, mask), mask)
        "
    );
    named("update", source)
}

pub(crate) fn scatter_add(x: &TensorType, axis: usize, picks: usize) -> (String, String) {
    let after: usize = x.shape[axis + 1..].iter().product();
    let length = x.shape[axis];
    let params = params(&[false; 3], None, true).replace(
        "    BLOCK_SIZE: tl.constexpr,",
        "    OUTLOAD: tl.constexpr,\n    BLOCK_SIZE: tl.constexpr,",
    );
    let source = formatdoc!(
        "
        @triton.jit
        def NAME(
        {params}):
            j = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = j < n
            o = j // {after}
            r = j % {after}
            for t in range(0, {picks}):
                k = LOAD1(in1, t, None).to(tl.int64)
                at = (o * {length} + k) * {after} + r
                value = OUTLOAD(out, at, mask) + LOAD2(in2, (o * {picks} + t) * {after} + r, mask)
                STORE(out, at, value, mask)
        "
    );
    named("scatter_add", source)
}
