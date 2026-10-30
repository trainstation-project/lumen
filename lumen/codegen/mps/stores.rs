//! Output store helpers shared by the Metal kernels: each output's store through its view.
use super::*;

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
