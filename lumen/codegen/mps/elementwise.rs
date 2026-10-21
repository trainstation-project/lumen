use super::*;

pub(super) fn elementwise_kernel(
    body: &Graph,
    by_value: &[bool],
    views: &HashMap<usize, (usize, Vec<usize>)>,
) -> (String, String) {
    let mut emitter = Emitter::new(body, by_value);
    let out = body.outputs()[0];
    let result = emitter.value(out, "j".into());
    let extra: Vec<String> = body.outputs()[1..]
        .iter()
        .map(|&v| emitter.value(v, "j".into()))
        .collect();
    let out_type = metal_type(body.type_of(out).dtype);
    let (params, args) = io_params(body, by_value, out_type, &[]);
    let writes = written(body, views, &result, &extra);
    let lines = &emitter.lines;
    named(formatdoc!(
        "
        kernel void NAME({params}ELEMENTWISE_ARGS({args})) {{
            FOR_EACH_ELEMENT(j, {out_type}) {{
                {lines}{writes}
            }}
        }}
        "
    ))
}

pub(super) fn written(
    body: &Graph,
    views: &HashMap<usize, (usize, Vec<usize>)>,
    result: &str,
    extra: &[String],
) -> String {
    let mut writes = String::new();
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
                for (d, (&n, &stride)) in shape.iter().zip(strides).enumerate().rev() {
                    if n == 1 {
                        continue;
                    }
                    let mut idx = match inner {
                        1 => "j".to_string(),
                        _ => format!("(j / {inner}u)"),
                    };
                    if d > 0 && shape[..d].iter().any(|&m| m != 1) {
                        idx = format!("({idx} % {n}u)");
                    }
                    at.push_str(&format!(" + ulong({idx}) * {stride}ul"));
                    inner *= n;
                }
                at
            }
            None => "j".to_string(),
        };
        writeln!(writes, "{out}[{at}] = {value};").unwrap();
    }
    writes
}
