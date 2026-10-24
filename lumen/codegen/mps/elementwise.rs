use super::*;

const GRID_THREADS: usize = 32 << 10;
const LOOP_GROUPS: [usize; 5] = [64, 128, 256, 512, 1024];
const LOOP_GROUP: usize = 256;

fn most_per_thread(body: &Graph) -> usize {
    let dtype = body.type_of(body.outputs()[0]).dtype;
    (crate::kernels::mps::BYTES_PER_THREAD / dtype.size_of()).max(1)
}

pub(crate) fn loop_tuning(body: &Graph) -> Tuning {
    let n = body.type_of(body.outputs()[0]).numel();
    Tuning::Loop {
        per_thread: (n / GRID_THREADS).clamp(1, most_per_thread(body)),
        group: LOOP_GROUP,
        contiguous: false,
    }
}

pub(crate) fn loop_tunings(body: &Graph) -> Vec<Tuning> {
    let most = most_per_thread(body);
    let per_threads = (0..).map(|k| 1 << k).take_while(|&p| p <= most);
    per_threads
        .flat_map(|per_thread| {
            let contiguous = match per_thread {
                1 => &[false][..],
                _ => &[false, true][..],
            };
            contiguous.iter().flat_map(move |&contiguous| {
                LOOP_GROUPS.map(|group| Tuning::Loop {
                    per_thread,
                    group,
                    contiguous,
                })
            })
        })
        .collect()
}

pub(super) fn elementwise_kernel(
    body: &Graph,
    by_value: &[bool],
    views: &HashMap<usize, (usize, Vec<usize>)>,
    tuning: Tuning,
) -> (String, String) {
    let Tuning::Loop {
        per_thread,
        group,
        contiguous,
    } = tuning;
    let each = match contiguous {
        true => {
            format!("for (uint j = i * {per_thread}u, k = 0; k < {per_thread}u && j < n; ++k, ++j)")
        }
        false => format!("for (uint j = i, k = 0; k < {per_thread}u && j < n; ++k, j += threads)"),
    };
    let mut emitter = Emitter::new(body, by_value);
    let out = body.outputs()[0];
    let result = emitter.value(out, "j".into());
    let extra: Vec<String> = body.outputs()[1..]
        .iter()
        .map(|&v| emitter.value(v, "j".into()))
        .collect();
    let out_type = metal_type(body.type_of(out).dtype);
    let (params, args) = io_params(body, by_value, out_type, &[]);
    let (stores, writes) = written(body, views, &result, &extra);
    let lines = &emitter.lines;
    named(formatdoc!(
        "
        {stores}
        [[max_total_threads_per_threadgroup({group})]]
        kernel void NAME(
            {params}ELEMENTWISE_ARGS({args})
        ) {{
            {each} {{
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
