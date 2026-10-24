use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Tuning {
    Loop {
        per_thread: usize,
        group: usize,
        contiguous: bool,
    },
}

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
