//! Configs for the simdgroup-matrix tiled matmul.

use std::sync::OnceLock;

use super::super::{Algorithm, GemmConfig, MAX_THREADGROUP_BYTES};

/// Every tiled config autotuning tries, filtered to balanced fragment counts, whole cooperative
/// loads and tiles that fit in threadgroup memory.
pub(crate) fn configs() -> &'static [GemmConfig] {
    static CONFIGS: OnceLock<Vec<GemmConfig>> = OnceLock::new();

    CONFIGS.get_or_init(|| {
        let mut out = Vec::new();

        for threads in [64, 128, 256, 512] {
            let simds = threads / 32;

            for cols in [1, 2, 4].into_iter().filter(|c| simds % c == 0) {
                let rows = simds / cols;

                // `rows x cols` simdgroups, each owning `fm x fn_` 8x8 fragments.
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

                    // The tiles' rows are padded by 16 bytes (`SG_PAD`): 4 float32 elements, the most bytes
                    // of any element type.
                    let bytes = ((bm * (bk + 4)).max(bk * (bm + 4)).max(simds * 64)
                        + (bk * (bn + 4)).max(bn * (bk + 4)))
                        * 4;

                    if (4..=16).contains(&fragments)
                        && fm <= 2 * fn_
                        && fn_ <= 2 * fm
                        && loads
                        && bytes <= MAX_THREADGROUP_BYTES
                    {
                        for prefetch in [true, false] {
                            out.push(GemmConfig {
                                bm,
                                bn,
                                bk,
                                threads,
                                cols,
                                prefetch,
                                algorithm: Algorithm::Tiled,
                            });
                        }
                    }
                }
            }
        }

        out
    })
}

/// The precompiled default, mid and small tiled configs, with their kernel-name suffixes.
pub(crate) const DEFAULT_CONFIGS: [(&str, GemmConfig); 3] = [
    (
        "",
        GemmConfig {
            bm: 128,
            bn: 64,
            bk: 16,
            threads: 256,
            cols: 2,
            prefetch: true,
            algorithm: Algorithm::Tiled,
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
            prefetch: true,
            algorithm: Algorithm::Tiled,
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
            prefetch: true,
            algorithm: Algorithm::Tiled,
        },
    ),
];
