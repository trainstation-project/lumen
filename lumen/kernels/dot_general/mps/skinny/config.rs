use std::sync::OnceLock;

use super::super::{Algorithm, GemmConfig, MAX_THREADGROUP_BYTES};

pub(crate) fn skinny_configs() -> &'static [GemmConfig] {
    static CONFIGS: OnceLock<Vec<GemmConfig>> = OnceLock::new();
    CONFIGS.get_or_init(|| {
        let mut out = Vec::new();
        for rows in [1, 2, 4, 8] {
            for threads in [128, 256] {
                for block in [32, 64, 128, 256] {
                    for width in [4, 8] {
                        let groups = block / width;
                        if groups <= threads
                            && (block + 1) * (threads / groups) * 4 <= MAX_THREADGROUP_BYTES
                        {
                            out.push(GemmConfig {
                                bm: rows,
                                bn: block,
                                bk: width,
                                threads,
                                cols: 1,
                                prefetch: false,
                                algorithm: Algorithm::Skinny,
                            });
                        }
                    }
                }
            }
        }
        out
    })
}
