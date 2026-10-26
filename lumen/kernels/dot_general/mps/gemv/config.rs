//! Configs for the matrix-vector (GEMV) kernel.

use std::sync::OnceLock;

use super::super::{Algorithm, GemmConfig};

/// Maximum number of vectors one GEMV launch multiplies at once.
pub(crate) const GEMV_VECTORS: usize = 8;

/// Elements each lane loads along K per step.
pub(crate) const GEMV_WIDTH: usize = 8;

/// GEMV config with `rows` matrix rows per threadgroup split over `simds` simdgroups.
fn config(vector_lhs: bool, rows: usize, simds: usize) -> GemmConfig {
    // The vector side always spans GEMV_VECTORS; the matrix side spans `rows`.
    let (bm, bn) = match vector_lhs {
        true => (GEMV_VECTORS, rows),
        false => (rows, GEMV_VECTORS),
    };
    GemmConfig {
        bm,
        bn,
        bk: GEMV_WIDTH,
        threads: simds * 32,
        cols: 1,
        prefetch: false,
        algorithm: Algorithm::Gemv,
    }
}

/// Every GEMV config autotuning tries, for the given vector side.
pub(crate) fn gemv_configs(vector_lhs: bool) -> &'static [GemmConfig] {
    static CONFIGS: OnceLock<[Vec<GemmConfig>; 2]> = OnceLock::new();
    let all = CONFIGS.get_or_init(|| {
        [false, true].map(|vector_lhs| {
            let mut out = Vec::new();
            for rows in [8, 16, 32, 64] {
                for simds in [1, 2, 4, 8] {
                    // Each simdgroup owns whole rows, at most eight of them.
                    if rows % simds == 0 && rows / simds <= 8 {
                        out.push(config(vector_lhs, rows, simds));
                    }
                }
            }
            out
        })
    });
    &all[usize::from(vector_lhs)]
}

/// GEMV config used without autotuning.
pub(crate) fn gemv_default(vector_lhs: bool) -> GemmConfig {
    config(vector_lhs, 16, 4)
}

/// GEMV config used when the epilogue writes whole row tiles (64 rows, vector lhs).
pub(crate) fn gemv_tiles_default() -> GemmConfig {
    config(true, 64, 8)
}

/// Matrix rows each threadgroup of `c` covers.
pub(crate) fn gemv_rows(c: &GemmConfig, vector_lhs: bool) -> usize {
    match vector_lhs {
        true => c.bn,
        false => c.bm,
    }
}
