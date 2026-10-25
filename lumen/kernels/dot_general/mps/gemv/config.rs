use std::sync::OnceLock;

use super::super::{Algorithm, GemmConfig};

pub(crate) const GEMV_VECTORS: usize = 8;

pub(crate) const GEMV_WIDTH: usize = 8;

fn config(vector_lhs: bool, rows: usize, simds: usize) -> GemmConfig {
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

pub(crate) fn gemv_configs(vector_lhs: bool) -> &'static [GemmConfig] {
    static CONFIGS: OnceLock<[Vec<GemmConfig>; 2]> = OnceLock::new();
    let all = CONFIGS.get_or_init(|| {
        [false, true].map(|vector_lhs| {
            let mut out = Vec::new();
            for rows in [8, 16, 32, 64] {
                for simds in [1, 2, 4, 8] {
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

pub(crate) fn gemv_default(vector_lhs: bool) -> GemmConfig {
    config(vector_lhs, 16, 4)
}

pub(crate) fn gemv_tiles_default() -> GemmConfig {
    config(true, 64, 8)
}

pub(crate) fn gemv_rows(c: &GemmConfig, vector_lhs: bool) -> usize {
    match vector_lhs {
        true => c.bn,
        false => c.bm,
    }
}
