//! Autotuning configs for the Metal Performance Primitives matmul.

use std::sync::OnceLock;

use super::super::{Algorithm, GemmConfig};

/// Every MPP config autotuning tries: each (tile, simdgroups) pair with K steps of 32, 64 and 128.
pub(crate) fn mpp_configs() -> &'static [GemmConfig] {
    static CONFIGS: OnceLock<Vec<GemmConfig>> = OnceLock::new();

    CONFIGS.get_or_init(|| {
        let tiles = [
            (64, 64, 4),
            (64, 32, 2),
            (32, 64, 2),
            (128, 64, 4),
            (128, 128, 8),
        ];

        tiles
            .into_iter()
            .flat_map(|(bm, bn, simds)| {
                [32, 64, 128].map(move |bk| GemmConfig {
                    bm,
                    bn,
                    bk,
                    threads: simds * 32,
                    cols: 1,
                    prefetch: false,
                    algorithm: Algorithm::Mpp,
                })
            })
            .collect()
    })
}
