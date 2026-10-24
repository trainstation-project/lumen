use std::sync::OnceLock;

use super::super::{Algorithm, GemmConfig};

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
