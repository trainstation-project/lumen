//! The TPU-tiled memory layout of tensors in the shared region, and host conversion to and from it.

use crate::DType;

/// Byte granularity of untiled (rank 0 and 1) arrays.
const ROW: usize = 512;

/// Matrices at most this many 128-column tiles wide are aligned to whole tile rows.
const ALIGNED_COLUMNS: usize = 40;

/// How a tensor of `dtype` and `shape` is stored on a TPU.
#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub shape: Vec<usize>,
    /// The logical dtype the host sees.
    pub dtype: DType,
    /// The dtype in memory: bools are stored as int32.
    pub stored: DType,
    /// Rows of one tile: 8 sublanes times `pack`.
    pub rows: usize,
    /// Values packed into one 32-bit word.
    pub pack: usize,
}

impl Layout {
    /// The layout of `dtype` and `shape`; fails for 64-bit dtypes, which TPUs don't store.
    pub fn new(dtype: DType, shape: &[usize]) -> Result<Self, String> {
        let stored = match dtype {
            DType::Bool => DType::I32,
            DType::U64 | DType::I64 | DType::F64 => {
                return Err(format!(
                    "{dtype} is not stored on a TPU: convert it to a 32-bit dtype"
                ));
            }
            other => other,
        };

        let pack = 32 / stored.bits();

        Ok(Layout {
            shape: shape.to_vec(),
            dtype,
            stored,
            rows: 8 * pack,
            pack,
        })
    }

    /// Whether the tensor is stored in `rows x 128` tiles over its last two dims (rank 2 and up).
    pub fn tiled(&self) -> bool {
        self.shape.len() >= 2
    }

    /// The leading dims' product and the last two dims padded to whole tiles.
    fn padded(&self) -> (usize, usize, usize) {
        let r = self.shape.len();
        let lead = self.shape[..r - 2].iter().product();
        let m = self.shape[r - 2].div_ceil(self.rows) * self.rows;
        let n = self.shape[r - 1].div_ceil(128) * 128;
        (lead, m, n)
    }

    /// Bytes the tensor takes in memory, padding included.
    pub fn bytes(&self) -> usize {
        let bits = self.stored.bits();

        if self.tiled() {
            let (lead, m, n) = self.padded();
            return lead * m * n * bits / 8;
        }

        let numel: usize = self.shape.iter().product();
        (numel * bits).div_ceil(8 * ROW) * ROW
    }

    /// Each element's position, in stored values, in row-major order.
    fn positions(&self) -> Vec<usize> {
        let numel: usize = self.shape.iter().product();

        if !self.tiled() {
            // Untiled arrays interleave `pack` 128-lane rows within each 32-bit word.
            let tile = 128 * self.pack;

            return (0..numel)
                .map(|i| {
                    let within = i % tile;
                    i - within + (within % 128) * self.pack + within / 128
                })
                .collect();
        }

        let r = self.shape.len();
        let (m, n) = (self.shape[r - 2], self.shape[r - 1]);
        let (_, mp, np) = self.padded();
        let tm = self.rows;
        let tile = tm * 128;
        let (tiles_m, tiles_n) = (mp / tm, np / 128);
        let mut out = Vec::with_capacity(numel);
        let lead = numel / (m * n).max(1);

        // Tiles are row-major over (lead, tile row, tile column); within a tile, `pack` consecutive rows share a word.
        for l in 0..lead {
            for i in 0..m {
                let (ti, within) = (i / tm, i % tm);
                let (word_row, sub) = (within / self.pack, within % self.pack);

                for j in 0..n {
                    let t = (l * tiles_m + ti) * tiles_n + j / 128;
                    out.push(t * tile + (word_row * 128 + j % 128) * self.pack + sub);
                }
            }
        }

        out
    }

    /// The host bytes `host` rearranged into the tiled memory image.
    pub fn tile(&self, host: &[u8]) -> Vec<u8> {
        let (from, bits) = (self.dtype.size_of(), self.stored.bits());
        let mut out = vec![0u8; self.bytes()];

        for (k, p) in self.positions().into_iter().enumerate() {
            let src = &host[k * from..(k + 1) * from];

            match self.dtype {
                DType::Bool => out[p * 4] = u8::from(src[0] != 0),
                _ if bits < 8 => {
                    let mask = (1u8 << bits) - 1;
                    out[p * bits / 8] |= (src[0] & mask) << (p * bits % 8);
                }
                _ => out[p * from..(p + 1) * from].copy_from_slice(src),
            }
        }

        out
    }

    /// Copies the tiled memory image `memory` back into row-major `host` bytes.
    pub fn untile(&self, memory: &[u8], host: &mut [u8]) {
        let (to, bits) = (self.dtype.size_of(), self.stored.bits());

        for (k, p) in self.positions().into_iter().enumerate() {
            let dst = &mut host[k * to..(k + 1) * to];

            match self.dtype {
                DType::Bool => {
                    dst[0] = u8::from(memory[p * 4..(p + 1) * 4].iter().any(|&b| b != 0))
                }
                _ if bits < 8 => {
                    let mask = (1u8 << bits) - 1;
                    dst[0] = (memory[p * bits / 8] >> (p * bits % 8)) & mask;
                }
                _ => dst.copy_from_slice(&memory[p * to..(p + 1) * to]),
            }
        }
    }

    /// The shape of one tile: none for scalars, `128 * pack` for vectors, `rows x 128` for matrices.
    pub fn tile_shape(&self) -> Vec<usize> {
        match self.shape.len() {
            0 => vec![],
            1 => vec![128 * self.pack],
            _ => vec![self.rows, 128],
        }
    }

    /// The Mosaic tiling string and the tile strides of each dim.
    pub fn tiles(&self) -> (String, Vec<usize>) {
        if self.shape.is_empty() {
            return (String::new(), vec![]);
        }

        if !self.tiled() {
            let tiles = match self.pack {
                1 => "(128)".into(),
                p => format!("({})(128)({p},1)", 128 * p),
            };

            return (tiles, vec![1]);
        }

        let r = self.shape.len();
        let (_, mp, np) = self.padded();
        let (tiles_m, tiles_n) = (mp / self.rows, np / 128);
        // The last two dims stride over tiles, the leading dims over whole matrices.
        let mut strides = vec![1, tiles_n];
        let mut stride = tiles_m * tiles_n;

        for d in (0..r - 2).rev() {
            strides.push(stride);
            stride *= self.shape[d];
        }

        strides.reverse();

        let tiles = match self.pack {
            1 => format!("({},128)", self.rows),
            p => format!("({},128)({p},1)", self.rows),
        };

        (tiles, strides)
    }

    /// The `#tpu.tiled` memory layout attribute Mosaic kernels view the tensor with.
    pub fn mosaic(&self) -> String {
        let (tiles, strides) = self.tiles();
        let list: Vec<String> = strides.iter().map(usize::to_string).collect();
        format!("#tpu.tiled<{tiles},[{}]>", list.join(","))
    }
}

/// Bytes a tensor of `dtype` and `shape` takes on a TPU.
pub fn bytes(dtype: DType, shape: &[usize]) -> Result<usize, String> {
    Ok(Layout::new(dtype, shape)?.bytes())
}

/// Byte alignment for a tensor of `dtype` and `shape`: narrow matrices align to eight tile rows'
/// worth of their columns, everything else to `ROW`.
pub fn alignment(dtype: DType, shape: &[usize]) -> usize {
    let Ok(layout) = Layout::new(dtype, shape) else {
        return ROW;
    };

    if !layout.tiled() {
        return ROW;
    }

    let columns = shape[shape.len() - 1].div_ceil(128);

    match columns <= ALIGNED_COLUMNS {
        true => 8 * columns * ROW,
        false => ROW,
    }
}

/// Whether `a` and `b` share dtype and element placement, so one's memory can be read as the other.
pub fn same(a: &crate::graph::TensorType, b: &crate::graph::TensorType) -> bool {
    let layout =
        |t: &crate::graph::TensorType| Layout::new(t.dtype, &t.shape).map(|l| l.positions());

    a.dtype == b.dtype && matches!((layout(a), layout(b)), (Ok(x), Ok(y)) if x == y)
}
