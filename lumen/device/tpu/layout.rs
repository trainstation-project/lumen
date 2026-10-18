//! How a TPU lays a tensor out in its memory: in the TPU's own tiles, as XLA
//! does, so lumen's kernels (Mosaic) read tensors as they are. Other devices
//! keep row-major bytes; a TPU's tensor's bytes are these, at its offset in
//! the TPU allocator's region, which a kernel views as a tiled array
//! (`tpu.reinterpret_cast`, `lumen/pallas`).
//!
//! * A tensor of two or more dimensions `[..., M, N]` is tiled on its last
//!   two: tiles of `tm × 128` elements (`tm` 8 for 32-bit dtypes, 16 for
//!   16-bit, 32 for 8-bit), `M` and `N` padded to whole tiles. The tiles
//!   follow each other row-major (each of the leading dimensions' indices a
//!   matrix of tiles), each tile its 128-wide rows in order; a 32-bit word
//!   of a tile holds `pack` (`32 / bits`) elements of one column, of rows
//!   one after another (the first in its low bits).
//! * A tensor of one dimension (or none: one element) is its elements in
//!   order, padded to whole rows of 512 bytes.
//! * bool is stored as int32 (0 or 1), as Mosaic loads it; 64-bit dtypes
//!   are not stored on a TPU.

use crate::DType;

/// A row of the TPU allocator's memory: what every allocation is whole of.
const ROW: usize = 512;

/// A tensor's layout on a TPU.
#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    /// Its shape (a scalar's `[]`).
    pub shape: Vec<usize>,
    /// Its dtype.
    pub dtype: DType,
    /// The dtype of its elements in memory (bool's int32).
    pub stored: DType,
    /// A tile's rows (`tm`), for two dimensions or more.
    pub rows: usize,
    /// Elements of a 32-bit word (`32 / bits`), down a tile's column.
    pub pack: usize,
}

impl Layout {
    /// `dtype` of `shape`'s layout; an error for what a TPU does not store.
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
        let pack = 4 / stored.size_of();
        Ok(Layout {
            shape: shape.to_vec(),
            dtype,
            stored,
            rows: 8 * pack,
            pack,
        })
    }

    /// Whether it is tiled (two dimensions or more).
    pub fn tiled(&self) -> bool {
        self.shape.len() >= 2
    }

    /// Its leading dimensions' elements, `M` padded and `N` padded (tiled).
    fn padded(&self) -> (usize, usize, usize) {
        let r = self.shape.len();
        let lead = self.shape[..r - 2].iter().product();
        let m = self.shape[r - 2].div_ceil(self.rows) * self.rows;
        let n = self.shape[r - 1].div_ceil(128) * 128;
        (lead, m, n)
    }

    /// Its bytes in memory: whole tiles, or whole rows.
    pub fn bytes(&self) -> usize {
        let size = self.stored.size_of();
        if self.tiled() {
            let (lead, m, n) = self.padded();
            return lead * m * n * size;
        }
        let numel: usize = self.shape.iter().product();
        (numel * size).div_ceil(ROW) * ROW
    }

    /// Its elements' positions in memory, in elements of `stored`, in
    /// row-major order of the tensor.
    fn positions(&self) -> Vec<usize> {
        let numel: usize = self.shape.iter().product();
        if !self.tiled() {
            return (0..numel).collect();
        }
        let r = self.shape.len();
        let (m, n) = (self.shape[r - 2], self.shape[r - 1]);
        let (_, mp, np) = self.padded();
        let tm = self.rows;
        let tile = tm * 128;
        let (tiles_m, tiles_n) = (mp / tm, np / 128);
        let mut out = Vec::with_capacity(numel);
        let lead = numel / (m * n).max(1);
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

    /// `host`'s bytes (the tensor row-major, as the host keeps it) as they
    /// are in memory, [`bytes`](Self::bytes) of them, the padding zero.
    pub fn tile(&self, host: &[u8]) -> Vec<u8> {
        let (from, to) = (self.dtype.size_of(), self.stored.size_of());
        let mut out = vec![0u8; self.bytes()];
        for (k, p) in self.positions().into_iter().enumerate() {
            let src = &host[k * from..(k + 1) * from];
            let dst = &mut out[p * to..(p + 1) * to];
            match from == to {
                true => dst.copy_from_slice(src),
                // bool to int32: 0 or 1.
                false => dst[0] = u8::from(src[0] != 0),
            }
        }
        out
    }

    /// The tensor row-major, from its bytes in memory: into `host`.
    pub fn untile(&self, memory: &[u8], host: &mut [u8]) {
        let (to, from) = (self.dtype.size_of(), self.stored.size_of());
        for (k, p) in self.positions().into_iter().enumerate() {
            let src = &memory[p * from..(p + 1) * from];
            let dst = &mut host[k * to..(k + 1) * to];
            match from == to {
                true => dst.copy_from_slice(src),
                false => dst[0] = u8::from(src.iter().any(|&b| b != 0)),
            }
        }
    }

    /// Its Mosaic memref layout: its tiles (`(8,128)`, `(16,128)(2,1)`) and
    /// each dimension's stride in tiles.
    pub fn tiles(&self) -> (String, Vec<usize>) {
        if !self.tiled() {
            // Its elements in order: 32-bit ones (others, packed in words,
            // are laid out otherwise in one dimension: not viewed yet).
            return ("(128)".into(), vec![1; self.shape.len().max(1)]);
        }
        let r = self.shape.len();
        let (_, mp, np) = self.padded();
        let (tiles_m, tiles_n) = (mp / self.rows, np / 128);
        // The last dimension's 1, the next a row of tiles, each leading one
        // all the tiles of those after it.
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

    /// Its Mosaic memref layout attribute (`#tpu.tiled<...>`).
    pub fn mosaic(&self) -> String {
        let (tiles, strides) = self.tiles();
        let list: Vec<String> = strides.iter().map(usize::to_string).collect();
        format!("#tpu.tiled<{tiles},[{}]>", list.join(","))
    }
}

/// A `dtype` tensor of `shape`'s bytes on a TPU.
pub fn bytes(dtype: DType, shape: &[usize]) -> Result<usize, String> {
    Ok(Layout::new(dtype, shape)?.bytes())
}

/// Whether tensors of `a` and of `b` (one dtype, as many elements) have the
/// same bytes on a TPU: a reshape between them is a view.
pub fn same(a: &crate::graph::TensorType, b: &crate::graph::TensorType) -> bool {
    let layout =
        |t: &crate::graph::TensorType| Layout::new(t.dtype, &t.shape).map(|l| l.positions());
    a.dtype == b.dtype && matches!((layout(a), layout(b)), (Ok(x), Ok(y)) if x == y)
}
