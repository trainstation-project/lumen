use crate::DType;

const ROW: usize = 512;

const ALIGNED_COLUMNS: usize = 40;

#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub stored: DType,
    pub rows: usize,
    pub pack: usize,
}

impl Layout {
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

    pub fn tiled(&self) -> bool {
        self.shape.len() >= 2
    }

    fn padded(&self) -> (usize, usize, usize) {
        let r = self.shape.len();
        let lead = self.shape[..r - 2].iter().product();
        let m = self.shape[r - 2].div_ceil(self.rows) * self.rows;
        let n = self.shape[r - 1].div_ceil(128) * 128;
        (lead, m, n)
    }

    pub fn bytes(&self) -> usize {
        let bits = self.stored.bits();
        if self.tiled() {
            let (lead, m, n) = self.padded();
            return lead * m * n * bits / 8;
        }
        let numel: usize = self.shape.iter().product();
        (numel * bits).div_ceil(8 * ROW) * ROW
    }

    fn positions(&self) -> Vec<usize> {
        let numel: usize = self.shape.iter().product();
        if !self.tiled() {
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

    pub fn tile_shape(&self) -> Vec<usize> {
        match self.shape.len() {
            0 => vec![],
            1 => vec![128 * self.pack],
            _ => vec![self.rows, 128],
        }
    }

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

    pub fn mosaic(&self) -> String {
        let (tiles, strides) = self.tiles();
        let list: Vec<String> = strides.iter().map(usize::to_string).collect();
        format!("#tpu.tiled<{tiles},[{}]>", list.join(","))
    }
}

pub fn bytes(dtype: DType, shape: &[usize]) -> Result<usize, String> {
    Ok(Layout::new(dtype, shape)?.bytes())
}

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

pub fn same(a: &crate::graph::TensorType, b: &crate::graph::TensorType) -> bool {
    let layout =
        |t: &crate::graph::TensorType| Layout::new(t.dtype, &t.shape).map(|l| l.positions());
    a.dtype == b.dtype && matches!((layout(a), layout(b)), (Ok(x), Ok(y)) if x == y)
}
