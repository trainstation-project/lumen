use super::Stream;

pub struct TpuStream {
    pub index: usize,
}

impl Stream for TpuStream {
    fn synchronize(&self) {
        crate::allocator::tpu::wait(self.index).unwrap_or_else(|e| panic!("{e}"))
    }
}
