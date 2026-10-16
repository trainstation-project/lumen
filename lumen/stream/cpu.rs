use super::Stream;

pub struct CpuStream;

impl Stream for CpuStream {
    fn synchronize(&self) {}
}
