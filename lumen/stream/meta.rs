use super::Stream;

pub struct MetaStream;

impl Stream for MetaStream {
    fn synchronize(&self) {}
}
