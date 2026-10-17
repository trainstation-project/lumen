#[derive(Default)]
pub(crate) struct Message(Vec<u8>);

impl Message {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.0
    }

    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.0.push(v as u8 | 0x80);
            v >>= 7;
        }
        self.0.push(v as u8);
    }

    fn key(&mut self, field: u32, wire: u64) {
        self.varint(u64::from(field) << 3 | wire);
    }

    pub(crate) fn int(mut self, field: u32, v: i64) -> Self {
        self.key(field, 0);
        self.varint(v as u64);
        self
    }

    pub(crate) fn bytes(mut self, field: u32, data: &[u8]) -> Self {
        self.key(field, 2);
        self.varint(data.len() as u64);
        self.0.extend_from_slice(data);
        self
    }

    pub(crate) fn string(self, field: u32, s: &str) -> Self {
        self.bytes(field, s.as_bytes())
    }

    pub(crate) fn message(self, field: u32, m: Message) -> Self {
        self.bytes(field, &m.0)
    }

    pub(crate) fn packed_ints(self, field: u32, values: impl IntoIterator<Item = i64>) -> Self {
        let mut packed = Message::new();
        for v in values {
            packed.varint(v as u64);
        }
        self.bytes(field, &packed.0)
    }

    pub(crate) fn packed_floats(self, field: u32, values: impl IntoIterator<Item = f32>) -> Self {
        let data: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
        self.bytes(field, &data)
    }

    pub(crate) fn entry(self, field: u32, key: &str, value: Message) -> Self {
        self.message(field, Message::new().string(1, key).message(2, value))
    }
}
