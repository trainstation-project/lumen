//! Minimal protobuf wire-format writer used to serialize Core ML model specs.
#[derive(Default)]
/// A protobuf message under construction, as its encoded bytes.
pub(crate) struct Message(Vec<u8>);

impl Message {
    /// Starts an empty message.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Returns the encoded bytes.
    pub(crate) fn into_bytes(self) -> Vec<u8> {
        self.0
    }

    /// Appends `v` as a base-128 varint.
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.0.push(v as u8 | 0x80);
            v >>= 7;
        }
        self.0.push(v as u8);
    }

    /// Appends the tag for `field` with wire type `wire`.
    fn key(&mut self, field: u32, wire: u64) {
        self.varint(u64::from(field) << 3 | wire);
    }

    /// Appends a varint field; negative values are written as their 64-bit two's complement.
    pub(crate) fn int(mut self, field: u32, v: i64) -> Self {
        self.key(field, 0);
        self.varint(v as u64);
        self
    }

    /// Appends a length-delimited field holding `data`.
    pub(crate) fn bytes(mut self, field: u32, data: &[u8]) -> Self {
        self.key(field, 2);
        self.varint(data.len() as u64);
        self.0.extend_from_slice(data);
        self
    }

    /// Appends a UTF-8 string field.
    pub(crate) fn string(self, field: u32, s: &str) -> Self {
        self.bytes(field, s.as_bytes())
    }

    /// Appends a nested message field.
    pub(crate) fn message(self, field: u32, m: Message) -> Self {
        self.bytes(field, &m.0)
    }

    /// Appends a packed repeated varint field.
    pub(crate) fn packed_ints(self, field: u32, values: impl IntoIterator<Item = i64>) -> Self {
        let mut packed = Message::new();
        for v in values {
            packed.varint(v as u64);
        }
        self.bytes(field, &packed.0)
    }

    /// Appends a packed repeated `float` field, little-endian.
    pub(crate) fn packed_floats(self, field: u32, values: impl IntoIterator<Item = f32>) -> Self {
        let data: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
        self.bytes(field, &data)
    }

    /// Appends one map entry (`key` as field 1, `value` as field 2) to map field `field`.
    pub(crate) fn entry(self, field: u32, key: &str, value: Message) -> Self {
        self.message(field, Message::new().string(1, key).message(2, value))
    }
}
