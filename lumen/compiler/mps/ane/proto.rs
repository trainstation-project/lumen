//! The protobuf wire format, written by hand for the few Core ML messages
//! [`super`] builds (`Model.proto`, `MIL.proto`, `FeatureTypes.proto` in
//! coremltools' `mlmodel/format`): each message a [`Message`], its fields
//! appended by number, nested messages length-delimited.

/// A message being encoded: its fields' bytes, in the order appended.
#[derive(Default)]
pub(super) struct Message(Vec<u8>);

impl Message {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn into_bytes(self) -> Vec<u8> {
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

    /// An integer field (int32, int64, uint64, bool, enum): a varint, a
    /// negative one as its two's complement (ten bytes).
    pub(super) fn int(mut self, field: u32, v: i64) -> Self {
        self.key(field, 0);
        self.varint(v as u64);
        self
    }

    pub(super) fn bytes(mut self, field: u32, data: &[u8]) -> Self {
        self.key(field, 2);
        self.varint(data.len() as u64);
        self.0.extend_from_slice(data);
        self
    }

    pub(super) fn string(self, field: u32, s: &str) -> Self {
        self.bytes(field, s.as_bytes())
    }

    pub(super) fn message(self, field: u32, m: Message) -> Self {
        self.bytes(field, &m.0)
    }

    /// A packed repeated integer field (int32, int64, bool).
    pub(super) fn packed_ints(self, field: u32, values: impl IntoIterator<Item = i64>) -> Self {
        let mut packed = Message::new();
        for v in values {
            packed.varint(v as u64);
        }
        self.bytes(field, &packed.0)
    }

    /// A packed repeated float field.
    pub(super) fn packed_floats(self, field: u32, values: impl IntoIterator<Item = f32>) -> Self {
        let data: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
        self.bytes(field, &data)
    }

    /// A map entry (`map<string, V>`: a repeated message of key 1, value 2).
    pub(super) fn entry(self, field: u32, key: &str, value: Message) -> Self {
        self.message(field, Message::new().string(1, key).message(2, value))
    }
}
