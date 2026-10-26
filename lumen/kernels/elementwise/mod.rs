//! Elementwise kernel helpers, currently only the TPU block-size picker.

#[cfg(lumen_tpu_linked)]
pub(crate) mod tpu;
