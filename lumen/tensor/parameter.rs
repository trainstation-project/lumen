//! Parameters: meta tensors whose memory a compiled function places on its
//! device (`lumen.compile`), never the caller. A parameter is placed the
//! first time a compiled function uses it there, and every compiled function
//! using it reads that placement, so they share it. Parameters the compiler
//! wants side by side (dots it merges) are placed together, as one block
//! ([`Tensor::pack`]). Placed memory starts zeroed; the caller copies the
//! data in (`copy_`).
//!
//! A parameter is a whole meta tensor: contiguous, over all of its storage
//! (what the factories and safetensors give).

use std::sync::{Mutex, PoisonError};

use super::tensor::Tensor;
use crate::graph::{Primitive, TensorType};
use crate::{Device, TensorOptions};

/// A meta storage's placements (none for other storages).
#[derive(Debug, Default)]
pub(crate) struct Parameter {
    placed: Mutex<Vec<Placement>>,
}

/// A parameter's memory on a device: its own, or a view of the block it was
/// packed into.
#[derive(Debug, Clone)]
struct Placement {
    device: Device,
    tensor: Tensor,
    block: Option<Tensor>,
}

impl Tensor {
    /// Whether this can be a parameter: a meta tensor over all of its
    /// storage, contiguous.
    pub fn is_parameter(&self) -> bool {
        self.device() == Device::Meta
            && self.storage_offset() == 0
            && self.is_contiguous()
            && self.nbytes() == self.storage().nbytes()
    }

    /// The parameter's memory on `device`: placed (allocated, zeroed) on
    /// first use there, shared by every later use.
    pub fn placed(&self, device: Device) -> Result<Tensor, String> {
        if !self.is_parameter() {
            return Err("a parameter must be a whole meta tensor".into());
        }
        if let Some(p) = self.placement(device) {
            return Ok(p.tensor);
        }
        let options = TensorOptions::new().dtype(self.dtype()).device(device);
        let tensor = Tensor::zeros(self.shape(), options);
        self.place(Placement {
            device,
            tensor: tensor.clone(),
            block: None,
        });
        Ok(tensor)
    }

    /// The parameters `members` placed on `device` side by side along
    /// `dimension`, as one block (their concatenation), each a view of it:
    /// placed so if none is placed there yet, else the block they were
    /// packed into in this order. `None` if they are placed otherwise, or
    /// are not distinct parameters.
    pub fn pack(
        members: &[Tensor],
        dimension: usize,
        device: Device,
    ) -> Result<Option<Tensor>, String> {
        let mut ids: Vec<usize> = members.iter().map(Tensor::storage_id).collect();
        ids.sort_unstable();
        ids.dedup();
        if ids.len() != members.len() || !members.iter().all(Tensor::is_parameter) {
            return Ok(None);
        }
        let placed: Vec<Option<Placement>> = members.iter().map(|m| m.placement(device)).collect();
        if placed.iter().any(Option::is_some) {
            // Packed in this order: each the next part of one block.
            let Some(Some(Placement {
                block: Some(block), ..
            })) = placed.first()
            else {
                return Ok(None);
            };
            let mut start = 0;
            for (m, p) in members.iter().zip(&placed) {
                let n = m.shape()[dimension];
                if start + n > block.shape()[dimension] {
                    return Ok(None);
                }
                let expected = block.narrow(dimension, start, n);
                let same = p.as_ref().is_some_and(|p| {
                    p.tensor.shares_storage_with(block)
                        && p.tensor.storage_offset() == expected.storage_offset()
                        && p.tensor.shape() == expected.shape()
                        && p.tensor.strides() == expected.strides()
                });
                if !same {
                    return Ok(None);
                }
                start += n;
            }
            return Ok((start == block.shape()[dimension]).then(|| block.clone()));
        }
        let types: Vec<TensorType> = members
            .iter()
            .map(|m| TensorType::new(m.dtype(), m.shape()))
            .collect();
        let ty = Primitive::Concatenate { dimension }.infer(&types.iter().collect::<Vec<_>>())?;
        let options = TensorOptions::new().dtype(ty.dtype).device(device);
        let block = Tensor::zeros(&ty.shape, options);
        let mut start = 0;
        for m in members {
            let n = m.shape()[dimension];
            m.place(Placement {
                device,
                tensor: block.narrow(dimension, start, n),
                block: Some(block.clone()),
            });
            start += n;
        }
        Ok(Some(block))
    }

    fn placement(&self, device: Device) -> Option<Placement> {
        let placed = self.storage().parameter.placed.lock();
        let placed = placed.unwrap_or_else(PoisonError::into_inner);
        placed.iter().find(|p| p.device == device).cloned()
    }

    fn place(&self, placement: Placement) {
        let placed = self.storage().parameter.placed.lock();
        placed
            .unwrap_or_else(PoisonError::into_inner)
            .push(placement);
    }
}
