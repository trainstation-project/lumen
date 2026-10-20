use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

use super::tensor::Tensor;
use crate::graph::{Primitive, TensorType};
use crate::{Device, TensorOptions};

#[derive(Debug, Default)]
pub(crate) struct Parameter {
    placed: Mutex<Vec<Placement>>,
    version: AtomicU64,
}

#[derive(Debug, Clone)]
struct Placement {
    device: Device,
    tensor: Tensor,
    block: Option<Tensor>,
}

impl Tensor {
    pub fn is_parameter(&self) -> bool {
        self.device() == Device::Meta(0)
            && self.storage_offset() == 0
            && self.is_contiguous()
            && self.nbytes() == self.storage().nbytes()
    }

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

    /// `members` (parameters) placed in one block on `device`, side by side
    /// along `dimension` (`interleave` 1), or interleaved along it by
    /// `interleave` (members of one size: element `j` of member `i` at `j *
    /// interleave + i`, `merge_dots`' interleaved block): each member a view
    /// of the block. The block they are placed in already, if so (`None` if
    /// they are placed otherwise).
    pub fn pack(
        members: &[Tensor],
        dimension: usize,
        interleave: usize,
        device: Device,
    ) -> Result<Option<Tensor>, String> {
        if interleave > 1
            && (members.len() != interleave
                || members
                    .iter()
                    .any(|m| m.shape()[dimension] != members[0].shape()[dimension]))
        {
            return Err(format!(
                "interleaved by {interleave}: {interleave} members of one size, got {}",
                members.len()
            ));
        }
        let mut ids: Vec<usize> = members.iter().map(Tensor::storage_id).collect();
        ids.sort_unstable();
        ids.dedup();
        if ids.len() != members.len() || !members.iter().all(Tensor::is_parameter) {
            return Ok(None);
        }
        let placed: Vec<Option<Placement>> = members.iter().map(|m| m.placement(device)).collect();
        if placed.iter().any(Option::is_some) {
            let Some(Some(Placement {
                block: Some(block), ..
            })) = placed.first()
            else {
                return Ok(None);
            };
            let mut start = 0;
            for (k, (m, p)) in members.iter().zip(&placed).enumerate() {
                let n = m.shape()[dimension];
                if start + n > block.shape()[dimension] {
                    return Ok(None);
                }
                let expected = member(block, dimension, interleave, start, n, k);
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
        let ty = Primitive::Concatenate {
            dimension,
            in_place: false,
        }
        .infer(&types.iter().collect::<Vec<_>>())?;
        let options = TensorOptions::new().dtype(ty.dtype).device(device);
        let block = Tensor::zeros(&ty.shape, options);
        let mut start = 0;
        for (k, m) in members.iter().enumerate() {
            let n = m.shape()[dimension];
            m.place(Placement {
                device,
                tensor: member(&block, dimension, interleave, start, n, k),
                block: Some(block.clone()),
            });
            start += n;
        }
        Ok(Some(block))
    }

    pub(crate) fn copy_into_placements(&self, src: &Tensor) -> Result<(), String> {
        let placed = self.storage().parameter.placed.lock();
        let placed = placed.unwrap_or_else(PoisonError::into_inner).clone();
        if !self.is_parameter() || placed.is_empty() {
            return Err("copy_: a meta tensor has no memory until a compiled function using it compiles: compile it first (call it on meta tensors)".into());
        }
        for p in &placed {
            p.tensor.copy_(src)?;
        }
        self.mark_written();
        Ok(())
    }

    pub fn version(&self) -> u64 {
        self.storage().parameter.version.load(Ordering::Acquire)
    }

    /// Count a write of this parameter's value ([`version`](Self::version)).
    pub fn mark_written(&self) {
        self.storage()
            .parameter
            .version
            .fetch_add(1, Ordering::AcqRel);
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

/// Member `k` of `block`, of `n` along `dimension`: side by side, at
/// `start` (`interleave` 1); or interleaved, its elements every
/// `interleave`th from `k`.
fn member(
    block: &Tensor,
    dimension: usize,
    interleave: usize,
    start: usize,
    n: usize,
    k: usize,
) -> Tensor {
    match interleave {
        1 => block.narrow(dimension, start, n),
        _ => {
            let mut shape = block.shape().to_vec();
            shape.splice(dimension..=dimension, [n, interleave]);
            block.reshape(&shape).select(dimension + 1, k)
        }
    }
}
