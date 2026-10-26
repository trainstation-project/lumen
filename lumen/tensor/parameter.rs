//! Parameters: whole meta tensors that get a real tensor per device when a compiled function first
//! uses them, optionally packed with other parameters into one shared block.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

use super::tensor::Tensor;
use crate::graph::{Primitive, TensorType};
use crate::{Device, TensorOptions};

/// Per-storage parameter state: the device tensors placed for it and a write version.
#[derive(Debug, Default)]
pub(crate) struct Parameter {
    /// One placement per device the parameter has been materialized on.
    placed: Mutex<Vec<Placement>>,
    /// Bumped on every write, so compiled functions can tell their copies are stale.
    version: AtomicU64,
}

/// A parameter's tensor on one device.
#[derive(Debug, Clone)]
struct Placement {
    device: Device,
    tensor: Tensor,
    /// The packed block the tensor is a view of, when packed with other parameters.
    block: Option<Tensor>,
}

impl Tensor {
    /// True for a whole contiguous meta tensor, which is what can be a parameter.
    pub fn is_parameter(&self) -> bool {
        self.device() == Device::Meta(0)
            && self.storage_offset() == 0
            && self.is_contiguous()
            && self.nbytes() == self.storage().nbytes()
    }

    /// The parameter's tensor on `device`, creating a zero-filled one on first use.
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

    /// Places `members` on `device` as views of one block concatenated along `dimension`, returning the block.
    /// `regions` splits the members into consecutive groups whose rows interleave along `dimension`;
    /// each group's members share a size. `None` when the members are not distinct parameters or
    /// are already placed some other way.
    pub fn pack(
        members: &[Tensor],
        dimension: usize,
        regions: &[usize],
        device: Device,
    ) -> Result<Option<Tensor>, String> {
        let sizes: Vec<usize> = members.iter().map(|m| m.shape()[dimension]).collect();
        check_regions(&sizes, regions)?;
        let mut ids: Vec<usize> = members.iter().map(Tensor::storage_id).collect();
        ids.sort_unstable();
        ids.dedup();
        if ids.len() != members.len() || !members.iter().all(Tensor::is_parameter) {
            return Ok(None);
        }
        let placed: Vec<Option<Placement>> = members.iter().map(|m| m.placement(device)).collect();
        // Already placed: reuse the block only if every member is placed as exactly the view packing would make.
        if placed.iter().any(Option::is_some) {
            let Some(Some(Placement {
                block: Some(block), ..
            })) = placed.first()
            else {
                return Ok(None);
            };
            if sizes.iter().sum::<usize>() != block.shape()[dimension] {
                return Ok(None);
            }
            let expected = views(block, dimension, regions, &sizes);
            let same = placed.iter().zip(&expected).all(|(p, e)| {
                p.as_ref().is_some_and(|p| {
                    p.tensor.shares_storage_with(block)
                        && p.tensor.storage_offset() == e.storage_offset()
                        && p.tensor.shape() == e.shape()
                        && p.tensor.strides() == e.strides()
                })
            });
            return Ok(same.then(|| block.clone()));
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
        for (m, view) in members
            .iter()
            .zip(views(&block, dimension, regions, &sizes))
        {
            m.place(Placement {
                device,
                tensor: view,
                block: Some(block.clone()),
            });
        }
        Ok(Some(block))
    }

    /// Copies `src` into every device tensor of this parameter and bumps its version.
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

    /// The parameter's write version.
    pub fn version(&self) -> u64 {
        self.storage().parameter.version.load(Ordering::Acquire)
    }

    /// Bumps the write version after the parameter's data changes.
    pub fn mark_written(&self) {
        self.storage()
            .parameter
            .version
            .fetch_add(1, Ordering::AcqRel);
    }

    /// This parameter's placement on `device`, if any.
    fn placement(&self, device: Device) -> Option<Placement> {
        let placed = self.storage().parameter.placed.lock();
        let placed = placed.unwrap_or_else(PoisonError::into_inner);
        placed.iter().find(|p| p.device == device).cloned()
    }

    /// Records a new placement.
    fn place(&self, placement: Placement) {
        let placed = self.storage().parameter.placed.lock();
        placed
            .unwrap_or_else(PoisonError::into_inner)
            .push(placement);
    }
}

/// Checks `regions` partitions the members into nonempty groups whose members all have the same size.
fn check_regions(sizes: &[usize], regions: &[usize]) -> Result<(), String> {
    if regions.contains(&0) || regions.iter().sum::<usize>() != sizes.len() {
        return Err(format!(
            "regions {regions:?} do not partition {} members",
            sizes.len()
        ));
    }
    let mut start = 0;
    for &k in regions {
        if sizes[start..start + k].iter().any(|&n| n != sizes[start]) {
            return Err(format!(
                "interleaved by {k}: members of one size, got {:?}",
                &sizes[start..start + k]
            ));
        }
        start += k;
    }
    Ok(())
}

/// Views of `block` for each member: in a group of `k`, member `j` takes every `k`th slice from the group's start plus `j`.
fn views(block: &Tensor, dimension: usize, regions: &[usize], sizes: &[usize]) -> Vec<Tensor> {
    let (mut out, mut member, mut start) = (Vec::new(), 0, 0);
    for &k in regions {
        let n = sizes[member];
        for j in 0..k {
            out.push(block.stepped(dimension, start + j, n, k));
        }
        member += k;
        start += k * n;
    }
    out
}
