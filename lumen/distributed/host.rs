//! Collectives between host (CPU) buffers through the default group's TCP store: each message is a key, set
//! by its sender and taken (read, then deleted) by its receiver, named after the two processes and how many
//! messages the pair has exchanged, so messages between a pair arrive in the order they were sent. Every
//! message goes through rank 0's store, so this is meant for tests and small CPU worlds, not throughput.
//! Collectives run synchronously, in the plan's step.
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use super::TcpStore;
use super::collective::{Communicator, position};
use crate::graph::Reduction;
use crate::tensor::dtype::dispatch_dtype;
use crate::{DType, Element, Scalar};

/// One process's view of a group of processes communicating through the default group's store.
pub(crate) struct Host {
    /// The group's global ranks.
    ranks: Vec<usize>,
    /// This process's position in `ranks`.
    me: usize,
    store: Arc<TcpStore>,
}

/// Per (sender, receiver) pair of global ranks and direction (sent, or taken), the messages this process has
/// exchanged between them; a process sending to itself counts both.
static MESSAGES: Mutex<Option<Counts>> = Mutex::new(None);

/// Message counts per (sender, receiver, sending).
type Counts = HashMap<(usize, usize, bool), u64>;

/// The store key of the next message from global rank `from` to `to` that this process sends (`sending`) or
/// takes, counting it.
fn next_key(from: usize, to: usize, sending: bool) -> String {
    let mut messages = MESSAGES.lock().unwrap_or_else(PoisonError::into_inner);
    let count = messages
        .get_or_insert_with(HashMap::new)
        .entry((from, to, sending))
        .or_insert(0);
    *count += 1;
    format!("lumen/host/{from}/{to}/{count}")
}

impl Host {
    /// This process's communicator for the group `ranks`, over the default group's store.
    pub(crate) fn new(ranks: &[usize]) -> Result<Self, String> {
        let store = super::default_store().ok_or(
            "a collective needs the default process group: call lumen.distributed.init_process_group",
        )?;
        Ok(Host {
            ranks: ranks.to_vec(),
            me: position(ranks, super::get_rank()?, "rank")?,
            store,
        })
    }

    /// Combines the `size()` blocks of `count` elements in `blocks` with `op`, in position order, into
    /// `recv`, rounding to `dtype` after each step.
    ///
    /// # Safety
    /// `recv` addresses `count` writable elements of `dtype`.
    unsafe fn fold(&self, blocks: &[u8], recv: *mut u8, count: usize, dtype: DType, op: Reduction) {
        dispatch_dtype!(dtype, T => {
            let blocks = blocks.as_ptr().cast::<T>();
            let recv = recv.cast::<T>();
            for i in 0..count {
                let mut acc = unsafe { blocks.add(i).read_unaligned() };
                for j in 1..self.size() {
                    let x = unsafe { blocks.add(j * count + i).read_unaligned() };
                    acc = T::from_scalar(combine(op, acc.to_scalar(), x.to_scalar()));
                }
                unsafe { recv.add(i).write_unaligned(acc) };
            }
        })
    }
}

/// `a` and `b` combined with `op`; booleans sum and take the maximum as an or, the minimum as an and.
fn combine(op: Reduction, a: Scalar, b: Scalar) -> Scalar {
    use Reduction::*;
    match (a, b) {
        (Scalar::Float(a), Scalar::Float(b)) => Scalar::Float(match op {
            Sum => a + b,
            Max => a.max(b),
            Min => a.min(b),
        }),
        (Scalar::Int(a), Scalar::Int(b)) => Scalar::Int(match op {
            Sum => a.wrapping_add(b),
            Max => a.max(b),
            Min => a.min(b),
        }),
        (Scalar::Bool(a), Scalar::Bool(b)) => Scalar::Bool(match op {
            Sum | Max => a | b,
            Min => a & b,
        }),
        (a, b) => unreachable!("one dtype's elements, got {a:?} and {b:?}"),
    }
}

impl Communicator for Host {
    /// This process's position in the group.
    fn rank(&self) -> usize {
        self.me
    }

    /// The number of processes in the group.
    fn size(&self) -> usize {
        self.ranks.len()
    }

    /// Sets the message's key to the `count` elements at `send`; never waits for the receiver.
    unsafe fn send(
        &self,
        send: *const u8,
        count: usize,
        dtype: DType,
        peer: usize,
    ) -> Result<(), String> {
        // an empty message may come from a dangling pointer, which a slice must not hold
        let bytes = match count {
            0 => &[][..],
            _ => unsafe { std::slice::from_raw_parts(send, count * dtype.size_of()) },
        };
        let key = next_key(self.ranks[self.me], self.ranks[peer], true);
        self.store.set(&key, bytes)
    }

    /// Waits for the message's key, copies it into `recv` and deletes it.
    unsafe fn recv(
        &self,
        recv: *mut u8,
        count: usize,
        dtype: DType,
        peer: usize,
    ) -> Result<(), String> {
        let key = next_key(self.ranks[peer], self.ranks[self.me], false);
        let bytes = self.store.get(&key)?;
        self.store.delete_key(&key)?;
        if bytes.len() != count * dtype.size_of() {
            return Err(format!(
                "received {} bytes from rank {}, expected {count} {dtype} elements",
                bytes.len(),
                self.ranks[peer]
            ));
        }
        if !bytes.is_empty() {
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), recv, bytes.len()) };
        }
        Ok(())
    }

    /// Writes zero bytes, which are zeros of every dtype.
    unsafe fn zero(&self, recv: *mut u8, count: usize, dtype: DType) -> Result<(), String> {
        if count > 0 {
            unsafe { std::ptr::write_bytes(recv, 0, count * dtype.size_of()) };
        }
        Ok(())
    }

    /// Gathers every process's elements, then folds them in position order.
    unsafe fn all_reduce(
        &self,
        send: *const u8,
        recv: *mut u8,
        count: usize,
        dtype: DType,
        op: Reduction,
    ) -> Result<(), String> {
        let mut blocks = vec![0u8; self.size() * count * dtype.size_of()];
        unsafe {
            self.all_gather(send, blocks.as_mut_ptr(), count, dtype)?;
            self.fold(&blocks, recv, count, dtype, op);
        }
        Ok(())
    }

    /// Gathers every process's elements to `root`, which folds them in position order.
    unsafe fn reduce(
        &self,
        send: *const u8,
        recv: *mut u8,
        count: usize,
        dtype: DType,
        op: Reduction,
        root: usize,
    ) -> Result<(), String> {
        let mut blocks = vec![0u8; self.size() * count * dtype.size_of()];
        unsafe {
            self.gather(send, blocks.as_mut_ptr(), count, dtype, root)?;
            match self.me == root {
                true => self.fold(&blocks, recv, count, dtype, op),
                false => self.zero(recv, count, dtype)?,
            }
        }
        Ok(())
    }

    /// Exchanges blocks all-to-all, then folds the ones received in position order.
    unsafe fn reduce_scatter(
        &self,
        send: *const u8,
        recv: *mut u8,
        count: usize,
        dtype: DType,
        op: Reduction,
    ) -> Result<(), String> {
        let mut blocks = vec![0u8; self.size() * count * dtype.size_of()];
        unsafe {
            self.all_to_all(send, blocks.as_mut_ptr(), count, dtype)?;
            self.fold(&blocks, recv, count, dtype, op);
        }
        Ok(())
    }
}
