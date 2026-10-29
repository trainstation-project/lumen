//! The collectives and point-to-point transfers a plan runs, defined once over a backend's `Communicator`:
//! NCCL on CUDA (`nccl.rs`), the default group's TCP store on the CPU (`host.rs`). A backend provides
//! sends, receives and the reducing collectives; the moving ones (gathers, scatters, broadcasts,
//! all-to-alls) default to grouped sends and receives, which a backend with native ones overrides.
use crate::DType;
use crate::graph::{Primitive, Reduction, TensorType};

/// One process's view of a group of processes that exchange buffers of its device's memory. Positions
/// (`peer`, `root`) index the group's ranks; counts are elements of `dtype`.
///
/// Every method is unsafe: its pointers must address that many elements of `dtype` in the backend's
/// memory, readable (`send`) or writable (`recv`), until the transfer has finished.
pub(crate) trait Communicator {
    /// This process's position in the group.
    fn rank(&self) -> usize;

    /// The number of processes in the group.
    fn size(&self) -> usize;

    /// Sends `count` elements at `send` to the process at position `peer`.
    ///
    /// # Safety
    /// `send` addresses `count` readable elements of `dtype`.
    unsafe fn send(
        &self,
        send: *const u8,
        count: usize,
        dtype: DType,
        peer: usize,
    ) -> Result<(), String>;

    /// Receives `count` elements from the process at position `peer` into `recv`.
    ///
    /// # Safety
    /// `recv` addresses `count` writable elements of `dtype`.
    unsafe fn recv(
        &self,
        recv: *mut u8,
        count: usize,
        dtype: DType,
        peer: usize,
    ) -> Result<(), String>;

    /// Runs the sends and receives `transfers` posts as one group, none waiting for another of the group.
    fn grouped(&self, transfers: &mut dyn FnMut() -> Result<(), String>) -> Result<(), String> {
        transfers()
    }

    /// Fills `recv` with `count` zeros.
    ///
    /// # Safety
    /// `recv` addresses `count` writable elements of `dtype`.
    unsafe fn zero(&self, recv: *mut u8, count: usize, dtype: DType) -> Result<(), String>;

    /// Combines every process's `count` elements at `send` with `op` into each one's `recv`.
    ///
    /// # Safety
    /// `send` and `recv` address `count` elements of `dtype`.
    unsafe fn all_reduce(
        &self,
        send: *const u8,
        recv: *mut u8,
        count: usize,
        dtype: DType,
        op: Reduction,
    ) -> Result<(), String>;

    /// Combines every process's `count` elements at `send` with `op` into `recv` of the process at position
    /// `root`; the others' `recv` get zeros.
    ///
    /// # Safety
    /// `send` and `recv` address `count` elements of `dtype`.
    unsafe fn reduce(
        &self,
        send: *const u8,
        recv: *mut u8,
        count: usize,
        dtype: DType,
        op: Reduction,
        root: usize,
    ) -> Result<(), String>;

    /// Combines every process's `size() * count` elements at `send` with `op`; the process at position `r`
    /// gets block `r` (`count` elements) of the result in `recv`.
    ///
    /// # Safety
    /// `send` addresses `size() * count` elements of `dtype`, `recv` `count`.
    unsafe fn reduce_scatter(
        &self,
        send: *const u8,
        recv: *mut u8,
        count: usize,
        dtype: DType,
        op: Reduction,
    ) -> Result<(), String>;

    /// Places the `count` elements at `send` of the process at position `j` as block `j` of every process's
    /// `recv`.
    ///
    /// # Safety
    /// `send` addresses `count` elements of `dtype`, `recv` `size() * count`.
    unsafe fn all_gather(
        &self,
        send: *const u8,
        recv: *mut u8,
        count: usize,
        dtype: DType,
    ) -> Result<(), String> {
        let bytes = count * dtype.size_of();
        self.grouped(&mut || {
            for j in 0..self.size() {
                unsafe { self.send(send, count, dtype, j)? };
            }
            for j in 0..self.size() {
                unsafe { self.recv(recv.add(j * bytes), count, dtype, j)? };
            }
            Ok(())
        })
    }

    /// Copies the `count` elements at `send` of the process at position `root` to every process's `recv`.
    ///
    /// # Safety
    /// `send` and `recv` address `count` elements of `dtype`.
    unsafe fn broadcast(
        &self,
        send: *const u8,
        recv: *mut u8,
        count: usize,
        dtype: DType,
        root: usize,
    ) -> Result<(), String> {
        self.grouped(&mut || {
            if self.rank() == root {
                for j in 0..self.size() {
                    unsafe { self.send(send, count, dtype, j)? };
                }
            }
            unsafe { self.recv(recv, count, dtype, root) }
        })
    }

    /// Sends block `j` (`count` elements) of `send` to the process at position `j`, which receives it as
    /// block `rank()` of its `recv`.
    ///
    /// # Safety
    /// `send` and `recv` address `size() * count` elements of `dtype`.
    unsafe fn all_to_all(
        &self,
        send: *const u8,
        recv: *mut u8,
        count: usize,
        dtype: DType,
    ) -> Result<(), String> {
        let bytes = count * dtype.size_of();
        self.grouped(&mut || {
            for j in 0..self.size() {
                unsafe { self.send(send.add(j * bytes), count, dtype, j)? };
            }
            for j in 0..self.size() {
                unsafe { self.recv(recv.add(j * bytes), count, dtype, j)? };
            }
            Ok(())
        })
    }

    /// Places the `count` elements at `send` of the process at position `j` as block `j` of the `recv` of
    /// the process at position `root`; the others' `recv` get zeros.
    ///
    /// # Safety
    /// `send` addresses `count` elements of `dtype`, `recv` `size() * count`.
    unsafe fn gather(
        &self,
        send: *const u8,
        recv: *mut u8,
        count: usize,
        dtype: DType,
        root: usize,
    ) -> Result<(), String> {
        let bytes = count * dtype.size_of();
        if self.rank() != root {
            unsafe { self.zero(recv, self.size() * count, dtype)? };
        }
        self.grouped(&mut || {
            unsafe { self.send(send, count, dtype, root)? };
            if self.rank() == root {
                for j in 0..self.size() {
                    unsafe { self.recv(recv.add(j * bytes), count, dtype, j)? };
                }
            }
            Ok(())
        })
    }

    /// Gives the process at position `j` block `j` (`count` elements) of the `send` of the process at
    /// position `root`, in its `recv`.
    ///
    /// # Safety
    /// `send` addresses `size() * count` elements of `dtype` (read on `root` only), `recv` `count`.
    unsafe fn scatter(
        &self,
        send: *const u8,
        recv: *mut u8,
        count: usize,
        dtype: DType,
        root: usize,
    ) -> Result<(), String> {
        let bytes = count * dtype.size_of();
        self.grouped(&mut || {
            if self.rank() == root {
                for j in 0..self.size() {
                    unsafe { self.send(send.add(j * bytes), count, dtype, j)? };
                }
            }
            unsafe { self.recv(recv, count, dtype, root) }
        })
    }
}

/// The position of process `rank` in the group `ranks`, as `what` names it in errors.
pub(crate) fn position(ranks: &[usize], rank: usize, what: &str) -> Result<usize, String> {
    ranks
        .iter()
        .position(|&r| r == rank)
        .ok_or_else(|| format!("{what} {rank} is not in the collective's group {ranks:?}"))
}

/// The group of processes the collective `primitive` spans, or `None` if it is not a collective.
pub(crate) fn ranks(primitive: &Primitive) -> Option<&[usize]> {
    use Primitive::*;
    match primitive {
        AllReduce { ranks, .. }
        | AllGather { ranks }
        | ReduceScatter { ranks, .. }
        | AllToAll { ranks }
        | Broadcast { ranks, .. }
        | Reduce { ranks, .. }
        | GatherTo { ranks, .. }
        | ScatterFrom { ranks, .. }
        | Send { ranks, .. }
        | Recv { ranks, .. } => Some(ranks),
        _ => None,
    }
}

/// Runs the collective `primitive` through `comm` (its group's communicator): from `send`, the operand (of
/// type `x`; null for a `recv`), into `recv`, the result (of type `out`; unwritten for a `send`).
///
/// # Safety
/// `send` addresses an `x` and `recv` an `out` in `comm`'s memory, until the collective has finished.
pub(crate) unsafe fn run(
    comm: &dyn Communicator,
    primitive: &Primitive,
    send: *const u8,
    recv: *mut u8,
    x: Option<&TensorType>,
    out: &TensorType,
) -> Result<(), String> {
    use Primitive::*;
    let ranks =
        ranks(primitive).ok_or_else(|| format!("{}: not a collective", primitive.name()))?;
    let at = |rank: usize, what| position(ranks, rank, what);
    let dtype = out.dtype;
    // Elements one process contributes: the operand's, or one block of it for a scatter.
    let count = x.map_or(0, TensorType::numel);
    let block = out.numel() / comm.size().max(1);
    unsafe {
        match primitive {
            AllReduce { op, .. } => comm.all_reduce(send, recv, count, dtype, *op),
            AllGather { .. } => comm.all_gather(send, recv, count, dtype),
            ReduceScatter { op, .. } => comm.reduce_scatter(send, recv, out.numel(), dtype, *op),
            AllToAll { .. } => comm.all_to_all(send, recv, block, dtype),
            Broadcast { root, .. } => comm.broadcast(send, recv, count, dtype, at(*root, "root")?),
            Reduce { op, root, .. } => {
                comm.reduce(send, recv, count, dtype, *op, at(*root, "root")?)
            }
            GatherTo { root, .. } => comm.gather(send, recv, count, dtype, at(*root, "root")?),
            ScatterFrom { root, .. } => {
                comm.scatter(send, recv, out.numel(), dtype, at(*root, "root")?)
            }
            Send { peer, .. } => {
                let x = x.expect("a send's operand");
                comm.send(send, x.numel(), x.dtype, at(*peer, "peer")?)
            }
            Recv { peer, .. } => comm.recv(recv, out.numel(), dtype, at(*peer, "peer")?),
            p => Err(format!("{}: not a collective", p.name())),
        }
    }
}
