//! Collectives on CUDA through NCCL: `libnccl.so.2` loaded at run time (as PyTorch's wheels do), one
//! communicator per group of ranks and device, created on first use from an id rank 0 shares through the
//! default group's store. Collectives run on a per-device NCCL stream after the work queued before them on
//! lumen's stream; `device_wait` makes lumen's stream wait for them. NCCL implements `Communicator` with its
//! native collectives, and its sends and receives in an `ncclGroupStart`/`ncclGroupEnd` group for the rest.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::{Mutex, OnceLock, PoisonError};

use super::collective::{self, Communicator, position};
use crate::device::cuda::CUDA;
use crate::graph::{Primitive, Reduction, TensorType};
use crate::stream::cuda::{EVENT_DISABLE_TIMING, ffi};
use crate::{DType, Device, Tensor};

/// An opaque `ncclComm_t`.
type Comm = *mut c_void;

/// `ncclUniqueId`: the 128 bytes rank 0 of a group creates and every member initialises its communicator from.
#[repr(C)]
#[derive(Clone, Copy)]
struct UniqueId([u8; 128]);

/// `dlopen` flags: resolve every symbol now, and share them with libraries loaded later.
const RTLD_NOW: c_int = 2;
const RTLD_GLOBAL: c_int = 0x100;

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

/// The NCCL entry points lumen calls, resolved from the loaded library.
struct Nccl {
    get_unique_id: unsafe extern "C" fn(*mut UniqueId) -> c_int,
    comm_init_rank: unsafe extern "C" fn(*mut Comm, c_int, UniqueId, c_int) -> c_int,
    all_reduce: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        c_int,
        c_int,
        Comm,
        *mut c_void,
    ) -> c_int,
    all_gather:
        unsafe extern "C" fn(*const c_void, *mut c_void, usize, c_int, Comm, *mut c_void) -> c_int,
    reduce_scatter: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        c_int,
        c_int,
        Comm,
        *mut c_void,
    ) -> c_int,
    broadcast: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        c_int,
        c_int,
        Comm,
        *mut c_void,
    ) -> c_int,
    reduce: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        c_int,
        c_int,
        c_int,
        Comm,
        *mut c_void,
    ) -> c_int,
    send: unsafe extern "C" fn(*const c_void, usize, c_int, c_int, Comm, *mut c_void) -> c_int,
    recv: unsafe extern "C" fn(*mut c_void, usize, c_int, c_int, Comm, *mut c_void) -> c_int,
    group_start: unsafe extern "C" fn() -> c_int,
    group_end: unsafe extern "C" fn() -> c_int,
    get_error_string: unsafe extern "C" fn(c_int) -> *const c_char,
}

/// A `*mut c_void` communicator, stream or event handle that may move between threads (none is tied to one).
struct Handle(Comm);

unsafe impl Send for Handle {}

/// The function at address `f`, typed `F`.
///
/// # Safety
///
/// `F` is a function pointer type matching the C signature of the function at `f`.
unsafe fn function<F: Copy>(f: *mut c_void) -> F {
    unsafe { std::mem::transmute_copy::<*mut c_void, F>(&f) }
}

/// Loads `libnccl.so.2` (or `LUMEN_NCCL_LIBRARY`) once and resolves the entry points.
fn nccl() -> Result<&'static Nccl, String> {
    static NCCL: OnceLock<Result<Nccl, String>> = OnceLock::new();
    NCCL.get_or_init(|| {
        let path = std::env::var("LUMEN_NCCL_LIBRARY").unwrap_or_else(|_| "libnccl.so.2".into());
        let name = CString::new(path.clone()).map_err(|e| e.to_string())?;
        let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if handle.is_null() {
            let reason = unsafe { CStr::from_ptr(dlerror()) }.to_string_lossy();
            return Err(format!(
                "loading NCCL ({path}) failed: {reason}; install nvidia-nccl-cu12 or -cu13, or set LUMEN_NCCL_LIBRARY"
            ));
        }
        let symbol = |name: &str| -> Result<*mut c_void, String> {
            let c = CString::new(name).expect("a symbol name has no NUL");
            let f = unsafe { dlsym(handle, c.as_ptr()) };
            match f.is_null() {
                true => Err(format!("{path} has no {name}")),
                false => Ok(f),
            }
        };
        // Each pointer is the named NCCL function, whose C signature the field's type spells out.
        unsafe {
            Ok(Nccl {
                get_unique_id: function(symbol("ncclGetUniqueId")?),
                comm_init_rank: function(symbol("ncclCommInitRank")?),
                all_reduce: function(symbol("ncclAllReduce")?),
                all_gather: function(symbol("ncclAllGather")?),
                reduce_scatter: function(symbol("ncclReduceScatter")?),
                broadcast: function(symbol("ncclBroadcast")?),
                reduce: function(symbol("ncclReduce")?),
                send: function(symbol("ncclSend")?),
                recv: function(symbol("ncclRecv")?),
                group_start: function(symbol("ncclGroupStart")?),
                group_end: function(symbol("ncclGroupEnd")?),
                get_error_string: function(symbol("ncclGetErrorString")?),
            })
        }
    })
    .as_ref()
    .map_err(Clone::clone)
}

/// Turns an `ncclResult_t` into `Err` with NCCL's message when it isn't `ncclSuccess`.
fn checked(nccl: &Nccl, call: &str, result: c_int) -> Result<(), String> {
    match result {
        0 => Ok(()),
        e => {
            let message = unsafe { CStr::from_ptr((nccl.get_error_string)(e)) }.to_string_lossy();
            Err(format!("{call} failed: {message} (error {e})"))
        }
    }
}

/// Communicators by their processes' ranks and CUDA device.
type Comms = HashMap<(Vec<usize>, usize), Handle>;

/// The communicator of the processes `ranks` on CUDA device `device`, created on first use: the first rank
/// of `ranks` publishes a new id under a key named after the group and every member initialises from it.
fn communicator(nccl: &Nccl, ranks: &[usize], device: usize) -> Result<Comm, String> {
    static COMMS: Mutex<Option<Comms>> = Mutex::new(None);
    let mut comms = COMMS.lock().unwrap_or_else(PoisonError::into_inner);
    let comms = comms.get_or_insert_with(HashMap::new);
    if let Some(Handle(comm)) = comms.get(&(ranks.to_vec(), device)) {
        return Ok(*comm);
    }
    let me = position(ranks, super::get_rank()?, "rank")?;
    let store = super::default_store().ok_or(
        "a collective needs the default process group: call lumen.distributed.init_process_group",
    )?;
    let names: Vec<String> = ranks.iter().map(usize::to_string).collect();
    let key = format!("lumen/nccl/{}", names.join(","));
    let mut id = UniqueId([0; 128]);
    if me == 0 {
        checked(nccl, "ncclGetUniqueId", unsafe {
            (nccl.get_unique_id)(&mut id)
        })?;
        store.set(&key, &id.0)?;
    } else {
        store.wait(&[key.as_str()], super::DEFAULT_TIMEOUT)?;
        let bytes = store.get(&key)?;
        id.0 = bytes
            .try_into()
            .map_err(|b: Vec<u8>| format!("{key} holds {} bytes, not an NCCL id's 128", b.len()))?;
    }
    CUDA.set_device(device);
    let mut comm: Comm = std::ptr::null_mut();
    let size = ranks.len() as c_int;
    checked(nccl, "ncclCommInitRank", unsafe {
        (nccl.comm_init_rank)(&mut comm, size, id, me as c_int)
    })?;
    comms.insert((ranks.to_vec(), device), Handle(comm));
    Ok(comm)
}

/// NCCL's `ncclDataType_t` for `dtype`.
fn data_type(dtype: DType) -> Result<c_int, String> {
    Ok(match dtype {
        DType::I8 => 0,
        DType::U8 => 1,
        DType::I32 => 2,
        DType::U32 => 3,
        DType::I64 => 4,
        DType::U64 => 5,
        DType::F16 => 6,
        DType::F32 => 7,
        DType::F64 => 8,
        DType::BF16 => 9,
        d => return Err(format!("NCCL has no {d} collectives")),
    })
}

/// NCCL's `ncclRedOp_t` for `op`.
fn red_op(op: Reduction) -> c_int {
    match op {
        Reduction::Sum => 0,
        Reduction::Max => 2,
        Reduction::Min => 3,
    }
}

/// `cudaStreamNonBlocking`: the stream doesn't implicitly synchronize with the legacy default stream.
const STREAM_NON_BLOCKING: u32 = 1;

/// Turns a `cudaError_t` into `Err` when it isn't `cudaSuccess`.
fn cuda_checked(call: &str, device: usize, result: i32) -> Result<(), String> {
    match result {
        0 => Ok(()),
        e => Err(format!("{call} on cuda:{device} failed (error {e})")),
    }
}

/// The non-blocking stream collectives on CUDA device `device` run on, created on first use.
fn stream(device: usize) -> Result<*mut c_void, String> {
    static STREAMS: Mutex<Option<HashMap<usize, Handle>>> = Mutex::new(None);
    let mut streams = STREAMS.lock().unwrap_or_else(PoisonError::into_inner);
    let streams = streams.get_or_insert_with(HashMap::new);
    if let Some(Handle(stream)) = streams.get(&device) {
        return Ok(*stream);
    }
    let mut stream = std::ptr::null_mut();
    cuda_checked("cudaStreamCreateWithFlags", device, unsafe {
        ffi::cudaStreamCreateWithFlags(&mut stream, STREAM_NON_BLOCKING)
    })?;
    streams.insert(device, Handle(stream));
    Ok(stream)
}

/// Per output buffer address, the events recorded when the collectives writing it finish (several when
/// empty outputs, such as sends' tokens, share an address).
static DONE: Mutex<Option<Events>> = Mutex::new(None);

/// Pending collectives' events per output buffer address.
type Events = HashMap<usize, Vec<Handle>>;

/// Makes lumen's stream on `x`'s device wait, without blocking the host, for the collectives writing `x`
/// (nothing when none is pending on it).
pub(crate) fn device_wait(x: &Tensor) -> Result<(), String> {
    let Device::Cuda(device) = x.device() else {
        return Ok(());
    };
    let done = DONE
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get_or_insert_with(HashMap::new)
        .remove(&(x.data_ptr() as usize))
        .unwrap_or_default();
    CUDA.set_device(device);
    for Handle(event) in done {
        // CUDA frees the destroyed event once the wait on it completes.
        let waited = unsafe { ffi::cudaStreamWaitEvent(std::ptr::null_mut(), event, 0) };
        unsafe { ffi::cudaEventDestroy(event) };
        cuda_checked("cudaStreamWaitEvent", device, waited)?;
    }
    Ok(())
}

/// One process's NCCL communicator for a group, with the stream its collectives run on.
struct Group {
    nccl: &'static Nccl,
    comm: Comm,
    stream: *mut c_void,
    /// This process's position in the group, and the group's size.
    rank: usize,
    size: usize,
    device: usize,
}

impl Communicator for Group {
    /// This process's position in the group.
    fn rank(&self) -> usize {
        self.rank
    }

    /// The number of processes in the group.
    fn size(&self) -> usize {
        self.size
    }

    /// `ncclSend`.
    unsafe fn send(
        &self,
        send: *const u8,
        count: usize,
        dtype: DType,
        peer: usize,
    ) -> Result<(), String> {
        let (nccl, dtype) = (self.nccl, data_type(dtype)?);
        checked(nccl, "ncclSend", unsafe {
            (nccl.send)(
                send.cast(),
                count,
                dtype,
                peer as c_int,
                self.comm,
                self.stream,
            )
        })
    }

    /// `ncclRecv`.
    unsafe fn recv(
        &self,
        recv: *mut u8,
        count: usize,
        dtype: DType,
        peer: usize,
    ) -> Result<(), String> {
        let (nccl, dtype) = (self.nccl, data_type(dtype)?);
        checked(nccl, "ncclRecv", unsafe {
            (nccl.recv)(
                recv.cast(),
                count,
                dtype,
                peer as c_int,
                self.comm,
                self.stream,
            )
        })
    }

    /// Runs `transfers` between `ncclGroupStart` and `ncclGroupEnd`, ending the group even when it fails.
    fn grouped(&self, transfers: &mut dyn FnMut() -> Result<(), String>) -> Result<(), String> {
        let nccl = self.nccl;
        checked(nccl, "ncclGroupStart", unsafe { (nccl.group_start)() })?;
        let posted = transfers();
        let ended = checked(nccl, "ncclGroupEnd", unsafe { (nccl.group_end)() });
        posted.and(ended)
    }

    /// `cudaMemsetAsync` on the collective stream.
    unsafe fn zero(&self, recv: *mut u8, count: usize, dtype: DType) -> Result<(), String> {
        let bytes = count * dtype.size_of();
        cuda_checked("cudaMemsetAsync", self.device, unsafe {
            ffi::cudaMemsetAsync(recv.cast(), 0, bytes, self.stream)
        })
    }

    /// `ncclAllReduce`.
    unsafe fn all_reduce(
        &self,
        send: *const u8,
        recv: *mut u8,
        count: usize,
        dtype: DType,
        op: Reduction,
    ) -> Result<(), String> {
        let (nccl, dtype) = (self.nccl, data_type(dtype)?);
        checked(nccl, "ncclAllReduce", unsafe {
            (nccl.all_reduce)(
                send.cast(),
                recv.cast(),
                count,
                dtype,
                red_op(op),
                self.comm,
                self.stream,
            )
        })
    }

    /// `ncclReduce`, then zeros on the processes other than `root`, whose `recv` NCCL leaves alone.
    unsafe fn reduce(
        &self,
        send: *const u8,
        recv: *mut u8,
        count: usize,
        dtype: DType,
        op: Reduction,
        root: usize,
    ) -> Result<(), String> {
        let (nccl, nccl_dtype) = (self.nccl, data_type(dtype)?);
        checked(nccl, "ncclReduce", unsafe {
            let op = red_op(op);
            (nccl.reduce)(
                send.cast(),
                recv.cast(),
                count,
                nccl_dtype,
                op,
                root as c_int,
                self.comm,
                self.stream,
            )
        })?;
        match self.rank == root {
            true => Ok(()),
            false => unsafe { self.zero(recv, count, dtype) },
        }
    }

    /// `ncclReduceScatter`.
    unsafe fn reduce_scatter(
        &self,
        send: *const u8,
        recv: *mut u8,
        count: usize,
        dtype: DType,
        op: Reduction,
    ) -> Result<(), String> {
        let (nccl, dtype) = (self.nccl, data_type(dtype)?);
        checked(nccl, "ncclReduceScatter", unsafe {
            (nccl.reduce_scatter)(
                send.cast(),
                recv.cast(),
                count,
                dtype,
                red_op(op),
                self.comm,
                self.stream,
            )
        })
    }

    /// `ncclAllGather`.
    unsafe fn all_gather(
        &self,
        send: *const u8,
        recv: *mut u8,
        count: usize,
        dtype: DType,
    ) -> Result<(), String> {
        let (nccl, dtype) = (self.nccl, data_type(dtype)?);
        checked(nccl, "ncclAllGather", unsafe {
            (nccl.all_gather)(
                send.cast(),
                recv.cast(),
                count,
                dtype,
                self.comm,
                self.stream,
            )
        })
    }

    /// `ncclBroadcast`.
    unsafe fn broadcast(
        &self,
        send: *const u8,
        recv: *mut u8,
        count: usize,
        dtype: DType,
        root: usize,
    ) -> Result<(), String> {
        let (nccl, dtype) = (self.nccl, data_type(dtype)?);
        checked(nccl, "ncclBroadcast", unsafe {
            (nccl.broadcast)(
                send.cast(),
                recv.cast(),
                count,
                dtype,
                root as c_int,
                self.comm,
                self.stream,
            )
        })
    }
}

/// Starts the collective `primitive` from `inputs` (its operand, if it has one) into `out` (contiguous, on one
/// CUDA device) on the device's NCCL stream once lumen's stream has produced them; `device_wait` on `out`
/// waits for it.
pub(crate) fn run(primitive: &Primitive, inputs: &[Tensor], out: &Tensor) -> Result<(), String> {
    let Device::Cuda(device) = out.device() else {
        return Err(format!(
            "{}: NCCL collectives run on CUDA, got {}",
            primitive.name(),
            out.device()
        ));
    };
    let nccl = nccl()?;
    let ranks = collective::ranks(primitive)
        .ok_or_else(|| format!("{}: not an NCCL collective", primitive.name()))?;
    let group = Group {
        nccl,
        comm: communicator(nccl, ranks, device)?,
        stream: stream(device)?,
        rank: position(ranks, super::get_rank()?, "rank")?,
        size: ranks.len(),
        device,
    };
    crate::stream::cuda::wait(device, group.stream as usize);
    let x = inputs.first();
    let x_type = x.map(|x| TensorType::new(x.dtype(), x.shape()));
    let send = x.map_or(std::ptr::null(), |x| x.data_ptr().cast_const());
    let out_type = TensorType::new(out.dtype(), out.shape());
    unsafe {
        collective::run(
            &group,
            primitive,
            send,
            out.data_ptr(),
            x_type.as_ref(),
            &out_type,
        )?
    };
    let mut event = std::ptr::null_mut();
    cuda_checked("cudaEventCreateWithFlags", device, unsafe {
        ffi::cudaEventCreateWithFlags(&mut event, EVENT_DISABLE_TIMING)
    })?;
    cuda_checked("cudaEventRecord", device, unsafe {
        ffi::cudaEventRecord(event, group.stream)
    })?;
    DONE.lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get_or_insert_with(HashMap::new)
        .entry(out.data_ptr() as usize)
        .or_default()
        .push(Handle(event));
    Ok(())
}
