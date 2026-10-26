//! Collectives on CUDA through NCCL: `libnccl.so.2` loaded at run time (as PyTorch's wheels do), one
//! communicator per group of ranks and device, created on first use from an id rank 0 shares through the
//! default group's store.

use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::{Mutex, OnceLock, PoisonError};

use crate::device::cuda::CUDA;
use crate::graph::{Primitive, Reduction};
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
    get_error_string: unsafe extern "C" fn(c_int) -> *const c_char,
}

/// A `*mut c_void` communicator handle that may move between threads (NCCL's are not tied to one).
struct Handle(Comm);

unsafe impl Send for Handle {}

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
                get_unique_id: std::mem::transmute(symbol("ncclGetUniqueId")?),
                comm_init_rank: std::mem::transmute(symbol("ncclCommInitRank")?),
                all_reduce: std::mem::transmute(symbol("ncclAllReduce")?),
                all_gather: std::mem::transmute(symbol("ncclAllGather")?),
                reduce_scatter: std::mem::transmute(symbol("ncclReduceScatter")?),
                get_error_string: std::mem::transmute(symbol("ncclGetErrorString")?),
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

/// The communicator of the processes `ranks` on CUDA device `device`, created on first use: the first rank
/// of `ranks` publishes a new id under a key named after the group and every member initialises from it.
fn communicator(nccl: &Nccl, ranks: &[usize], device: usize) -> Result<Comm, String> {
    static COMMS: Mutex<Option<HashMap<(Vec<usize>, usize), Handle>>> = Mutex::new(None);
    let mut comms = COMMS.lock().unwrap_or_else(PoisonError::into_inner);
    let comms = comms.get_or_insert_with(HashMap::new);
    if let Some(Handle(comm)) = comms.get(&(ranks.to_vec(), device)) {
        return Ok(*comm);
    }
    let rank = super::get_rank()?;
    let me = ranks
        .iter()
        .position(|&r| r == rank)
        .ok_or_else(|| format!("rank {rank} is not in the collective's group {ranks:?}"))?;
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

/// Runs the collective `primitive` from `x` into `out` (contiguous, on one CUDA device) on lumen's stream.
pub(crate) fn run(primitive: &Primitive, x: &Tensor, out: &Tensor) -> Result<(), String> {
    let Device::Cuda(device) = out.device() else {
        return Err(format!(
            "{}: NCCL collectives run on CUDA, got {}",
            primitive.name(),
            out.device()
        ));
    };
    let nccl = nccl()?;
    let dtype = data_type(x.dtype())?;
    let (send, recv) = (x.data_ptr().cast_const().cast(), out.data_ptr().cast());
    // lumen's kernels run on the legacy default stream, so collectives there stay in program order.
    let stream = std::ptr::null_mut();
    match primitive {
        Primitive::AllReduce { op, ranks } => {
            let comm = communicator(nccl, ranks, device)?;
            checked(nccl, "ncclAllReduce", unsafe {
                (nccl.all_reduce)(send, recv, x.numel(), dtype, red_op(*op), comm, stream)
            })
        }
        Primitive::AllGather { ranks } => {
            let comm = communicator(nccl, ranks, device)?;
            checked(nccl, "ncclAllGather", unsafe {
                (nccl.all_gather)(send, recv, x.numel(), dtype, comm, stream)
            })
        }
        Primitive::ReduceScatter { op, ranks } => {
            let comm = communicator(nccl, ranks, device)?;
            checked(nccl, "ncclReduceScatter", unsafe {
                (nccl.reduce_scatter)(send, recv, out.numel(), dtype, red_op(*op), comm, stream)
            })
        }
        p => Err(format!("{}: not an NCCL collective", p.name())),
    }
}
