use std::ffi::{CString, c_char, c_void};

use crate::graph::{Primitive, TensorType};
use crate::graph_compiler::plan::Step;
use crate::stream::mps::{self, Completion};
use crate::tensor::dtype::dispatch_dtype;
use crate::tensor::storage::as_bytes;
use crate::{DType, Element, Scalar, Tensor};

unsafe extern "C" {
    // In mps.mm.
    fn lumen_mps_launch_kernel(
        name: *const c_char,
        buffers: *const *const u8,
        nbuffers: usize,
        args: *const *const u8,
        arg_lens: *const usize,
        nargs: usize,
        grid: *const usize,
        groups: i32,
        timed: i32,
        done: Completion,
        context: *mut c_void,
    ) -> i32;
}

pub(crate) fn encode(
    step: &Step,
    inputs: &[*const u8],
    output: *mut u8,
    scratch: *mut u8,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    use Primitive::*;
    let n = step.output.1.numel();
    if n > u32::MAX as usize {
        return Err(format!(
            "{}: {n} elements are more than MPS kernels index",
            step.primitive.name()
        ));
    }

    if let ReduceSum { .. } | ReduceMax { .. } = step.primitive {
        return super::reduce::mps::encode(step, inputs, output, scratch, keep);
    }
    if let Cumsum { .. } = step.primitive {
        return super::scan::mps::encode(step, inputs, output, scratch, keep);
    }
    if let Fusion { .. } = step.primitive {
        return crate::graph_compiler::mps::encode(step, inputs, output, scratch, keep);
    }
    if let LinearCrossEntropy = step.primitive {
        return super::dot_general::mps::encode_linear_cross_entropy(
            step, inputs, output, scratch, keep,
        );
    }

    if let Concatenate { .. } = step.primitive {
        return Err("concatenate runs in a fusion on MPS: compile with fuse".into());
    }

    let encode = match step.primitive {
        Add
        | Sub
        | Mul
        | Div
        | Max
        | Eq
        | Lt
        | Neg
        | Exp
        | Log
        | Sqrt
        | Tanh
        | Logistic
        | Cast { .. }
        | Select => super::elementwise::mps::encode,
        ReduceSum { .. } | ReduceMax { .. } | Cumsum { .. } | Concatenate { .. } => {
            unreachable!("encoded above")
        }
        DotGeneral { .. } => super::dot_general::mps::encode,

        Reshape { .. } | BroadcastInDim { .. } | Transpose { .. } | Slice { .. } => {
            super::layout::mps::encode
        }
        DynamicSlice { .. } | DynamicUpdateSlice => super::dynamic_slice::mps::encode,
        Gather { .. } | ScatterAdd { .. } => super::gather::mps::encode,
        Wait => super::wait::encode,
        ToHost | ToDevice => {
            unreachable!("a stage's transfers are between its plans (compiler::stages)")
        }
        Full { .. } | Iota { .. } | RandomBits { .. } => super::factory::mps::encode,
        Fusion { .. } | LinearCrossEntropy => unreachable!("encoded above"),
        FusionOutput { .. } => unreachable!("a fusion's kernel writes it: no step"),
        CustomCall { .. } => unreachable!("the plan calls a custom op's function itself"),
        PallasCall { .. } => unreachable!("a plan with a lumen.pallas kernel runs on a TPU"),
    };
    encode(step, inputs, output, keep)
}

pub(crate) fn scratch_bytes(p: &Primitive, inputs: &[&TensorType], _output: &TensorType) -> usize {
    match p {
        Primitive::ReduceSum { axes, .. } | Primitive::ReduceMax { axes } => {
            let accum = super::reduce::mps::accum_dtype(p, inputs[0].dtype);
            super::reduce::mps::scratch_bytes(inputs[0], axes, accum)
        }
        Primitive::Cumsum {
            axis, accum_dtype, ..
        } => super::scan::mps::scratch_bytes(inputs[0], *axis, *accum_dtype),
        Primitive::Fusion { body, .. } => crate::graph_compiler::mps::fusion_scratch_bytes(body),
        Primitive::LinearCrossEntropy => {
            super::dot_general::mps::linear_cross_entropy_scratch(inputs[0], inputs[1])
        }
        _ => 0,
    }
}

pub(crate) fn u32_arg(v: u32) -> Vec<u8> {
    v.to_ne_bytes().to_vec()
}

pub(crate) fn u64_arg(v: usize) -> Vec<u8> {
    (v as u64).to_ne_bytes().to_vec()
}

pub(crate) fn dims_arg(v: impl IntoIterator<Item = usize>) -> Vec<u8> {
    let mut v: Vec<u64> = v.into_iter().map(|d| d as u64).collect();
    if v.is_empty() {
        v.push(0);
    }
    as_bytes(&v).to_vec()
}

pub(crate) fn dims32_arg(v: impl IntoIterator<Item = usize>) -> Vec<u8> {
    let mut v: Vec<u32> = v.into_iter().map(|d| d as u32).collect();
    if v.is_empty() {
        v.push(0);
    }
    as_bytes(&v).to_vec()
}

pub(crate) fn element_arg(dtype: DType, value: Scalar) -> Vec<u8> {
    dispatch_dtype!(dtype, T => as_bytes(std::slice::from_ref(&T::from_scalar(value))).to_vec())
}

pub(crate) const BYTES_PER_THREAD: usize = 16;

pub(crate) fn elementwise_grid(n: usize, dtype: DType) -> Grid {
    Grid::Threads([
        n.div_ceil((BYTES_PER_THREAD / dtype.size_of()).max(1)),
        1,
        1,
    ])
}

pub(crate) use crate::custom_op::mps::Grid;

pub(crate) fn launch_step(
    step: &Step,
    kernel: &str,
    inputs: &[*const u8],
    output: *mut u8,
    args: &[Vec<u8>],
    grid: Grid,
    keep: Vec<Tensor>,
) -> Result<(), String> {
    let mut buffers = inputs.to_vec();
    buffers.push(output.cast_const());
    launch(kernel, &buffers, args, grid, keep, step.label)
}

pub(crate) fn launch(
    kernel: &str,
    buffers: &[*const u8],
    args: &[Vec<u8>],
    grid: Grid,
    keep: Vec<Tensor>,
    name: &'static str,
) -> Result<(), String> {
    let (sizes, groups) = match grid {
        Grid::Threads(sizes) => (sizes, false),
        Grid::Groups(sizes) => (sizes, true),
    };
    if sizes.contains(&0) {
        crate::profiler::clear_next_launch();
        return Ok(());
    }
    let c_name = CString::new(kernel).expect("kernel names have no NUL");
    let pointers: Vec<*const u8> = args.iter().map(|a| a.as_ptr()).collect();
    let lens: Vec<usize> = args.iter().map(Vec::len).collect();
    let (context, done, timed) = mps::submit(keep, name, kernel);
    let status = unsafe {
        lumen_mps_launch_kernel(
            c_name.as_ptr(),
            buffers.as_ptr(),
            buffers.len(),
            pointers.as_ptr(),
            lens.as_ptr(),
            args.len(),
            sizes.as_ptr(),
            groups.into(),
            timed.into(),
            done,
            context,
        )
    };
    if status != 0 {
        mps::cancel(context);
        return Err(format!("the MPS kernel {kernel} could not run ({status})"));
    }
    Ok(())
}
