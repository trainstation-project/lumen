use crate::Tensor;
use crate::kernels::Op;
use crate::tensor::scalar::Scalar;

pub type DummyOpKernel = fn(&Tensor, Scalar);

pub static DUMMY_OP: Op<DummyOpKernel> = Op::new(op_name!("dummy_op"), |_| None);

#[cfg(feature = "python")]
pub static DUMMY_OP_PY: Op<crate::kernels::python::KernelHandle> =
    Op::new(op_name!("dummy_op__python"), |_| None);

pub fn dummy_op(t: &Tensor, value: Scalar) {
    #[cfg(feature = "python")]
    if let Some(handle) = crate::kernels::python::handle_for(
        DUMMY_OP.name(),
        crate::kernels::DispatchKey::of(t.device()),
    ) && handle.launch(
        DUMMY_OP.name(),
        t.device(),
        t.dtype(),
        t.shape(),
        t.strides(),
        t.data_ptr() as usize,
        value,
    ) {
        return;
    }
    DUMMY_OP.dispatch(t.device())(t, value);
}
