//! A plan's steps: the buffers they read and write, their memoized kernels and tuned launches.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use crate::graph::{Primitive, Scope, TensorType, View};
use crate::tensor::contiguous_strides;

/// Where a step reads or writes a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Buffer {
    /// The caller's input `i`, or a donated input that holds an output.
    Input(usize),
    /// The freshly allocated output `i`.
    Output(usize),
    /// A byte offset into the shared workspace.
    Workspace(usize),
    /// Input `i` passed as a host-side scalar.
    Scalar(usize),
}

impl fmt::Display for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Buffer::Input(i) => write!(f, "in{i}"),
            Buffer::Output(i) => write!(f, "out{i}"),
            Buffer::Workspace(offset) => write!(f, "ws+{offset}"),
            Buffer::Scalar(i) => write!(f, "s{i}"),
        }
    }
}

/// One kernel launch: a primitive with the buffers it reads and writes.
#[derive(Debug, Clone)]
pub struct Step {
    pub primitive: Primitive,
    /// Name shown in profiles and error messages.
    pub label: &'static str,
    /// Operand buffers with their types.
    pub inputs: Vec<(Buffer, TensorType)>,
    /// Per operand, a strided view of its buffer, or `None` when it is read contiguously.
    pub views: Vec<Option<View>>,
    /// The buffer and type the step writes.
    pub output: (Buffer, TensorType),
    /// Further outputs of a multi-output fusion or custom call, in output index order.
    pub extra_outputs: Vec<(Buffer, TensorType)>,
    /// Per output (main output first), where it is written inside a larger buffer, if anywhere.
    pub output_views: Vec<Option<View>>,
    /// Workspace `(offset, bytes)` reserved for the kernel's own scratch.
    pub scratch: Option<(usize, usize)>,
    /// Profiler scope stack the step was traced under.
    pub scope: Scope,
    /// Python source locations `(file, line)` the step was traced from.
    pub sources: Vec<(&'static str, u32)>,
    /// The autotuned kernel choice, if one was made.
    pub tuned: Option<Tuned>,
    /// The kernel the backend's codegen generated for the step, when it keeps one.
    pub kernel: Option<Arc<dyn Any + Send + Sync>>,
    /// What a backend prepared to launch the step, per layout of the tensors it ran on.
    pub memo: Memo,
}

/// A step's launch preparations (generated helpers, compiled kernels), each kept under a key a backend
/// derives from what it was prepared for (its tensors' layouts), so a step that runs again on the same
/// layouts skips regenerating them. Clones of a step share it.
#[derive(Clone, Default)]
pub struct Memo(Arc<std::sync::Mutex<std::collections::HashMap<u64, Arc<dyn Any + Send + Sync>>>>);

impl Memo {
    /// The preparation kept under `key`, made by `make` and kept on first use; a failure isn't kept.
    pub(crate) fn get<T: Send + Sync + 'static>(
        &self,
        key: u64,
        make: impl FnOnce() -> Result<T, String>,
    ) -> Result<Arc<T>, String> {
        let kept = self
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&key)
            .cloned();

        if let Some(t) = kept.and_then(|k| k.downcast::<T>().ok()) {
            return Ok(t);
        }

        let made = Arc::new(make()?);

        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(key, made.clone());

        Ok(made)
    }
}

impl fmt::Debug for Memo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Memo")
    }
}

/// A kernel picked by autotuning, with the backend-specific launch parameters it needs.
#[derive(Clone)]
pub struct Tuned {
    /// Name of the chosen kernel.
    pub kernel: String,
    /// Backend launch configuration, recovered with `Step::launch`.
    launch: Arc<dyn Any + Send + Sync>,
}

impl Tuned {
    /// Wraps a backend's launch configuration under the kernel name.
    pub(crate) fn new<L: Send + Sync + 'static>(kernel: String, launch: L) -> Self {
        Tuned {
            kernel,
            launch: Arc::new(launch),
        }
    }
}

impl fmt::Debug for Tuned {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.kernel)
    }
}

impl Step {
    /// The tuned launch configuration, if one of type `L` was stored.
    pub(crate) fn launch<L: 'static>(&self) -> Option<&L> {
        self.tuned.as_ref()?.launch.downcast_ref::<L>()
    }

    /// Element strides each operand is read with: its view's, or contiguous ones.
    pub(crate) fn input_strides(&self) -> Vec<Vec<usize>> {
        let strides = |(_, ty): &(Buffer, TensorType), view: &Option<View>| match view {
            Some(view) => view.strides.clone(),
            None => contiguous_strides(&ty.shape),
        };

        self.inputs
            .iter()
            .zip(&self.views)
            .map(|(i, v)| strides(i, v))
            .collect()
    }

    /// Element strides each output is written with: its view's, or contiguous ones.
    pub(crate) fn output_strides(&self) -> Vec<Vec<usize>> {
        std::iter::once(&self.output)
            .chain(&self.extra_outputs)
            .zip(&self.output_views)
            .map(|((_, ty), view)| match view {
                Some(view) => view.strides.clone(),
                None => contiguous_strides(&ty.shape),
            })
            .collect()
    }
}
