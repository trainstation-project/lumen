//! Python bindings for the tracer's walk over a compiled call's arguments: the `Walker` and `ModuleWalk` classes of
//! `lumen.graph.tracing._C`. The walk gives the arguments' structure (what a plan depends on), each scalar's slot and
//! value, and the modules' weights; a module argument's walk is cached while it holds (`ModuleWalk::holds`), so a
//! call re-reads only its scalars.

use std::collections::{HashMap, HashSet};

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyList, PyString, PyTuple, PyType};
use pyo3::{PyTraverseError, PyVisit};

use crate::tensor::python::PyTensor;

/// Where a scalar or leaf field sits: in the walked module itself (held weakly through the cache's key, so the walk
/// doesn't keep its module alive), or in another module, list or tuple of its tree.
enum Holder {
    Root,
    Of(Py<PyAny>),
}

impl Holder {
    /// The holder's object, `module` for the root.
    fn bind<'py>(&self, module: &Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        match self {
            Holder::Root => module.clone(),
            Holder::Of(o) => o.bind(module.py()).clone(),
        }
    }
}

/// One module argument's walk, kept while it holds: its structure (a `_Structure`, hashed once), its tensors' storage
/// ids in walk order and the tensors, each scalar's path and `(holder, key)`, each leaf field's `(holder, key,
/// value)` (`None` for a scalar, re-read each call), and each list in it with its elements' identities.
#[pyclass(module = "lumen.graph.tracing._C")]
pub(crate) struct ModuleWalk {
    structure: Option<Py<PyAny>>,
    ids: Vec<usize>,
    tensors: Vec<Py<PyAny>>,
    scalars: Vec<(Py<PyTuple>, Holder, Py<PyAny>)>,
    leaves: Vec<(Holder, Py<PyString>, Option<Py<PyAny>>)>,
    lists: Vec<(Py<PyList>, Vec<usize>)>,
}

#[pymethods]
impl ModuleWalk {
    /// Visits the objects the walk holds, for Python's cycle collector.
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.structure)?;

        for t in &self.tensors {
            visit.call(t)?;
        }

        for (path, holder, key) in &self.scalars {
            visit.call(path)?;
            visit.call(key)?;

            if let Holder::Of(o) = holder {
                visit.call(o)?;
            }
        }

        for (holder, key, value) in &self.leaves {
            visit.call(key)?;
            visit.call(value)?;

            if let Holder::Of(o) = holder {
                visit.call(o)?;
            }
        }

        for (list, _) in &self.lists {
            visit.call(list)?;
        }

        Ok(())
    }

    /// Drops the held objects, breaking any cycle through them.
    fn __clear__(&mut self) {
        self.structure = None;
        self.tensors.clear();
        self.scalars.clear();
        self.leaves.clear();
        self.lists.clear();
    }
}

impl ModuleWalk {
    /// Whether `module` still has this walk: every list holds the same elements, every leaf field the same object,
    /// or a scalar where it held one.
    fn holds(&self, walker: &Walker, module: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = module.py();

        for (list, snapshot) in &self.lists {
            let list = list.bind(py);

            if list.len() != snapshot.len()
                || list
                    .iter()
                    .zip(snapshot)
                    .any(|(v, &k)| v.as_ptr() as usize != k)
            {
                return Ok(false);
            }
        }

        for (holder, key, value) in &self.leaves {
            let current = holder.bind(module).getattr(key.bind(py))?;

            let same = match value {
                Some(v) => current.is(v.bind(py)),
                None => walker.is_scalar(&current)?,
            };

            if !same {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

/// A walk's outputs while it runs: each scalar's `(slot, value)`, the module tensors' storage ids and the tensors.
struct Found<'py> {
    slots: Vec<Bound<'py, PyAny>>,
    ids: Vec<usize>,
    tensors: Vec<Bound<'py, PyAny>>,
}

/// What a module argument's walk records besides its outputs: the module, and its scalars, leaf fields and lists.
struct Record<'py> {
    module: Bound<'py, PyAny>,
    scalars: Vec<(Py<PyTuple>, Holder, Py<PyAny>)>,
    leaves: Vec<(Holder, Py<PyString>, Option<Py<PyAny>>)>,
    lists: Vec<(Py<PyList>, Vec<usize>)>,
}

impl Record<'_> {
    /// `holder` as recorded: the root for the walked module.
    fn holder(&self, holder: &Bound<'_, PyAny>) -> Holder {
        match holder.is(&self.module) {
            true => Holder::Root,
            false => Holder::Of(holder.clone().unbind()),
        }
    }
}

/// The tracer's walk over a compiled call's arguments, with what it needs from Python: the module class, the scalar
/// test (`nn.is_scalar`), the scalar marker (`_SCALAR`), each module class's field names, the cached structure's
/// class (`_Structure`) and the cache of module walks (a `WeakKeyDictionary` by module).
#[pyclass(module = "lumen.graph.tracing._C")]
pub(crate) struct Walker {
    module: Py<PyType>,
    scalar: Py<PyAny>,
    marker: Py<PyAny>,
    fields: Py<PyAny>,
    structure: Py<PyAny>,
    walks: Py<PyAny>,
}

impl Walker {
    /// Whether `x` is a scalar a compiled function can take at runtime: a Python float, int or bool (not a subclass),
    /// else as `nn.is_scalar` says (a numpy scalar).
    fn is_scalar(&self, x: &Bound<'_, PyAny>) -> PyResult<bool> {
        if x.is_exact_instance_of::<PyFloat>()
            || x.is_exact_instance_of::<PyInt>()
            || x.is_exact_instance_of::<PyBool>()
        {
            return Ok(true);
        }

        if x.is_none() || x.is_instance_of::<PyString>() {
            return Ok(false);
        }

        self.scalar.bind(x.py()).call1((x,))?.is_truthy()
    }

    /// The structure of `x`, at `path` of argument `i` (inside a module when `module`), held at `key` of `holder`,
    /// adding its scalars, module tensors and, into `record` (a module argument's walk being built), its scalars'
    /// holders, leaf fields and lists.
    #[allow(clippy::too_many_arguments)]
    fn tree<'py>(
        &self,
        x: &Bound<'py, PyAny>,
        i: usize,
        path: &Bound<'py, PyTuple>,
        module: bool,
        holder: Option<(&Bound<'py, PyAny>, &Bound<'py, PyAny>)>,
        found: &mut Found<'py>,
        record: &mut Option<Record<'py>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = x.py();

        // A path one step further: `(*path, key)`.
        let further = |key: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyTuple>> {
            PyTuple::new(
                py,
                path.iter()
                    .chain(std::iter::once(key.clone()))
                    .collect::<Vec<_>>(),
            )
        };

        if x.cast::<PyTensor>().is_ok() {
            if module {
                if !x.getattr("_is_parameter")?.is_truthy()? {
                    return Err(PyTypeError::new_err(format!(
                        "a module's weights must be whole meta tensors, got a {} tensor of shape {}",
                        x.getattr("device")?.str()?,
                        x.getattr("shape")?.str()?
                    )));
                }

                found.ids.push(x.getattr("storage_id")?.extract()?);
                found.tensors.push(x.clone());
            }

            let shape = PyTuple::new(
                py,
                x.getattr("shape")?
                    .try_iter()?
                    .collect::<PyResult<Vec<_>>>()?,
            )?;

            return Ok(PyTuple::new(
                py,
                [
                    PyString::new(py, "tensor").into_any(),
                    x.getattr("dtype")?,
                    shape.into_any(),
                    x.getattr("names")?,
                    x.getattr("placement")?,
                ],
            )?
            .into_any());
        }

        if x.is_instance(self.module.bind(py))? {
            let names = self.fields.bind(py).call1((x.get_type(),))?;
            let mut fields = Vec::new();

            for n in names.try_iter()? {
                let n = n?;
                let value = x.getattr(n.cast::<PyString>()?)?;
                let sub =
                    self.tree(&value, i, &further(&n)?, true, Some((x, &n)), found, record)?;
                fields.push(PyTuple::new(py, [n, sub])?.into_any());
            }

            return Ok(PyTuple::new(
                py,
                [
                    x.get_type().into_any(),
                    PyTuple::new(py, fields)?.into_any(),
                ],
            )?
            .into_any());
        }

        if x.is_instance_of::<PyList>() || x.is_instance_of::<PyTuple>() {
            if let (Some(r), Ok(list)) = (record.as_mut(), x.cast::<PyList>()) {
                let ids = list.iter().map(|v| v.as_ptr() as usize).collect();
                r.lists.push((list.clone().unbind(), ids));
            }

            let mut items = Vec::new();

            for (k, v) in x.try_iter()?.enumerate() {
                let k = k.into_pyobject(py)?.into_any();
                items.push(self.tree(
                    &v?,
                    i,
                    &further(&k)?,
                    module,
                    Some((x, &k)),
                    found,
                    record,
                )?);
            }

            return Ok(PyTuple::new(
                py,
                [x.get_type().into_any(), PyTuple::new(py, items)?.into_any()],
            )?
            .into_any());
        }

        let scalar = self.is_scalar(x)?;

        if let (Some(r), Some((holder, key))) = (record.as_mut(), holder) {
            let held = r.holder(holder);

            if holder.is_instance(self.module.bind(py))? {
                let value = (!scalar).then(|| x.clone().unbind());
                r.leaves
                    .push((held, key.cast::<PyString>()?.clone().unbind(), value));
            }

            if scalar {
                r.scalars.push((
                    path.clone().unbind(),
                    r.holder(holder),
                    key.clone().unbind(),
                ));
            }
        }

        if scalar {
            let slot = PyTuple::new(
                py,
                [i.into_pyobject(py)?.into_any(), path.clone().into_any()],
            )?;
            found
                .slots
                .push(PyTuple::new(py, [slot.into_any(), x.clone()])?.into_any());
            return Ok(self.marker.bind(py).clone());
        }

        Ok(PyTuple::new(py, [PyString::new(py, "static").into_any(), x.clone()])?.into_any())
    }

    /// `module`'s walk as an argument: the cached one while it holds, else walked again and cached.
    fn module_walk<'py>(&self, module: &Bound<'py, PyAny>) -> PyResult<Bound<'py, ModuleWalk>> {
        let py = module.py();
        let walks = self.walks.bind(py);
        let cached = walks.call_method1("get", (module,))?;

        if let Ok(walk) = cached.cast::<ModuleWalk>()
            && walk.borrow().holds(self, module)?
        {
            return Ok(walk.clone());
        }

        let mut found = Found {
            slots: Vec::new(),
            ids: Vec::new(),
            tensors: Vec::new(),
        };

        let mut record = Some(Record {
            module: module.clone(),
            scalars: Vec::new(),
            leaves: Vec::new(),
            lists: Vec::new(),
        });

        let structure = self.tree(
            module,
            0,
            &PyTuple::empty(py),
            false,
            None,
            &mut found,
            &mut record,
        )?;
        let record = record.expect("the record");

        let walk = Bound::new(
            py,
            ModuleWalk {
                structure: Some(self.structure.bind(py).call1((structure,))?.unbind()),
                ids: found.ids,
                tensors: found.tensors.into_iter().map(Bound::unbind).collect(),
                scalars: record.scalars,
                leaves: record.leaves,
                lists: record.lists,
            },
        )?;

        walks.set_item(module, &walk)?;
        Ok(walk)
    }
}

#[pymethods]
impl Walker {
    /// A walker using the module class, the scalar test, the scalar marker, the field names of a module class, the
    /// cached structure's class and the cache of module walks.
    #[new]
    fn new(
        module: Py<PyType>,
        scalar: Py<PyAny>,
        marker: Py<PyAny>,
        fields: Py<PyAny>,
        structure: Py<PyAny>,
        walks: Py<PyAny>,
    ) -> Self {
        Walker {
            module,
            scalar,
            marker,
            fields,
            structure,
            walks,
        }
    }

    /// One pass over `args`: `((structure, shared), slots, weights)`, the arguments' structure (a marker at each
    /// scalar) with which module tensors share storage (each its first sharer's index), each scalar's `((argument,
    /// path), value)`, and the modules' weights, each once, in order. A module argument's walk is cached and only its
    /// scalars re-read while it holds.
    fn walk<'py>(
        &self,
        args: &Bound<'py, PyTuple>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyList>, Bound<'py, PyList>)> {
        let py = args.py();

        let mut found = Found {
            slots: Vec::new(),
            ids: Vec::new(),
            tensors: Vec::new(),
        };

        let mut structure = Vec::new();

        for (i, a) in args.iter().enumerate() {
            if !a.is_instance(self.module.bind(py))? {
                structure.push(self.tree(
                    &a,
                    i,
                    &PyTuple::empty(py),
                    false,
                    None,
                    &mut found,
                    &mut None,
                )?);
                continue;
            }

            let walk = self.module_walk(&a)?;
            let walk = walk.borrow();
            structure.push(
                walk.structure
                    .as_ref()
                    .expect("a walked structure")
                    .bind(py)
                    .clone(),
            );
            found.ids.extend(&walk.ids);
            found
                .tensors
                .extend(walk.tensors.iter().map(|t| t.bind(py).clone()));

            for (path, holder, key) in &walk.scalars {
                let holder = holder.bind(&a);

                let value =
                    match holder.is_instance_of::<PyList>() || holder.is_instance_of::<PyTuple>() {
                        true => holder.get_item(key.bind(py))?,
                        false => holder.getattr(key.bind(py).cast::<PyString>()?)?,
                    };

                let slot = PyTuple::new(
                    py,
                    [
                        i.into_pyobject(py)?.into_any(),
                        path.bind(py).clone().into_any(),
                    ],
                )?;
                found
                    .slots
                    .push(PyTuple::new(py, [slot.into_any(), value])?.into_any());
            }
        }

        let mut first: HashMap<usize, usize> = HashMap::new();
        let shared: Vec<usize> = found
            .ids
            .iter()
            .enumerate()
            .map(|(n, &k)| *first.entry(k).or_insert(n))
            .collect();
        let mut seen = HashSet::new();
        let mut weights = Vec::new();

        for (t, &id) in found.tensors.iter().zip(&found.ids) {
            if seen.insert(id) {
                weights.push(t.clone());
            }
        }

        let walked = PyTuple::new(
            py,
            [
                PyTuple::new(py, structure)?.into_any(),
                PyTuple::new(py, shared)?.into_any(),
            ],
        )?;
        Ok((
            walked,
            PyList::new(py, found.slots)?,
            PyList::new(py, weights)?,
        ))
    }
}

/// Adds the `Walker` and `ModuleWalk` classes to `lumen.graph.tracing._C`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Walker>()?;
    m.add_class::<ModuleWalk>()?;
    Ok(())
}
