//! A node of the graph (one primitive application), the scope and source lines it was traced at, and their interning
//! as `'static` slices.

use std::collections::BTreeSet;
use std::sync::{Mutex, PoisonError};

use super::{Primitive, Var};

/// One primitive application producing a single output value.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub primitive: Primitive,
    /// Values the primitive reads, in operand order.
    pub inputs: Vec<Var>,
    pub output: Var,
    /// Named ranges the node was traced inside, outermost first.
    pub scope: Scope,
    /// Python source lines the node came from.
    pub sources: Lines,
    /// Display name overriding the primitive's own, set for fused or renamed nodes.
    pub label: Option<&'static str>,
    /// Whether the node computes gradients: traced by autograd's backward pass (a transpose rule, a
    /// `Function`'s `backward`), or rebuilt from such a node.
    pub backward: bool,
}

/// Interned stack of named ranges a node was traced in.
pub type Scope = &'static [Range];

/// Interned list of `(file, line)` source locations.
pub type Lines = &'static [(&'static str, u32)];

/// One entry of a scope: a range name and which call of it this is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Range {
    pub name: &'static str,
    /// Distinguishes separate invocations of the same named range.
    pub call: u64,
}

/// Returns a `'static` copy of `label`, leaking each distinct string once.
pub(crate) fn intern(label: String) -> &'static str {
    static LABELS: Mutex<BTreeSet<&'static str>> = Mutex::new(BTreeSet::new());
    let mut labels = LABELS.lock().unwrap_or_else(PoisonError::into_inner);

    if let Some(&interned) = labels.get(label.as_str()) {
        return interned;
    }

    let interned: &'static str = Box::leak(label.into_boxed_str());
    labels.insert(interned);
    interned
}

/// Returns a `'static` copy of `lines`, leaking each distinct list once.
pub(crate) fn intern_lines(lines: &[(&'static str, u32)]) -> Lines {
    static LINES: Mutex<BTreeSet<Lines>> = Mutex::new(BTreeSet::new());
    let mut interned = LINES.lock().unwrap_or_else(PoisonError::into_inner);

    if let Some(&lines) = interned.get(lines) {
        return lines;
    }

    let lines: Lines = Box::leak(lines.to_vec().into_boxed_slice());
    interned.insert(lines);
    lines
}

/// Returns a `'static` copy of `ranges`, leaking each distinct scope once.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn intern_scope(ranges: &[Range]) -> Scope {
    static SCOPES: Mutex<BTreeSet<Scope>> = Mutex::new(BTreeSet::new());
    let mut scopes = SCOPES.lock().unwrap_or_else(PoisonError::into_inner);

    if let Some(&interned) = scopes.get(ranges) {
        return interned;
    }

    let interned: Scope = Box::leak(ranges.to_vec().into_boxed_slice());
    scopes.insert(interned);
    interned
}
