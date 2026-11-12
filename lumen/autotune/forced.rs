//! Configs forced in place of autotuning (`lumen.autotune.use` in Python): while one of a kernel family's config
//! type is forced, every step the family handles is built with it, untimed, whether or not autotuning is on.

use std::any::Any;
use std::sync::{Arc, PoisonError, RwLock};

/// A forced config, of whichever kernel family's config type.
pub(crate) type Forced = Arc<dyn Any + Send + Sync>;

/// The configs of each enclosing `lumen.autotune.use` block, innermost last.
static FORCED: RwLock<Vec<Vec<Forced>>> = RwLock::new(Vec::new());

/// Enters a block forcing `configs`, invalidating compiled plans so they are built with them.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn push(configs: Vec<Forced>) {
    FORCED
        .write()
        .unwrap_or_else(PoisonError::into_inner)
        .push(configs);

    crate::graph_compiler::config::touch();
}

/// Leaves the innermost block, invalidating compiled plans built with its configs.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn pop() {
    FORCED.write().unwrap_or_else(PoisonError::into_inner).pop();
    crate::graph_compiler::config::touch();
}

/// Whether any config is forced.
pub(crate) fn forcing() -> bool {
    FORCED
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .any(|configs| !configs.is_empty())
}

/// The forced config of type `C` of the innermost block that forces one, if any.
pub(crate) fn forced<C: Clone + 'static>() -> Option<C> {
    let blocks = FORCED.read().unwrap_or_else(PoisonError::into_inner);

    blocks
        .iter()
        .rev()
        .find_map(|configs| configs.iter().find_map(|c| c.downcast_ref::<C>().cloned()))
}
