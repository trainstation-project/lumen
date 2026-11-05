//! Metal layout: the generator of transposes fused with elementwise ops, staged through tiles, and the
//! reshape steps the plan copies outputs with. Other lone layout steps run as one-node elementwise fusions.

pub(crate) mod transpose;
