#![recursion_limit = "256"]
#![allow(unused)]
#![allow(incomplete_features)]
#![feature(inherent_associated_types)]
pub mod tensor;
pub mod module;
pub mod sgd;
pub mod layer;

pub use tensor::Tensor;
pub use module::{Module};
pub use layer::{Dense, ReLU, Sequential};
pub use sgd::SGD;
