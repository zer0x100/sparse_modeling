//! # Sparse Modeling
//! 
//! 'sparse_modeling' is a collection of utilities to calculate sparse solutions.
pub mod cg;
pub mod gen_signal;
pub mod lasso_alg;
pub mod math_func;
pub mod mk_matrix;
pub mod sparse_alg;
mod prelude {
    //! # Prelude
    //! functions, structures, and so on, used throughout this crate
    pub use crate::gen_signal::*;
    pub use crate::lasso_alg::*;
    pub use crate::math_func::*;
    pub use crate::mk_matrix::*;
    pub use crate::sparse_alg::*;
    pub use anyhow::{anyhow, Result};
    pub use ndarray::prelude::*;
    pub use ndarray_linalg::{c64, generate, svd::SVD, Inverse, Norm, Scalar};
    pub use plotters::prelude::*;
    pub use rand::{
        distributions::{Distribution, Uniform},
        rngs::ThreadRng,
        Rng,
    };
    pub use std::cmp;
    pub use std::collections::HashSet;
    pub use std::{f64::consts::PI, fs};
    pub const F64_EPS: f64 = 1e-10;

    pub use crate::cg::*;
}