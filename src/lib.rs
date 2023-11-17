mod cg;
mod data_convert;
mod gen_signal;
mod lasso_alg;
mod math_func;
mod mk_matrix;
mod sparse_alg;
pub mod prelude {
    pub use crate::data_convert::*;
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
