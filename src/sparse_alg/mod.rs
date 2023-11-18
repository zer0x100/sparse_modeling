//! # Sparse Alg
//! 
//! 'sparse_alg' is a collection of algorithms for calculate sparse solutions
mod l1_relaxzation;
mod matching_pursuit;

use crate::prelude::*;

pub use l1_relaxzation::{by_lasso::SparseAlgLasso, focuss::L1Focuss, L1Relaxzation};
pub use matching_pursuit::{mp::Mp, omp::Omp, threshold_alg::ThresholdAlg, wmp::Wmp};

pub trait SparseAlg {
    fn solve(&self, mat: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>>;
}
