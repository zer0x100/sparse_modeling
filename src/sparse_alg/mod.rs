mod l1_relaxzation;
mod matching_pursuit;

use crate::prelude::*;

pub use l1_relaxzation::{by_lasso::SparseAlgLasso, L1Relaxzation};
pub use matching_pursuit::{mp::Mp, omp::Omp, threshold_alg::ThresholdAlg, wmp::Wmp};

pub trait SparseAlg {
    fn solve(&self, mat: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>>;
}
