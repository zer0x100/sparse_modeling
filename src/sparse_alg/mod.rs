mod l1_relaxzation;
mod matching_pursuit;

pub use matching_pursuit::{mp::Mp, omp::Omp, threshold_alg::ThresholdAlg, wmp::Wmp};
pub use l1_relaxzation::{*, by_lasso::SparseAlgLasso};

pub trait SparseAlg {
    fn solve(&self, mat: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>>;
}
