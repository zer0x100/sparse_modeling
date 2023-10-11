mod lasso;
mod matching_pursuit;

use crate::prelude::*;
pub use lasso::{
    fista::LassoFista, ista::LassoIsta, ista_lipshitz_search::LassoIstaLipshitzSearch,
};
pub use matching_pursuit::{
    mp::Mp,
    omp::Omp,
    wmp::Wmp,
    threshold_alg::ThresholdAlg,
};

pub trait SparseAlg {
    fn solve(&self, mat: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>>;
}
