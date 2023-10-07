/*
Lassoを解くアルゴリズム実装
(1/2λ)||y - Ax||^(2) + ||x||_(1) <- minimize
}}
 */
mod fista;
mod ista;
mod ista_lipshitz_search;

use crate::prelude::*;
pub use ista::LassoIsta;
pub use ista_lipshitz_search::LassoIstaLipshitzSearch;
pub use fista::LassoFista;

pub trait Lasso {
    fn solve(&self, mat: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>>;
}
