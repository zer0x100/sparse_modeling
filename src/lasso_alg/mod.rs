//! # Lasso Alg
//! 
//! 'lasso_alg' is a collection of tools for solve lasso
//! (minimize (1/2λ)||y - Ax||^(2) + ||x||_(1) <- minimize).

mod fista;
mod irls;
mod ista;
mod ista_lipshitz_search;
mod ssf;
mod irls_shrinkage;
#[cfg(test)]
mod tests;

use crate::prelude::*;

pub use fista::LassoFista;
pub use irls::LassoIrls;
pub use ista::LassoIsta;
pub use ista_lipshitz_search::LassoIstaLipshitzSearch;
pub use ssf::LassoSSF;
pub use irls_shrinkage::LassoIrlsShrink;

pub trait LassoAlg {
    fn solve(&self, mat: &Array2<f64>, y: &Array1<f64>, lambda: f64) -> Result<Array1<f64>>;
}
