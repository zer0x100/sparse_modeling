pub use super::L1Relaxzation;
pub use crate::prelude::*;

pub struct SparseAlgLasso {
    bs_lasso_lambda: f64, //basis pursuit(matrix's columns are normalized)'s lambda of lasso
    lasso_alg: Box<dyn LassoAlg>,
}

impl SparseAlgLasso {
    #[allow(dead_code)]
    pub fn new(bs_lasso_lambda: f64, lasso_alg: Box<dyn LassoAlg>) -> Self {
        Self { bs_lasso_lambda, lasso_alg }
    }
    #[allow(dead_code)]
    pub fn set(&mut self, bs_lasso_lambda: f64, lasso_alg: Box<dyn LassoAlg>) {
        self.bs_lasso_lambda = bs_lasso_lambda;
        self.lasso_alg = lasso_alg;
    }
}

impl L1Relaxzation for SparseAlgLasso {
    fn solve_l1(&self, mat: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
        let solution_l1 = self.lasso_alg.solve(mat, y, self.bs_lasso_lambda)?;

        Ok(solution_l1)
    }
}
