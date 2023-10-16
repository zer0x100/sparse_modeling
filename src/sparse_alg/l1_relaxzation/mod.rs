/*
凸緩和により、スパースな解を求めるアルゴリズム。
L0ノルムをLp(p = 0~1)など凸なものに緩和する。
*/
pub mod by_lasso;
pub mod focuss;

use crate::prelude::*;

pub trait L1Relaxzation {
    fn solve_l1(&self, mat: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>>;
}

impl<T: L1Relaxzation> SparseAlg for T {
    //L1緩和では、Aのノルムが大きい列に対応する要素が非ゼロになりやすいバイアスがあるため、
    //スケーリングしたものを解とする。
    fn solve(&self, mat: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
        match is_underestimated_sys(mat, y) {
            Err(msg) => return Err(msg),
            Ok(_) => (),
        }

        //calucalate mat's column sizes and normalize them
        let scaling = mat
            .columns()
            .into_iter()
            .map(|column| column.norm_l2())
            .enumerate();
        let normalized_mat = normalize_columns(mat)?;

        //solve L1 minimization
        let mut solution = self
            .solve_l1(&normalized_mat, y)
            .expect("failed to solve l1 minimization");

        //scaling
        scaling.for_each(|(i, scale)| {
            solution[i] *= scale;
        });

        Ok(solution)
    }
}
