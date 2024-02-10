use super::super::SparseAlg;
use crate::prelude::*;

pub struct Omp {
    threshold: f64,
    iter_num: usize,
}

impl Omp {
    #[allow(dead_code)]
    pub fn new(threshold: f64, iter_num: usize) -> Self {
        Self { threshold, iter_num }
    }
    #[allow(dead_code)]
    pub fn set(&mut self, threshold: f64, iter_num: usize) {
        self.threshold = threshold;
        self.iter_num = iter_num;
    }
}

impl SparseAlg for Omp {
    fn solve(&self, mat: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
        match is_underestimated_sys(mat, y) {
            Err(msg) => return Err(msg),
            Ok(_) => (),
        }

        //initialization
        let mat_normalized = normalize_columns(mat).unwrap();
        let mut x: Array1<f64> = Array::zeros(mat.shape()[1]);
        let mut r = y.clone();
        let mut support = HashSet::new();

        for _ in 0..std::cmp::min(mat.shape()[1], self.iter_num) {
            //rの射影が最大となる列探索
            let (target_idx, _) = mat_normalized
                .t()
                .dot(&r)
                .iter()
                .map(|v| v.abs())
                .enumerate()
                .filter(|(i, _)| !support.contains(i))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .expect("failed to get max projection");

            //support update
            support.insert(target_idx);

            //update tentative solution(x)
            x = lsm_with_support(mat, y, &support).expect("Can't solve lsm");

            //update residual(r)
            r = y - mat.dot(&x);

            if r.norm_l2() < self.threshold {
                break;
            }
        }

        Ok(x)
    }
}
