use super::super::SparseAlg;
use crate::prelude::*;

pub struct Omp {
    threshold: f64,
}

impl Omp {
    #[allow(dead_code)]
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }
    #[allow(dead_code)]
    pub fn set(&mut self, threshold: f64) {
        self.threshold = threshold;
    }
}

impl SparseAlg for Omp {
    fn solve(&self, mat: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
        if mat.shape()[0] != y.shape()[0] || mat.shape()[0] > mat.shape()[1] {
            return Err(anyhow!(format!(
                "mat's shape is {}x{} / y's size is {}",
                mat.shape()[0],
                mat.shape()[1],
                y.shape()[0]
            )
            .to_string()));
        }

        //initialization
        let mut x: Array1<f64> = Array::zeros(mat.shape()[1]);
        let mut r = y.clone();
        let mut support = HashSet::new();

        for _ in 0..mat.shape()[1] {
            //rの射影が最大となる列探索
            let mut max_j = 0;
            let mut max_proj = 0.;
            for j in 0..mat.shape()[1] {
                let column_j = mat.slice(s![.., j]);
                let proj = (column_j.t().dot(&r) / column_j.norm_l2()).abs();
                if max_proj <= proj {
                    max_j = j;
                    max_proj = proj;
                }
            }
            //support update
            support.insert(max_j);

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
