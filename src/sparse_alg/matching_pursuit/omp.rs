use super::super::SparseAlg;
use crate::prelude::*;

pub struct Omp {
    threshold: f64,
}

impl Omp {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    pub fn set(&mut self, threshold: f64, iter_num: usize) {
        self.threshold = threshold;
    }
}

impl SparseAlg for Omp {
    fn solve(&self, mat: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
        if mat.shape()[1] != y.shape()[0] {
            return Err(anyhow!("mat's column size is different from y's size"));
        }
        if mat.shape()[0] > mat.shape()[1] {
            return Err(anyhow!("mat's row size is more than column size"));
        }

        //initialization
        let mut x: Array1<f64> = Array::zeros(y.shape()[0]);
        let mut r = y.clone();
        let mut support = HashSet::new();

        for _ in 0..mat.shape()[1] {
            //rの射影が最大となる列探索
            let mut max_j = 0;
            let mut max_proj = 0.;
            for j in 0..mat.shape()[1] {
                let row_j = mat.slice(s![.., j]);
                let proj = (row_j.t().dot(&r) / row_j.norm_l2()).abs();
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