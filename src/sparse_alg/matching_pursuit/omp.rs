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
        let mat_normalized = normalize_columns(mat).unwrap();
        let mut x: Array1<f64> = Array::zeros(mat.shape()[1]);
        let mut r = y.clone();
        let mut support = HashSet::new();

        for _ in 0..mat.shape()[1] {
            //rの射影が最大となる列探索
            let (target_col, _) = mat_normalized.t().dot(&r)
                .iter()
                .map(|v| v.abs())
                .enumerate()
                .filter(|(i, _)| !support.contains(i))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .expect("failed to get max projection");

            //support update
            support.insert(target_col);

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
