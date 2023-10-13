use super::super::SparseAlg;
use crate::prelude::*;

pub struct Mp {
    threshold: f64,
    iter_num: usize,
}

impl Mp {
    #[allow(dead_code)]
    pub fn new(threshold: f64, iter_num: usize) -> Self {
        Self {
            threshold,
            iter_num,
        }
    }
    #[allow(dead_code)]
    pub fn set(&mut self, threshold: f64, iter_num: usize) {
        self.threshold = threshold;
        self.iter_num = iter_num;
    }
}

impl SparseAlg for Mp {
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

        for _ in 0..self.iter_num {
            //rの射影が最大となる列探索
            let (target_idx, _) = mat_normalized.t().dot(&r)
                .iter()
                .map(|v| v.abs())
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .expect("failed to get max projection");
            
            //support update
            support.insert(target_idx);

            //update tentative solution(x)
            let target_col = mat.slice(s![.., target_idx]).to_owned();
            let temp = target_col.t().dot(&r) / target_col.norm_l2().powf(2.0);
            x[target_idx] += temp;

            //update residual(r)
            r = r - temp * target_col;

            if r.norm_l2() < self.threshold {
                break;
            }
        }

        Ok(x)
    }
}
