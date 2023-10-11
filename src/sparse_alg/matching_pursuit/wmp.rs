use super::super::SparseAlg;
use crate::prelude::*;

pub struct Wmp {
    threshold: f64,
    iter_num: usize,
    proj_ratio: f64,
}

impl Wmp {
    pub fn new(threshold: f64, iter_num: usize, proj_ratio: f64) -> Result<Self> {
        if proj_ratio >= 1.0 || proj_ratio <= 0.0 {
            return Err(anyhow!(format!(
                "proj_threhold is {}, it is needed between 0 and 1",
                proj_ratio
            )
            .to_string()));
        }
        Ok(Self {
            threshold,
            iter_num,
            proj_ratio,
        })
    }

    pub fn set(&mut self, threshold: f64, iter_num: usize, proj_ratio: f64) -> Result<()> {
        if proj_ratio >= 1.0 || proj_ratio <= 0.0 {
            return Err(anyhow!(""));
        }
        self.threshold = threshold;
        self.iter_num = iter_num;
        self.proj_ratio = proj_ratio;
        Ok(())
    }
}

impl SparseAlg for Wmp {
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

        for _ in 0..self.iter_num {
            //rの射影が最大となる列探索
            let mut target_j = 0;
            for j in 0..mat.shape()[1] {
                let column_j = mat.slice(s![.., j]);
                let proj = (column_j.t().dot(&r) / column_j.norm_l2()).abs();
                if proj >= self.proj_ratio * r.norm_l2() {
                    target_j = j;
                    break;
                }
            }
            //support update
            support.insert(target_j);

            //update tentative solution(x)
            let target_column = mat.slice(s![.., target_j]);
            x[target_j] += target_column.t().dot(&r) / target_column.norm_l2().powf(2.0);

            //update residual(r)
            r = y - mat.dot(&x);

            if r.norm_l2() < self.threshold {
                break;
            }
        }

        Ok(x)
    }
}
