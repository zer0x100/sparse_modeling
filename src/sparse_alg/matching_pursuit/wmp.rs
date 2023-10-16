use super::super::SparseAlg;
use crate::prelude::*;

pub struct Wmp {
    threshold: f64,
    iter_num: usize,
    proj_ratio: f64,
}

impl Wmp {
    #[allow(dead_code)]
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

    #[allow(dead_code)]
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
        match is_underestimated_sys(mat, y) {
            Err(msg) => return Err(msg),
            Ok(_) => (),
        }

        //initialization
        let mut x: Array1<f64> = Array::zeros(mat.shape()[1]);
        let mut r = y.clone();
        let mut support = HashSet::new();

        for _ in 0..self.iter_num {
            //rの射影が最初に閾値を超える列を探す
            let mut target_idx = 0;
            let mut max_proj = 0.;
            for j in 0..mat.shape()[1] {
                let column_j = mat.slice(s![.., j]);
                let proj = (column_j.t().dot(&r) / column_j.norm_l2()).abs();
                if proj >= self.proj_ratio * r.norm_l2() {
                    target_idx = j;
                    max_proj = proj;
                    break;
                }
                if max_proj < proj {
                    target_idx = j;
                    max_proj = proj;
                }
            }
            if max_proj.abs() < F64_ERR_RANGE {
                return Ok(x);
            }

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
