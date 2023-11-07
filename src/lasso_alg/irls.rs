use crate::prelude::*;

pub struct LassoIrls {
    iter_num: usize,
    threshold: f64,
}

impl LassoIrls {
    pub fn new(iter_num: usize, threshold: f64) -> Self {
        Self {
            iter_num,
            threshold,
        }
    }
    pub fn set(&mut self, iter_num: usize, threshold: f64) {
        self.iter_num = iter_num;
        self.threshold = threshold;
    }
}

impl LassoAlg for LassoIrls {
    fn solve(&self, mat: &Array2<f64>, y: &Array1<f64>, lambda: f64) -> Result<Array1<f64>> {
        //check data
        match is_underestimated_sys(mat, y) {
            Err(msg) => return Err(msg),
            Ok(_) => (),
        }

        //initialization
        let mut x: Array1<f64> = ArrayBase::ones(mat.shape()[1]);
        let mut prev_x;
        let mut weights = x.clone();

        for _ in 0..self.iter_num {
            prev_x = x;

            //update x

            //update weights

            if (x.clone() - prev_x).norm_l2() < self.threshold {
                break;
            }
        }

        Ok(x)
    }
}
