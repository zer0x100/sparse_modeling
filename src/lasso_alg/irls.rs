use crate::prelude::*;

pub struct LassoIrls {
    iter_num: usize,
    threshold: f64,
}

impl LassoIrls {
    #[allow(dead_code)]
    pub fn new(iter_num: usize, threshold: f64) -> Self {
        Self {
            iter_num,
            threshold,
        }
    }
    #[allow(dead_code)]
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
        let e = 1e-2;

        for _ in 0..self.iter_num {
            //update x, weights
            let mut temp = mat.t().dot(mat);
            for i in 0..x.shape()[0] {
                temp[[i, i]] += 2. * lambda / weights[i];
            }
            prev_x = x;
            x = conjugate_gradient(&temp, &mat.t().dot(y), 15, 0.).unwrap();
            for i in 0..x.shape()[0] {
                weights[i] = x[i].abs() + e;
            }

            if (x.clone() - prev_x).norm_l2() < self.threshold {
                break;
            }
        }

        Ok(x)
    }
}
