use crate::prelude::*;

pub struct LassoIsta {
    iter_num: usize,
    threshold: f64,
}

impl LassoIsta {
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

impl LassoAlg for LassoIsta {
    fn solve(&self, mat: &Array2<f64>, y: &Array1<f64>, lambda: f64) -> Result<Array1<f64>> {
        //check data
        match is_underestimated_sys(mat, y) {
            Err(msg) => return Err(msg),
            Ok(_) => (),
        }

        //initialization
        let mut x = mat.t().dot(y);
        let mut prev_x;
        let lipshitz = matrix_l2(&(mat.t().dot(mat))) / lambda;

        for _ in 0..self.iter_num {
            prev_x = x.clone();
            let v = &x + 1. / lipshitz / lambda * mat.t().dot(&(y - mat.dot(&x)));
            x = st_array1(1. / lipshitz, &v);
            if (prev_x - x.clone()).norm_l2() < self.threshold {
                break;
            }
        }

        Ok(x)
    }
}
