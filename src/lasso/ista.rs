use super::Lasso;
use crate::prelude::*;

pub struct LassoIsta {
    lambda: f64,
    iter_num: usize,
    threshold: f64,
}

impl LassoIsta {
    pub fn new(lambda: f64, iter_num: usize, threshold: f64) -> Self {
        Self {
            lambda,
            iter_num,
            threshold,
        }
    }
    pub fn set(&mut self, lambda: f64, iter_num: usize, threshold: f64) {
        self.lambda = lambda;
        self.iter_num = iter_num;
        self.threshold = threshold;
    }
}

impl Lasso for LassoIsta {
    fn solve(&self, mat: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
        //check data
        if mat.shape()[0] != y.shape()[0] {
            return Err(anyhow!("mat's row size and y's size are different."));
        }
        if mat.shape()[0] > mat.shape()[1] {
            return Err(anyhow!("mat's row size is bigger than its column size"));
        }

        //initialization
        let mut x = mat.t().dot(y);
        let mut prev_x;
        let lipshitz = matrix_l2(&(mat.t().dot(mat))) / self.lambda;

        for _ in 0..self.iter_num {
            prev_x = x.clone();
            let v = &x + 1. / lipshitz / self.lambda * mat.t().dot(&(y - mat.dot(&x)));
            x = st_array1(1. / lipshitz, &v);
            if (prev_x - x.clone()).norm_l2() < self.threshold {
                break;
            }
        }

        Ok(x)
    }
}