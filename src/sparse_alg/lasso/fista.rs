use super::super::SparseAlg;
use crate::prelude::*;

pub struct LassoFista {
    lambda: f64,
    iter_num: usize,
    threshold: f64,
}

impl LassoFista {
    #[allow(dead_code)]
    pub fn new(lambda: f64, iter_num: usize, threshold: f64) -> Self {
        Self {
            lambda,
            iter_num,
            threshold,
        }
    }
    #[allow(dead_code)]
    pub fn set(&mut self, lambda: f64, iter_num: usize, threshold: f64) {
        self.lambda = lambda;
        self.iter_num = iter_num;
        self.threshold = threshold;
    }
}

impl SparseAlg for LassoFista {
    fn solve(&self, mat: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
        //check data
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
        let mut x = mat.t().dot(y);
        let mut preve_x;
        let mut z = mat.t().dot(y);
        let mut prev_z;
        let lipshitz = matrix_l2(&mat.t().dot(mat)) / self.lambda;
        let mut beta = 0.;
        let mut prev_beta;

        for _ in 0..self.iter_num {
            preve_x = x.clone();
            let v = &z + 1. / lipshitz / self.lambda * mat.t().dot(&(y - mat.dot(&x)));
            x = st_array1(1. / lipshitz, &v);
            prev_beta = beta;
            beta = (1. + (1. + 4. * beta.powf(2.)).sqrt()) * 0.5;
            prev_z = z.clone();
            z = &x + (prev_beta - 1.) / beta * (&x - preve_x);

            if (&z - prev_z).norm_l2() < self.threshold {
                break;
            }
        }

        Ok(z)
    }
}
