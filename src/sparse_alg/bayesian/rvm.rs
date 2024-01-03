use super::super::SparseAlg;
use crate::prelude::*;

pub struct Rvm {
    threshold: f64,
    iter_num: usize,
}

impl Rvm {
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

impl SparseAlg for Rvm {
    fn solve(&self, mat: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
        match is_underestimated_sys(mat, y) {
            Err(msg) => return Err(msg),
            Ok(_) => (),
        }

        //initialization
        let mut a: Array1<f64> = ArrayBase::ones(mat.shape()[1] + 1);
        let mut prev_a;
        let mut temp = a[0] * mat.t().dot(mat);
        for i in 0..mat.shape()[1] {
            temp[[i, i]] += a[i+1];
        }
        let mut cov = Inverse::inv(&temp).unwrap();
        let mut mu = a[0] * cov.dot(&mat.t()).dot(y);
        let mut gamma: Array1<f64> = ArrayBase::from_shape_fn(
            mat.shape()[1],
            |i| 1. - a[i+1] * cov[[i, i]],
        );

        //estimate hyperparameters
        for _ in 0..self.iter_num {
            prev_a = a.clone();
            a[0] = (mat.shape()[0] as f64 - gamma.sum()) / (y - mat.dot(&mu)).norm_l2().powf(2.);
            for i in 0..mat.shape()[1] {
                a[i+1] = gamma[i] / (mu[i]*mu[i]);
            }
            temp = a[0] * mat.t().dot(mat);
            for i in 0..mat.shape()[1] {
                temp[[i, i]] += a[i+1];
            }
            cov = Inverse::inv(&temp).unwrap();
            mu = a[0] * cov.dot(&mat.t()).dot(y);
            gamma = ArrayBase::from_shape_fn(
                mat.shape()[1],
                |i| 1. - a[i+1] * cov[[i, i]],
            );

            if (&a - &prev_a).norm_l2() < self.threshold {
                break;
            }
        }

        Ok(mu)
    }
}