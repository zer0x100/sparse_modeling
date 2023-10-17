use super::L1Relaxzation;
use crate::prelude::*;

pub struct FOCUSS {
    threshold: f64,
    iter_num: usize,
    by_bp: bool,
}

impl FOCUSS {
    #[allow(dead_code)]
    pub fn new(threshold: f64, iter_num: usize, by_bp: bool) -> Self {
        Self { threshold, iter_num, by_bp }
    }
    #[allow(dead_code)]
    pub fn set(&mut self, threshold: f64, iter_num: usize, by_bp: bool) {
        self.threshold = threshold;
        self.iter_num = iter_num;
        self.by_bp = by_bp;
    }
}

impl L1Relaxzation for FOCUSS {
    fn solve_l1(&self, mat: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
        match is_underestimated_sys(mat, y) {
            Err(msg) => return Err(msg),
            Ok(_) => (),
        }

        //initialization
        let mut x: Array1<f64> = Array::ones(mat.shape()[1]);
        let mut prev_x;
        let mut weight_mat: Array2<f64> = ArrayBase::from_shape_fn(
            (mat.shape()[1], mat.shape()[1]),
            |(i, j)| {
                if i != j { 0.0 }
                else { 1. }
            }
        );

        for _ in 0..self.iter_num {
            prev_x = x;
            x = weight_mat
                .dot(&weight_mat)
                .dot(&mat.t())
                .dot(&pseudo_inverse(
                    &mat.dot(&weight_mat)
                        .dot(&weight_mat)
                        .dot(&mat.t())
                ).unwrap())
                .dot(y);
            for i in 0..x.shape()[0] {
                weight_mat[[i, i]] = x[i].abs().sqrt();
            }

            if (x.clone() - prev_x).norm_l2() < self.threshold {
                break;
            }
        }

        Ok(x)
    }

    fn by_basis_pursuit(&self) -> bool {
        self.by_bp
    }
}