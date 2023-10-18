use super::L1Relaxzation;
use crate::prelude::*;

pub struct L1Focuss {
    threshold: f64,
    iter_num: usize,
    by_bp: bool,
}

impl L1Focuss {
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

impl L1Relaxzation for L1Focuss {
    fn solve_l1(&self, mat: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
        match is_underestimated_sys(mat, y) {
            Err(msg) => return Err(msg),
            Ok(_) => (),
        }

        //initialization
        let mut x: Array1<f64> = Array::ones(mat.shape()[1]);
        let mut prev_x;

        let mut weights = x.clone();

        for _ in 0..self.iter_num {
            prev_x = x;
            let mut temp = mat.t().to_owned();
            for i in 0..weights.shape()[0] {
                for j in 0..temp.shape()[1] {
                    temp[[i, j]] *= weights[i];
                }
            }

            //update x
            x = temp.dot(
                &pseudo_inverse(
                    &mat.dot(&temp)
                ).unwrap()
                )
                .dot(y);

            //update weights
            for i in 0..x.shape()[0] {
                weights[i] = x[i].abs();
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