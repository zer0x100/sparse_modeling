use crate::prelude::*;

pub struct LassoIrlsShrink {
    iter_num: usize,
    threshold: f64,
}

impl LassoIrlsShrink {
    #[allow(dead_code)]
    pub fn new(iter_num: usize, threshold: f64) -> Self {
        Self { iter_num, threshold }
    }
    #[allow(dead_code)]
    pub fn set(&mut self, iter_num: usize, threshold: f64) {
        self.iter_num = iter_num;
        self.threshold = threshold;
    }
}

impl LassoAlg for LassoIrlsShrink {
    fn solve(&self, mat: &Array2<f64>, y: &Array1<f64>, lambda: f64) -> Result<Array1<f64>> {
        //check data
        match is_underestimated_sys(mat, y) {
            Err(msg) => return Err(msg),
            Ok(_) => (),
        }

        //initialization
        let mut x: Array1<f64> = ArrayBase::ones(mat.shape()[1]);
        let mut prev_x;
        let mut r = y - mat.dot(&x);
        let (_, s, _) = mat.svd(false, false).unwrap();
        let c = s.norm_max().powf(2.0) / 2. + 1.;

        for _ in 0..self.iter_num {
            let e = mat.t().dot(&r);
            prev_x = x.clone();
            for i in 0..x.shape()[0] {
                let shrink = x[i].abs() / ( 2. * lambda / c +  x[i].abs() );
                x[i] = shrink * (x[i] + e[i] / c);
            }
            r = y - mat.dot(&x);

            if (prev_x - &x).norm_l2() < self.threshold {
                break;
            }
        }

        Ok(x)
    }
}