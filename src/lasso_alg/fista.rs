use crate::prelude::*;

pub struct LassoFista {
    iter_num: usize,
    threshold: f64,
    lipshitz: Option<f64>,
}

impl LassoFista {
    #[allow(dead_code)]
    pub fn new(iter_num: usize, threshold: f64) -> Self {
        Self {
            iter_num,
            threshold,
            lipshitz: None,
        }
    }
    #[allow(dead_code)]
    pub fn set(&mut self, iter_num: usize, threshold: f64) {
        self.iter_num = iter_num;
        self.threshold = threshold;
    }
    #[allow(dead_code)]
    pub fn set_lipshitz(&mut self, lipshitz: f64) {
        self.lipshitz = Some(lipshitz)
    }
    #[allow(dead_code)]
    pub fn lipshitz_to_none(&mut self) {
        self.lipshitz = None;
    }
}

impl LassoAlg for LassoFista {
    fn solve(&self, mat: &Array2<f64>, y: &Array1<f64>, lambda: f64) -> Result<Array1<f64>> {
        //check data
        match is_underestimated_sys(mat, y) {
            Err(msg) => return Err(msg),
            Ok(_) => (),
        }

        //initialization
        let mut x = mat.t().dot(y);
        let mut prev_x;
        let mut z = mat.t().dot(y);
        let mut prev_z;
        let lipshitz = if let Some(lip) = self.lipshitz {
            lip
        } else {
            let (_, mut s, _) = mat.svd(false, false).unwrap();
            s.iter_mut().for_each(|v| *v = *v * *v);
            s.norm_max() / lambda
        };

        let mut beta = 0.;
        let mut prev_beta;

        for _ in 0..self.iter_num {
            prev_x = x.clone();
            let v = &z + 1. / lipshitz / lambda * mat.t().dot(&(y - mat.dot(&x)));
            x = st_array1(1. / lipshitz, &v);
            prev_beta = beta;
            beta = (1. + (1. + 4. * beta.powf(2.)).sqrt()) * 0.5;
            prev_z = z.clone();
            z = &x + (prev_beta - 1.) / beta * (&x - prev_x);

            if (&z - prev_z).norm_l2() < self.threshold {
                break;
            }
        }

        Ok(z)
    }
}
