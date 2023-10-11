use super::super::SparseAlg;
use crate::prelude::*;

pub struct LassoIstaLipshitzSearch {
    lambda: f64,
    iter_num: usize,
    threshold: f64,
}

impl LassoIstaLipshitzSearch {
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

impl SparseAlg for LassoIstaLipshitzSearch {
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
        let mut prev_x;
        let mut lipshitz = 1.;

        for _ in 0..self.iter_num {
            prev_x = x.clone();
            //二次関数部分の前回の値
            let prev_temp = 0.5 * (y - mat.dot(&x)).norm_l2().powi(2);
            //prev_xでの二次関数の勾配計算
            let grad_x = -1. * mat.t().dot(&(y - mat.dot(&x)));
            //二次関数部分のメジャライザーを最小化する点(lipshitz定数が正しいならメジャライザーが定まる)
            let mut v = &x - &grad_x / lipshitz;

            //vでの二次関数の値
            let mut temp = 0.5 * (y - mat.dot(&v)).norm_l2().powi(2);
            //vでのprev_x起点のメジャライザーの値
            let mut m_temp = prev_temp
                + grad_x.t().dot(&(&v - &prev_x))
                + 0.5 * lipshitz * (&v - &prev_x).norm_l2().powi(2);
            //二次関数部分のメジャライザーが二次関数より小さいことはないのでもしそうなったらlipshitzを大きな値に更新
            while m_temp < temp {
                lipshitz = lipshitz * 1.1;

                //更新候補位置を新たなlipshitzの元で計算
                v = &x - &grad_x / lipshitz;

                //その地点での二次関数と目じゃライザーの値を計算
                temp = 0.5 * (y - mat.dot(&v)).norm_l2().powi(2);
                m_temp = prev_temp
                    + grad_x.t().dot(&(&v - &prev_x))
                    + 0.5 * lipshitz * (&v - &prev_x).norm_l2().powi(2);
            }

            x = st_array1(self.lambda / lipshitz, &v);
            if (prev_x - x.clone()).norm_l2() < self.threshold {
                break;
            }
        }

        Ok(x)
    }
}
