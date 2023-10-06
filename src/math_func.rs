use crate::prelude::*;

//soft theresholding function
pub fn st(lambda: f64, x: f64) -> f64 {
    if x.abs() <= lambda {
        return 0.;
    }
    if x > lambda {
        return x - lambda;
    }
    x + lambda
}

//st for Array1
pub fn st_array1(lambda: f64, x: &Array1<f64>) -> Array1<f64> {
    let mut y = x.clone();
    y.iter_mut().for_each(|v| {
        *v = st(lambda, *v);
    });
    y
}

pub fn matrix_l2(mat: &Array2<f64>) -> f64 {
    let (_, s, _) = mat.svd(false, false).unwrap();
    s.norm_max()
}
