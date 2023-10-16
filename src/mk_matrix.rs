use crate::prelude::*;

//Wn_{i, j} = exp(-2*pi/n*i*j)
//FFTするわけではないのでサイズは2^(整数)じゃなくていい
#[allow(dead_code)]
pub fn mk_dft_mat(n: usize) -> Array2<c64> {
    let theta = -2. * PI / n as f64;
    ArrayBase::from_shape_fn((n, n), |(i, j)| c64 {
        re: f64::cos(theta * i as f64 * j as f64),
        im: f64::sin(theta * i as f64 * j as f64),
    })
}

#[allow(dead_code)]
pub fn mk_idft_mat(n: usize) -> Array2<c64> {
    let matrix = mk_dft_mat(n);
    generate::conjugate(&matrix)
}
