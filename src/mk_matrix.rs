use crate::prelude::*;

//Wn_{i, j} = exp(-2*pi/n*i*j)
//FFTするわけではないのでサイズは2^(整数)じゃなくていい
pub fn mk_dft_mat(n: usize) -> Array2<c64> {
    let mut ft_matrix: Array2<c64> = ndarray_linalg::random((n, n));

    let theta = -2. * PI / n as f64;
    for i in 0..n {
        for j in 0..n {
            ft_matrix[[i, j]] = c64 {
                re: f64::cos(theta * i as f64 * j as f64),
                im: f64::sin(theta * i as f64 * j as f64),
            } / (n as f64).sqrt();
        }
    }

    ft_matrix
}

pub fn mk_idft_mat(n: usize) -> Array2<c64> {
    let matrix = mk_dft_mat(n);
    generate::conjugate(&matrix)
}
