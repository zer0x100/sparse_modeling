
use ndarray_linalg::Scalar;

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

pub fn mutal_coherence(mat: &Array2<f64>) -> f64 {
    let mut mat_sub = mat.clone();
    normalize_columns(&mut mat_sub);
    let mut gram = mat_sub.t().dot(&mat_sub);
    for i in 0..gram.shape()[0] {
        gram[[i, i]] = 0.;
    }
    gram.norm_max()
}

pub fn babel_func(mat: &Array2<f64>, p: usize) -> Result<f64> {
    if mat.shape()[1] <= p {
        return Err(anyhow!("p is more than mat's column size"));
    }

    let mut matrix = mat.clone();
    normalize_columns(&mut matrix);

    let gram = matrix.t().dot(&matrix);

    let mut max = 0.;
    for i in 0..matrix.shape()[1] {
        let mut vec: Vec<f64> = gram.slice(s![i, ..;1]).to_vec();
        vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut sum = 0.;
        for j in 1..=p {
            sum += vec[j];
        }
        max = if max >= sum { max } else { sum };
    }

    Ok(max)
}

pub fn normalize_columns(mat: &mut Array2<f64>) {
    for i in 0..mat.shape()[1] {
        let columns = mat.slice_mut(s![..;1, i]);
        let norm = columns.norm_l2();
        for x in columns {
            *x = *x / norm;
        }
    }
}

//least suqres method with limitation of support
pub fn lsm_with_support(mat: &Array2<f64>, y: &Array1<f64>, support: &HashSet<usize>) -> Result<Array1<f64>> {
    if mat.shape()[0] != y.shape()[0] {
        return Err(anyhow!("mat's column size and y's size are different"));
    }

    let mat_sub = columns_to_2darray(
        mat.shape()[0],
        support.clone()
            .into_iter()
            .map(|i| {
                mat.slice(s![.., i]).to_owned()
            }
        )
    ).unwrap();

    let x_sub = pseudo_inverse(&mat_sub).dot(y);

    let mut x = Array::zeros(mat.shape()[1]);
    support.iter()
        .enumerate()
        .for_each(|(sub_i, x_i)| {
            x[*x_i] = x_sub[sub_i];
        }
    );

    Ok(x)
}

//integrate columns into 2d-array
pub fn columns_to_2darray<I: Iterator<Item=Array1<f64>>>(size: usize, columns: I) -> Result<Array2<f64>>{
    let mut result = Array::zeros((size, 0));

    columns.for_each(|column| {
        result.push_column(column.view()).expect("pushing columns failed");
    });

    Ok(result)
}

pub fn pseudo_inverse(mat: &Array2<f64>) -> Array2<f64> {
    let (u, s, vt) = mat.svd(true, true).unwrap();
    let u = u.unwrap();
    let vt = vt.unwrap();
    
    let mut sv_inverse = Array::zeros((mat.shape()[1], mat.shape()[0]));
    let sv_size = cmp::min(mat.shape()[0], mat.shape()[1]);
    for i in 0..sv_size {
        if s[i] != 0. {
            sv_inverse[[i, i]] = 1. / s[i];
        }
    }
    vt.t().dot(&sv_inverse.dot(&u.t()))
}