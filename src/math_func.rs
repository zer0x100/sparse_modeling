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
