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

#[allow(dead_code)]
pub fn mutal_coherence(mat: &Array2<f64>) -> f64 {
    let mut mat_sub = mat.clone();
    mat_sub = normalize_columns(&mat_sub).expect("can't normalize columns");
    let mut gram = mat_sub.t().dot(&mat_sub);
    for i in 0..gram.shape()[0] {
        gram[[i, i]] = 0.;
    }
    gram.norm_max()
}

#[allow(dead_code)]
pub fn babel_func(mat: &Array2<f64>, p: usize) -> Result<f64> {
    if mat.shape()[1] <= p {
        return Err(anyhow!("p is more than mat's column size"));
    }

    let mut matrix = mat.clone();
    matrix = normalize_columns(&matrix).expect("can't normalize columns");

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

#[allow(dead_code)]
pub fn normalize_columns(mat: &Array2<f64>) -> Result<Array2<f64>> {
    let mut result_mat = mat.clone();
    for i in 0..result_mat.shape()[1] {
        let column = result_mat.slice_mut(s![..;1, i]);
        let l2_norm = column.norm_l2();
        if l2_norm == 0.0 {
            return Err(anyhow!("0 column exits"));
        }
        for x in column {
            *x = *x / l2_norm;
        }
    }
    Ok(result_mat)
}

//least suqres method with limitation of support
pub fn lsm_with_support(
    mat: &Array2<f64>,
    y: &Array1<f64>,
    support: &HashSet<usize>,
) -> Result<Array1<f64>> {
    if mat.shape()[0] != y.shape()[0] {
        return Err(anyhow!("mat's row size and y's size are different"));
    }
    if support.is_empty() {
        return Err(anyhow!("support is empty."));
    }

    let mat_sub = columns_to_2darray(
        mat.shape()[0],
        support
            .clone()
            .into_iter()
            .filter(|i| *i < mat.shape()[1])
            .map(|i| mat.slice(s![.., i]).to_owned()),
    )
    .unwrap();

    let mut x = Array::zeros(mat.shape()[1]);
    let x_sub = pseudo_inverse(&mat_sub)
        .expect("can't compute pseudo inverse")
        .dot(y);
    support.iter().enumerate().for_each(|(sub_i, x_i)| {
        x[*x_i] = x_sub[sub_i];
    });

    Ok(x)
}

//integrate columns into 2d-array
pub fn columns_to_2darray<I: Iterator<Item = Array1<f64>>>(
    size: usize,
    columns: I,
) -> Result<Array2<f64>> {
    let mut result = Array::zeros((size, 0));

    columns.for_each(|column| {
        result
            .push_column(column.view())
            .expect("pushing columns failed");
    });

    Ok(result)
}

pub fn pseudo_inverse(mat: &Array2<f64>) -> Result<Array2<f64>> {
    if mat.shape()[0] < 1 || mat.shape()[1] < 1 {
        return Err(anyhow!("mat is empty(row size or column size is zeo."));
    }

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
    Ok(vt.t().dot(&sv_inverse.dot(&u.t())))
}

#[allow(dead_code)]
pub fn support_distance(vec1: &Array1<f64>, vec2: &Array1<f64>, err_range: f64) -> Result<f64> {
    if vec1.len() != vec2.len() {
        return Err(anyhow!(format!(
            "two vector sizes are different. vec1.len(): {}/ vec2.len(): {}",
            vec1.len(),
            vec2.len()
        )
        .to_string()));
    }
    let supp1 = support(&vec1, err_range);
    let supp2 = support(&vec2, err_range);
    let supp1_and_supp2: HashSet<usize> = supp1
        .iter()
        .filter(|i| supp2.contains(*i))
        .map(|i| *i)
        .collect();
    if supp1.len() == 0 && supp2.len() == 0 {
        return Ok(0.0);
    }
    Ok(1. - supp1_and_supp2.len() as f64 / cmp::max(supp1.len(), supp2.len()) as f64)
}

#[allow(dead_code)]
pub fn support(vec: &Array1<f64>, err_range: f64) -> HashSet<usize> {
    let mut supp = HashSet::new();
    vec.iter().enumerate().for_each(|(i, v)| {
        if v.abs() > err_range * 0.5 {
            supp.insert(i);
        }
    });
    supp
}

#[allow(dead_code)]
pub fn l2_relative_err(exact_x: &Array1<f64>, estimated_x: &Array1<f64>) -> Result<f64> {
    if exact_x.len() != estimated_x.len() {
        return Err(anyhow!(format!(
            "exact_x's size is {} / estimated_x's is {}",
            exact_x.len(),
            estimated_x.len()
        )
        .to_string()));
    }

    let exact_x_norm = exact_x.norm_l2();
    let diff_norm = (exact_x - estimated_x).norm_l2();
    if exact_x_norm == 0.0 {
        if diff_norm == 0.0 {
            return Ok(0.0);
        } else {
            return Ok(std::f64::MAX);
        }
    }
    Ok(diff_norm / exact_x_norm)
}

pub fn is_underestimated_sys(mat: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
    if mat.shape()[0] != y.shape()[0] || mat.shape()[0] > mat.shape()[1] {
        return Err(anyhow!(format!(
            "mat's shape is {}x{} / y's size is {}",
            mat.shape()[0],
            mat.shape()[1],
            y.shape()[0]
        )
        .to_string()));
    }
    Ok(())
}
