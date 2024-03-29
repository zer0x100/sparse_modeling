//! # Math Func
//! 
//! 'math_func' is a collection of mathamtic functions.
use crate::prelude::*;

///Soft theresholding function.
pub fn st(lambda: f64, x: f64) -> f64 {
    if x.abs() <= lambda {
        return 0.;
    }
    if x > lambda {
        return x - lambda;
    }
    x + lambda
}

///Soft thresholding function for Array1<f64>.
pub fn st_array1(lambda: f64, x: &Array1<f64>) -> Array1<f64> {
    let mut y = x.clone();
    y.iter_mut().for_each(|v| {
        *v = st(lambda, *v);
    });
    y
}

///Operator l2 norm for Array2<f64>.
pub fn matrix_l2(mat: &Array2<f64>) -> f64 {
    let (_, s, _) = mat.svd(false, false).unwrap();
    s.norm_max()
}

///Mutal coherence for Array2<f64>.
/// 
/// # Examples
/// 
/// ```
/// use ndarray::array;
/// 
///let a = array![[1., 1., 1.], [1., 2., 3.],];
///assert_eq!(0.9899494936611665, sparse_modeling::math_func::mutal_coherence(&a));
/// ```
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

///Babel function for Array2<f64>.
/// 
/// # Examples
/// 
/// ```
///use ndarray::array;
///let a = array![[1., 1., 1.], [1., 2., 3.],];
///assert_eq!(0.9899494936611665, sparse_modeling::math_func::babel_func(&a, 1).unwrap());
/// ```
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

///Normalize columns of mat(Array2<f64>).
/// 
/// # Examples
/// 
/// ```
/// use ndarray::array;
/// 
///let arr = array![[1., 2., 3.], [2., 5., 7.],];
///let arr = sparse_modeling::math_func::normalize_columns(&arr).unwrap();
///assert_eq!(
///    arr,
///    array![
///        [
///            1. / 5.0f64.powf(0.5),
///            2. / 29.0f64.powf(0.5),
///            3. / 58.0f64.powf(0.5)
///        ],
///        [
///            2. / 5.0f64.powf(0.5),
///            5. / 29.0f64.powf(0.5),
///            7. / 58.0f64.powf(0.5)
///        ],
///    ]
///);
/// ```
/// 
/// # Errors
/// if 0 column exists return Err.
#[allow(dead_code)]
pub fn normalize_columns(mat: &Array2<f64>) -> Result<Array2<f64>> {
    let mut result_mat = mat.clone();
    for i in 0..result_mat.shape()[1] {
        let column = result_mat.slice_mut(s![..;1, i]);
        let l2_norm = column.norm_l2();
        if l2_norm == 0.0 {
            for x in column {
                *x = 0.;
            }
        } else {
            for x in column {
                *x = *x / l2_norm;
        }
        }
    }
    Ok(result_mat)
}

///Least suqres method with limitation of support
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

///Integrate columns into 2d-array
/// 
/// # Examples
/// 
/// ```
/// use ndarray::array;
/// 
///let arr = array![0., 1.];
///let arr2 = array![2., 3.];
///let arr3 = array![4., 5.];
///assert_eq!(
///    sparse_modeling::math_func::columns_to_2darray(2, [arr, arr2, arr3].into_iter()).unwrap(),
///    array![[0., 2., 4.], [1., 3., 5.],]
///);
///``` 
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

///Output pseudo inverse of input Array2<f64>.
/// 
/// # Examples
/// 
/// ```
/// use ndarray::array;
/// use ndarray_linalg::{Norm, Inverse};
/// 
///let a = array![[1., 3.], [1., 2.]];
///assert!((sparse_modeling::math_func::pseudo_inverse(&a).unwrap() - a.inv().unwrap()).norm_l2() < 1e-8);
///```
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
        if s[i].abs() > F64_EPS {
            sv_inverse[[i, i]] = 1. / s[i];
        }
    }
    Ok(vt.t().dot(&sv_inverse.dot(&u.t())))
}

///Support distance
/// S1: vec1's support. S2: vec2's support.
/// return |(S1 & S2)| / max(|S1|, |S2|)
/// err_range is eps. if a is less than err_range. we treat a as zero.
/// 
/// # Examples
/// 
/// ```
/// use ndarray::array;
/// 
/// let a = array![1., 2., 0., 5., 0.];
/// let b = array![0., 3., 0., 4., 2.];
/// assert!((sparse_modeling::math_func::support_distance(&a, &b, 1e-10).unwrap() - 0.33333333333333).abs() < 1e-8);
///```
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

///Return Support of input Array1<f64>
///err_range is eps. if a is less than err_range. we treat a as zero.
/// 
/// # Examples
/// 
/// ```
/// use ndarray::array;
/// 
/// let a = array![1., 2., 0., 5., 0.];
/// let supp = sparse_modeling::math_func::support(&a, 1e-10);
/// assert!(
///     supp.contains(&0)
///         && supp.contains(&1)
///         && !supp.contains(&2)
///         && supp.contains(&3)
///         && !supp.contains(&4)
/// );
/// ```
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

///L2 relative error of two Array2<f64> inputs
/// return ||exact_x - estimated_x||_2 / ||exact_x||_2
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

/// judge whether (mat, y) is under estimated system
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

