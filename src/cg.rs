use crate::prelude::*;

pub fn conjugate_gradient(
    mat: &Array2<f64>,
    y: &Array1<f64>,
    iter_num: usize,
    threshold: f64,
) -> Result<Array1<f64>> {
    //check whether mat is symmetric
    let mut is_symmetric = true;
    if mat.shape() != [y.shape()[0], y.shape()[0]] {
        is_symmetric = false;
    } else {
        for i in 0..y.shape()[0] {
            for j in 0..y.shape()[0] {
                if mat[[i, j]] != mat[[j, i]] {
                    is_symmetric = false;
                }
            }
        }
    }
    if !is_symmetric {
        return Err(anyhow!("mat is not symmetric"));
    }

    //initialization
    let mut x: Array1<f64> = ArrayBase::zeros(y.shape()[0]);
    let mut r = y.clone();
    let mut prev_r;
    let mut d = r.clone();

    for _ in 0..iter_num {
        let a = d.dot(&r) / d.dot(&mat.dot(&d));
        x = x + a * d.clone();
        prev_r = r.clone();
        r = r - a * mat.dot(&d);

        if r.norm_l2() / y.norm_l2() < threshold {
            break;
        }

        let b = r.dot(&r) / prev_r.dot(&prev_r);
        d = &r + b * d;
    }

    Ok(x)
}

#[test]
fn cgtest() {}
