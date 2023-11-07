use crate::prelude::*;

pub fn conjugate_gradient(mat: &Array2<f64>, y: &Array1<f64>, threshold: f64) -> Result<Array1<f64>> {
    //check whether mat is symmetric
    let mut is_symmetric = true;
    if mat.shape() != [y.shape()[0], y.shape()[0]] { is_symmetric = false; }
    else {
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
    let mut p = r.clone();
    let mat_dot_y = mat.dot(y);

    for itr in 0..y.shape()[0] {

        if (&x - &mat_dot_y).norm_l2() < threshold {
            break;
        }
        if itr == y.shape()[0] - 1 {
            return Err(anyhow!("CG failed to reduce error lower than threshold"));
        }
    }

    Ok(x)
}