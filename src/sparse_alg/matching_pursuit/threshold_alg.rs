use super::super::SparseAlg;
use crate::prelude::*;

pub struct ThresholdAlg {
    support_size: usize,
}

impl ThresholdAlg {
    pub fn new(support_size: usize) -> Self {
        Self { support_size }
    }
    pub fn set(&mut self, support_size: usize) {
        self.support_size = support_size;
    }
}

impl SparseAlg for ThresholdAlg {
    fn solve(&self, mat: &Array2<f64>, y: &Array1<f64>) -> Result<Array1<f64>> {
        if mat.shape()[0] != y.shape()[0] || mat.shape()[0] > mat.shape()[1] {
            return Err(anyhow!(format!(
                "mat's shape is {}x{} / y's size is {}",
                mat.shape()[0],
                mat.shape()[1],
                y.shape()[0]
            )
            .to_string()));
        }

        //matの列番号と、その列方向へのyの射影の絶対値をペアにして降順に並べる
        let mut proj_list: Vec<(usize, f64)> = normalize_columns(mat).expect("can't normalize mat").t().dot(y)
            .iter()
            .map(|v| v.abs())
            .enumerate()
            .collect();
        proj_list.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        proj_list.reverse();

        let mut support = HashSet::new();
        for i in 0..cmp::min(self.support_size, mat.shape()[1]) {
            support.insert(proj_list[i].0);
        }
        let x = lsm_with_support(mat, y, &support).unwrap();

        Ok(x)
    }
}
