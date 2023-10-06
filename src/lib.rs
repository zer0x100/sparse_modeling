mod dft;
mod gen_signal;
mod lasso;
mod math_func;
mod data_convert;

mod prelude {
    pub use crate::dft::*;
    pub use crate::gen_signal::*;
    pub use crate::math_func::*;
    pub use crate::data_convert::*;
    pub use crate::lasso::*;
    pub use anyhow::{anyhow, Result};
    pub use ndarray::{array, Array, Array1, Array2};
    pub use ndarray_linalg::{c64, generate, svd::SVD, Norm, Scalar};
    pub use rand::{
        distributions::{Distribution, Uniform},
        Rng,
    };
    pub use std::{f64::consts::PI, fs};
}

use crate::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use plotters::prelude::*;

    #[test]
    fn ista_test() {
        let input_data: Array1<f64> = Array::from(rand_sparse_signal(100, 9, 1.5));
        let matrix: Array2<f64> = generate::random((input_data.shape()[0] / 2, input_data.shape()[0]));
        let output_data = matrix.dot(&input_data);
        let lasso_ista = LassoIsta::new(1., 5000, 0.);
        let ista_result = lasso_ista.solve(&matrix, &output_data)
            .expect("can't solve lasso's ista");

        //Draw ista_result and input_data
        //描画先をBackendとして指定。ここでは画像に出力するためBitMapBackend
        let root = BitMapBackend::new("results/ista.png", (640, 480)).into_drawing_area();
        //背景を白に
        root.fill(&WHITE).unwrap();

        let max = input_data.norm_max();

        //グラフの軸設定など
        let mut chart = ChartBuilder::on(&root)
            .caption("ista_test", ("sans-serif", 50).into_font())
            .margin(10)//上下左右の余白
            .x_label_area_size(30)//x軸ラベル部分の余白
            .y_label_area_size(30)//y軸ラベル部分の余白
            .build_cartesian_2d(
                0..input_data.shape()[0],//x軸の設定
                -2.*max..2.*max//y軸の設定
            ).unwrap();
        
        //x軸y軸、グリッド線など描画
        chart.configure_mesh().draw().unwrap();
        //データの描画。(x, y)のイテレータとしてデータ点を渡す。
        let line_series = LineSeries::new(
            input_data.iter().enumerate().map(|(i, x)| (i, *x)),
            &RED,
        );
        chart.draw_series(line_series).unwrap();
        let line_series = LineSeries::new(
            ista_result.iter().enumerate().map(|(i, x)| (i, *x)),
            &BLUE,
        );
        chart.draw_series(line_series).unwrap();
    }
}
