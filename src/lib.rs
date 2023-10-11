mod data_convert;
mod gen_signal;
mod math_func;
mod mk_matrix;
mod sparse_alg;

mod prelude {
    pub use crate::data_convert::*;
    pub use crate::gen_signal::*;
    pub use crate::math_func::*;
    pub use crate::mk_matrix::*;
    pub use crate::sparse_alg::*;
    pub use anyhow::{anyhow, Result};
    pub use ndarray::prelude::*;
    pub use ndarray_linalg::{c64, generate, svd::SVD, Inverse, Norm, Scalar};
    pub use plotters::prelude::*;
    pub use rand::{
        distributions::{Distribution, Uniform},
        Rng,
    };
    pub use std::cmp;
    pub use std::collections::HashSet;
    pub use std::{f64::consts::PI, fs};
}

#[cfg(test)]
mod tests {

    use crate::prelude::*;

    #[test]
    fn lasso_test() {
        let input_data: Array1<f64> = rand_sparse_signal(100, 12, 1.5);
        let matrix: Array2<f64> =
            generate::random((input_data.shape()[0] / 2, input_data.shape()[0]));
        let output_data = matrix.dot(&input_data);

        let lambda = 1.;
        let iter_num = 200;
        let threshold = 0.;
        let lasso_ista = LassoIsta::new(lambda, iter_num, threshold);
        let ista_result = lasso_ista
            .solve(&matrix, &output_data)
            .expect("can't solve lasso's ista");
        let lasso_fista = LassoFista::new(lambda, iter_num, threshold);
        let fista_result = lasso_fista
            .solve(&matrix, &output_data)
            .expect("can't solve fista");

        //Draw ista_result, fista_result, and input_data
        //描画先をBackendとして指定。ここでは画像に出力するためBitMapBackend
        let root = BitMapBackend::new("results/lasso.png", (640, 480)).into_drawing_area();
        //背景を白に
        root.fill(&WHITE).unwrap();

        let max = input_data.norm_max();

        //グラフの軸設定など
        let mut chart = ChartBuilder::on(&root)
            .caption("lasso_test", ("sans-serif", 50).into_font())
            .margin(10) //上下左右の余白
            .x_label_area_size(30) //x軸ラベル部分の余白
            .y_label_area_size(30) //y軸ラベル部分の余白
            .build_cartesian_2d(
                0..input_data.shape()[0], //x軸の設定
                -1.5 * max..1.5 * max,    //y軸の設定
            )
            .unwrap();

        //x軸y軸、グリッド線など描画
        chart.configure_mesh().draw().unwrap();
        //データの描画。(x, y)のイテレータとしてデータ点を渡す。
        let point_series = PointSeries::<_, _, Circle<_, _>, _>::new(
            input_data.iter().enumerate().map(|(i, x)| (i, *x)),
            2,
            &RED,
        );
        chart.draw_series(point_series).unwrap();
        let point_series = PointSeries::<_, _, Circle<_, _>, _>::new(
            ista_result.iter().enumerate().map(|(i, x)| (i, *x)),
            2,
            &BLUE,
        );
        chart.draw_series(point_series).unwrap();
        let point_series = PointSeries::<_, _, Circle<_, _>, _>::new(
            fista_result.iter().enumerate().map(|(i, x)| (i, *x)),
            2,
            &GREEN,
        );
        chart.draw_series(point_series).unwrap();
    }

    #[test]
    fn mp_test() {
        std::env::set_var("RUST_BACKTRACE", "1");

        let input_data: Array1<f64> = rand_sparse_signal(100, 15, 1.5);
        let matrix: Array2<f64> =
            generate::random((input_data.shape()[0] / 2, input_data.shape()[0]));
        let output_data = matrix.dot(&input_data);

        let threshold = 0.01;
        let iter_num = 100;
        let wmp = Wmp::new(threshold, iter_num, 0.5).unwrap();
        let wmp_result = wmp.solve(&matrix, &output_data).unwrap();
        let mp = Mp::new(threshold, iter_num);
        let mp_result = mp.solve(&matrix, &output_data).unwrap();
        let omp = Omp::new(threshold);
        let omp_result = omp.solve(&matrix, &output_data).unwrap();
        let threshold_alg = ThresholdAlg::new(15);
        let threshold_result = threshold_alg.solve(&matrix, &output_data).unwrap();

        //Draw ista_result, fista_result, and input_data
        //描画先をBackendとして指定。ここでは画像に出力するためBitMapBackend
        let root = BitMapBackend::new("results/mp.png", (640, 480)).into_drawing_area();
        //背景を白に
        root.fill(&WHITE).unwrap();

        let max = input_data.norm_max();

        //グラフの軸設定など
        let mut chart = ChartBuilder::on(&root)
            .caption("mp_test", ("sans-serif", 50).into_font())
            .margin(10) //上下左右の余白
            .x_label_area_size(30) //x軸ラベル部分の余白
            .y_label_area_size(30) //y軸ラベル部分の余白
            .build_cartesian_2d(
                0..input_data.shape()[0], //x軸の設定
                -1.5 * max..1.5 * max,    //y軸の設定
            )
            .unwrap();

        //x軸y軸、グリッド線など描画
        chart.configure_mesh().draw().unwrap();
        //データの描画。(x, y)のイテレータとしてデータ点を渡す。
        let point_series = PointSeries::<_, _, Circle<_, _>, _>::new(
            input_data.iter().enumerate().map(|(i, x)| (i, *x)),
            4,
            &RED,
        );
        chart.draw_series(point_series).unwrap();
        let point_series = PointSeries::<_, _, Circle<_, _>, _>::new(
            mp_result.iter().enumerate().map(|(i, x)| (i, *x)),
            3,
            &BLUE,
        );
        chart.draw_series(point_series).unwrap();
        let point_series = PointSeries::<_, _, Circle<_, _>, _>::new(
            omp_result.iter().enumerate().map(|(i, x)| (i, *x)),
            2,
            &GREEN,
        );
        chart.draw_series(point_series).unwrap();
        let point_series = PointSeries::<_, _, Circle<_, _>, _>::new(
            wmp_result.iter().enumerate().map(|(i, x)| (i, *x)),
            2,
            &BLACK,
        );
        chart.draw_series(point_series).unwrap();
        let point_series = PointSeries::<_, _, Circle<_, _>, _>::new(
            threshold_result.iter().enumerate().map(|(i, x)| (i, *x)),
            7,
            &YELLOW,
        );
        chart.draw_series(point_series).unwrap();


    }

    #[test]
    fn math_func_test() {
        std::env::set_var("RUST_BACKTRACE", "1");

        let a = array![[1., 1., 1.], [1., 2., 3.],];
        assert_eq!(0.9899494936611665, mutal_coherence(&a));
        assert_eq!(0.9899494936611665, babel_func(&a, 1).unwrap());

        let arr = array![0., 1.];
        let arr2 = array![2., 3.];
        let arr3 = array![4., 5.];
        assert_eq!(
            columns_to_2darray(2, [arr, arr2, arr3].into_iter()).unwrap(),
            array![[0., 2., 4.], [1., 3., 5.],]
        );

        let a = array![[1., 3.], [1., 2.]];
        assert_eq!(pseudo_inverse(&a).unwrap(), a.inv().unwrap(),);
    }
}
