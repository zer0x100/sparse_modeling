mod data_convert;
mod gen_signal;
mod lasso_alg;
mod math_func;
mod mk_matrix;
mod sparse_alg;

mod prelude {
    pub use crate::data_convert::*;
    pub use crate::gen_signal::*;
    pub use crate::lasso_alg::*;
    pub use crate::math_func::*;
    pub use crate::mk_matrix::*;
    pub use crate::sparse_alg::*;
    pub use anyhow::{anyhow, Result};
    pub use ndarray::prelude::*;
    pub use ndarray_linalg::{c64, generate, svd::SVD, Inverse, Norm, Scalar};
    pub use plotters::prelude::*;
    pub use rand::{
        distributions::{Distribution, Uniform},
        rngs::ThreadRng,
        Rng,
    };
    pub use std::cmp;
    pub use std::collections::HashSet;
    pub use std::{f64::consts::PI, fs};
    pub const F64_ERR_RANGE: f64 = 1e-8;
}

#[cfg(test)]
mod tests {

    use crate::prelude::*;

    #[test]
    fn lasso_test() {
        let mut rng = rand::thread_rng();
        let input_data: Array1<f64> =
            rand_pulses_signal(&mut rng, 100, 12, 1.0, 2.0).expect("can't generate a signal");
        let matrix: Array2<f64> = ArrayBase::from_shape_fn((50, 100), |_| rng.gen_range(-1.0..1.0));
        let output_data = matrix.dot(&input_data);

        let lambda = 1.;
        let iter_num = 200;
        let threshold = 0.;
        let lasso_ista = LassoIsta::new(iter_num, threshold);
        let ista_result = lasso_ista
            .solve(&matrix, &output_data, lambda)
            .expect("can't solve lasso's ista");
        let lasso_fista = LassoFista::new(iter_num, threshold);
        let fista_result = lasso_fista
            .solve(&matrix, &output_data, lambda)
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
    fn mp_1sample_test() {
        std::env::set_var("RUST_BACKTRACE", "1");

        let mut rng = rand::thread_rng();
        let input_data: Array1<f64> =
            rand_pulses_signal(&mut rng, 50, 5, 1.0, 2.0).expect("can't generate signal");
        let matrix: Array2<f64> = ArrayBase::from_shape_fn((30, 50), |_| rng.gen_range(-1.0..1.0));
        let matrix = normalize_columns(&matrix).unwrap();
        let output_data = matrix.dot(&input_data);

        let threshold = 1e-2;
        let iter_num = 10000;
        let threshold_alg = ThresholdAlg::new(15);
        let threshold_result = threshold_alg.solve(&matrix, &output_data).unwrap();
        let wmp = Wmp::new(threshold, iter_num, 0.5).unwrap();
        let wmp_result = wmp.solve(&matrix, &output_data).unwrap();
        let mp = Mp::new(threshold, iter_num);
        let mp_result = mp.solve(&matrix, &output_data).unwrap();
        let omp = Omp::new(threshold);
        let omp_result = omp.solve(&matrix, &output_data).unwrap();

        println!(
            "supp_dist|| threshold: {}, wmp: {}, mp: {}, omp: {}",
            support_distance(&input_data, &threshold_result).unwrap(),
            support_distance(&input_data, &wmp_result).unwrap(),
            support_distance(&input_data, &mp_result).unwrap(),
            support_distance(&input_data, &omp_result).unwrap(),
        );
        println!(
            "l2_relative_err|| threshold: {}, wmp: {}, mp: {}, omp: {}",
            l2_relative_err(&input_data, &threshold_result)
                .unwrap()
                .powf(2.0),
            l2_relative_err(&input_data, &wmp_result).unwrap().powf(2.0),
            l2_relative_err(&input_data, &mp_result).unwrap().powf(2.0),
            l2_relative_err(&input_data, &omp_result).unwrap().powf(2.0),
        );

        //Draw ista_result, fista_result, and input_data
        //描画先をBackendとして指定。ここでは画像に出力するためBitMapBackend
        let root = BitMapBackend::new("results/mp_one_sample.png", (640, 480)).into_drawing_area();
        //背景を白に
        root.fill(&WHITE).unwrap();

        let max = input_data.norm_max();

        //グラフの軸設定など
        let mut chart = ChartBuilder::on(&root)
            .caption("mp_one_sample", ("sans-serif", 50).into_font())
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
    fn compare_mps_test() {
        //set parameters
        let threshold = 1e-2;
        let sample_size = 200;
        let iter_num = 10000;
        let matrix_shape = (30, 50);
        let supp_sizes_range = 1..11;
        let pulse_value_range = (1.0 /* min */, 2.0 /* max */); //絶対値
                                                                //set algorithms
        let mut threshold_alg = ThresholdAlg::new(10);
        let wmp = Wmp::new(threshold, iter_num, 0.5)
            .expect("failed to create wmp(weak matching pursuit)");
        let mp = Mp::new(threshold, iter_num);
        let omp = Omp::new(threshold);

        //ThresholdAlg, Wmp, Mp, Ompの順で結果を格納
        let mut supp_dist_list = Vec::<(usize, [f64; 4])>::new();
        let mut l2_err_list = Vec::<(usize, [f64; 4])>::new();

        println!("calucalating mps...");
        let mut rng = rand::thread_rng();
        for support_size in supp_sizes_range.clone() {
            println!(
                "signals, whose support sizes are {}, are generated and test mps",
                support_size
            );
            let mut supp_dist = (support_size, [0.; 4]);
            let mut l2_err = (support_size, [0.; 4]);
            for it in 0..sample_size {
                if it % 100 == 0 {
                    println!("support size {}/sample num {}", support_size, it);
                }
                let matrix: Array2<f64> =
                    ArrayBase::from_shape_fn(matrix_shape, |_| rng.gen_range(-1.0..1.0));
                let matrix = normalize_columns(&matrix).expect("can't normalize matrix");
                let input_signal = rand_pulses_signal(
                    &mut rng,
                    matrix_shape.1,
                    support_size,
                    pulse_value_range.0,
                    pulse_value_range.1,
                )
                .expect("failed to generate a signal");
                let output_signal = matrix.dot(&input_signal);

                threshold_alg.set(support_size);
                let threshold_alg_result = threshold_alg
                    .solve(&matrix, &output_signal)
                    .expect("ThresholdAlg failed");
                let wmp_result = wmp.solve(&matrix, &output_signal).expect("wmp failed");
                let mp_result = mp
                    .solve(&matrix, &output_signal)
                    .expect("mp(matching pursuit) failed");
                let omp_result = omp
                    .solve(&matrix, &output_signal)
                    .expect("omp(orthogonal matching pursuit) failed");

                supp_dist.1[0] += support_distance(&input_signal, &threshold_alg_result)
                    .expect("failed to compute support distace");
                supp_dist.1[1] += support_distance(&input_signal, &wmp_result)
                    .expect("failed to compute support distace");
                supp_dist.1[2] += support_distance(&input_signal, &mp_result)
                    .expect("failed to compute support distace");
                supp_dist.1[3] += support_distance(&input_signal, &omp_result)
                    .expect("failed to compute support distace");

                l2_err.1[0] += l2_relative_err(&input_signal, &threshold_alg_result)
                    .expect("can't calucalate l2 error")
                    .powf(2.0);
                l2_err.1[1] += l2_relative_err(&input_signal, &wmp_result)
                    .expect("can't calucalate l2 error")
                    .powf(2.0);
                l2_err.1[2] += l2_relative_err(&input_signal, &mp_result)
                    .expect("can't calucalate l2 error")
                    .powf(2.0);
                l2_err.1[3] += l2_relative_err(&input_signal, &omp_result)
                    .expect("can't calucalate l2 error")
                    .powf(2.0);
            }
            supp_dist.1.iter_mut().for_each(|dist| {
                *dist = *dist / sample_size as f64;
            });
            l2_err.1.iter_mut().for_each(|dist| {
                *dist = *dist / sample_size as f64;
            });
            supp_dist_list.push(supp_dist);
            l2_err_list.push(l2_err);
        }
        println!("finished calucalating mps");

        //Plot
        //support distance
        let root = BitMapBackend::new("results/mp_supp_dist.png", (640, 480)).into_drawing_area();
        //背景を白に
        root.fill(&WHITE).unwrap();

        let max = 1.;

        //グラフの軸設定など
        let mut chart = ChartBuilder::on(&root)
            .caption(
                "mp_test/x: pulse num/y: support distance",
                ("sans-serif", 20).into_font(),
            )
            .margin(10) //上下左右の余白
            .x_label_area_size(30) //x軸ラベル部分の余白
            .y_label_area_size(30) //y軸ラベル部分の余白
            .build_cartesian_2d(
                supp_sizes_range.clone(), //x軸の設定
                0.0..max,                 //y軸の設定
            )
            .unwrap();

        //x軸y軸、グリッド線など描画
        chart.configure_mesh().draw().unwrap();
        //データの描画。(x, y)のイテレータとしてデータ点を渡す。
        let line_series = LineSeries::new(
            supp_dist_list
                .iter()
                .map(|(supp_size, dist)| (*supp_size, dist[0])),
            &RED,
        );
        chart.draw_series(line_series).unwrap();
        let line_series = LineSeries::new(
            supp_dist_list
                .iter()
                .map(|(supp_size, dist)| (*supp_size, dist[1])),
            &BLUE,
        );
        chart.draw_series(line_series).unwrap();
        let line_series = LineSeries::new(
            supp_dist_list
                .iter()
                .map(|(supp_size, dist)| (*supp_size, dist[2])),
            &GREEN,
        );
        chart.draw_series(line_series).unwrap();
        let line_series = LineSeries::new(
            supp_dist_list
                .iter()
                .map(|(supp_size, dist)| (*supp_size, dist[3])),
            &BLACK,
        );
        chart.draw_series(line_series).unwrap();

        //l2 relative error
        let root = BitMapBackend::new("results/mp_l2_rerr.png", (640, 480)).into_drawing_area();
        //背景を白に
        root.fill(&WHITE).unwrap();

        let max = 1.;

        //グラフの軸設定など
        let mut chart = ChartBuilder::on(&root)
            .caption(
                "mp_test/x: pulse num/y: l2 relative error",
                ("sans-serif", 20).into_font(),
            )
            .margin(10) //上下左右の余白
            .x_label_area_size(30) //x軸ラベル部分の余白
            .y_label_area_size(30) //y軸ラベル部分の余白
            .build_cartesian_2d(
                supp_sizes_range, //x軸の設定
                0.0..max,         //y軸の設定
            )
            .unwrap();

        //x軸y軸、グリッド線など描画
        chart.configure_mesh().draw().unwrap();
        //データの描画。(x, y)のイテレータとしてデータ点を渡す。
        let line_series = LineSeries::new(
            l2_err_list
                .iter()
                .map(|(supp_size, dist)| (*supp_size, dist[0])),
            &RED,
        );
        chart.draw_series(line_series).unwrap();
        let line_series = LineSeries::new(
            l2_err_list
                .iter()
                .map(|(supp_size, dist)| (*supp_size, dist[1])),
            &BLUE,
        );
        chart.draw_series(line_series).unwrap();
        let line_series = LineSeries::new(
            l2_err_list
                .iter()
                .map(|(supp_size, dist)| (*supp_size, dist[2])),
            &GREEN,
        );
        chart.draw_series(line_series).unwrap();
        let line_series = LineSeries::new(
            l2_err_list
                .iter()
                .map(|(supp_size, dist)| (*supp_size, dist[3])),
            &BLACK,
        );
        chart.draw_series(line_series).unwrap();
    }

    #[test]
    fn l1_relaxzation_test() {
        let mut rng = rand::thread_rng();
        let input_data = rand_pulses_signal(&mut rng, 50, 5, 1.0, 2.0).unwrap();
        let matrix = ArrayBase::from_shape_fn((30, 50), |_| rng.gen_range(-1.0..1.0));
        let output_data = matrix.dot(&input_data);

        let lasso_lambda = 1e-2;
        let iter_num = 1000;
        let bs_threshold = 1e-2;
        let sparse_alg_lasso = SparseAlgLasso::new(lasso_lambda, Box::new(LassoFista::new(iter_num, bs_threshold)));
        let result = sparse_alg_lasso.solve(&matrix, &output_data).unwrap();

        //Draw ista_result, fista_result, and input_data
        //描画先をBackendとして指定。ここでは画像に出力するためBitMapBackend
        let root = BitMapBackend::new("results/l1_relaxzation.png", (640, 480)).into_drawing_area();
        //背景を白に
        root.fill(&WHITE).unwrap();

        let max = input_data.norm_max();

        //グラフの軸設定など
        let mut chart = ChartBuilder::on(&root)
            .caption("l1_relaxzation_test", ("sans-serif", 50).into_font())
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
            result.iter().enumerate().map(|(i, x)| (i, *x)),
            2,
            &BLUE,
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

        let arr = array![[1., 2., 3.], [2., 5., 7.],];
        let arr = normalize_columns(&arr).unwrap();
        assert_eq!(
            arr,
            array![
                [
                    1. / 5.0f64.powf(0.5),
                    2. / 29.0f64.powf(0.5),
                    3. / 58.0f64.powf(0.5)
                ],
                [
                    2. / 5.0f64.powf(0.5),
                    5. / 29.0f64.powf(0.5),
                    7. / 58.0f64.powf(0.5)
                ],
            ]
        );

        let a = array![[1., 3.], [1., 2.]];
        assert!((pseudo_inverse(&a).unwrap() - a.inv().unwrap()).norm_l2() < F64_ERR_RANGE);

        let a = array![1., 2., 0., 5., 0.];
        let supp = support(&a);
        assert!(
            supp.contains(&0)
                && supp.contains(&1)
                && !supp.contains(&2)
                && supp.contains(&3)
                && !supp.contains(&4)
        );
        let b = array![0., 3., 0., 4., 2.];
        assert!((support_distance(&a, &b).unwrap() - 0.33333333333333).abs() < F64_ERR_RANGE);
    }
}
