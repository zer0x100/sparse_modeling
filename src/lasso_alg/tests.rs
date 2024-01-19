use plotters::style::full_palette::{ORANGE, PURPLE};

use super::*;

#[test]
fn lasso_1sample_test() {
    let mut rng = rand::thread_rng();
    let input_data: Array1<f64> =
        rand_pulses_signal(&mut rng, 50, 3, 1.0, 2.0).expect("can't generate a signal");
    let matrix: Array2<f64> = ArrayBase::from_shape_fn((30, 50), |_| rng.gen_range(-1.0..1.0));
    let output_data = matrix.dot(&input_data);

    let lambda = 1e-2;
    let iter_num = 500;
    let threshold = 1e-20;
    let supp_err_range = 1e-2;
    let lasso_ista = LassoIsta::new(iter_num, threshold);
    let ista_result = lasso_ista
        .solve(&matrix, &output_data, lambda)
        .expect("can't solve lasso's ista");
    let lasso_fista = LassoFista::new(iter_num, threshold);
    let fista_result = lasso_fista
        .solve(&matrix, &output_data, lambda)
        .expect("can't solve fista");
    let irls = LassoIrls::new(iter_num, threshold, 1e-4);
    let irls_result = irls
        .solve(&matrix, &output_data, lambda)
        .expect("can't solve lasso's irls");
    let ssf = LassoSSF::new(iter_num, threshold);
    let ssf_result = ssf
        .solve(&matrix, &output_data, lambda)
        .expect("can't solve lasso's ssf");
    let irls_shrinkage = LassoIrlsShrink::new(iter_num, threshold);
    let irls_shrinkage_result = irls_shrinkage
        .solve(&matrix, &output_data, lambda)
        .expect("can't solve lasso's irls_shrinkage");

    //Draw ista_result, fista_result, and input_data
    //描画先をBackendとして指定。ここでは画像に出力するためBitMapBackend
    let root = BitMapBackend::new("results/lasso.png", (640, 480)).into_drawing_area();
    //背景を白に
    root.fill(&WHITE).unwrap();

    let max = input_data.norm_max();

    //グラフの軸設定など
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "supp_dist|| ista: {:.3} / fista: {:.3}",
                support_distance(&input_data, &ista_result, supp_err_range).unwrap(),
                support_distance(&input_data, &fista_result, supp_err_range).unwrap(),
            ),
            ("sans-serif", 50).into_font(),
        )
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
        ista_result.iter().enumerate().map(|(i, x)| (i, *x)),
        3,
        &BLUE,
    );
    chart.draw_series(point_series).unwrap();
    let point_series = PointSeries::<_, _, Circle<_, _>, _>::new(
        fista_result.iter().enumerate().map(|(i, x)| (i, *x)),
        2,
        &GREEN,
    );
    chart.draw_series(point_series).unwrap();
    let point_series = PointSeries::<_, _, Circle<_, _>, _>::new(
        irls_result.iter().enumerate().map(|(i, x)| (i, *x)),
        2,
        &BLACK,
    );
    chart.draw_series(point_series).unwrap();
    let point_series = PointSeries::<_, _, Circle<_, _>, _>::new(
        ssf_result.iter().enumerate().map(|(i, x)| (i, *x)),
        2,
        &ORANGE,
    );
    chart.draw_series(point_series).unwrap();
    let point_series = PointSeries::<_, _, Circle<_, _>, _>::new(
        irls_shrinkage_result.iter().enumerate().map(|(i, x)| (i, *x)),
        2,
        &PURPLE,
    );
    chart.draw_series(point_series).unwrap();
}
