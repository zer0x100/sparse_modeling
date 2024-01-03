pub use crate::prelude::*;

#[test]
fn bayesian_1sample_test() {
    std::env::set_var("RUST_BACKTRACE", "1");

    let mut rng = rand::thread_rng();
    let input_data: Array1<f64> =
        rand_pulses_signal(&mut rng, 50, 2, 1.0, 2.0).expect("can't generate signal");
    let matrix: Array2<f64> = ArrayBase::from_shape_fn((30, 50), |_| rng.gen_range(-1.0..1.0));
    let matrix = normalize_columns(&matrix).unwrap();
    let output_data = matrix.dot(&input_data);

    let threshold = 1e-5;
    let iter_num = 1000;
    let rvm = Rvm::new(threshold, iter_num);
    let rvm_result = rvm.solve(&matrix, &output_data).unwrap();

    //Draw ista_result, fista_result, and input_data
    //描画先をBackendとして指定。ここでは画像に出力するためBitMapBackend
    let root = BitMapBackend::new("results/bayesian_one_sample.png", (640, 480)).into_drawing_area();
    //背景を白に
    root.fill(&WHITE).unwrap();

    let max = input_data.norm_max();

    //グラフの軸設定など
    let mut chart = ChartBuilder::on(&root)
        .caption("bayesian_one_sample", ("sans-serif", 50).into_font())
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
        rvm_result.iter().enumerate().map(|(i, x)| (i, *x)),
        4,
        &GREEN,
    );
    chart.draw_series(point_series).unwrap();
}