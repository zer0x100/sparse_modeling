use crate::prelude::*;

pub fn sparse_signal(size: usize, pulses: &[(usize, f64)]) -> Array1<f64> {
    let mut signal = Array::zeros(size);

    for pulse in pulses {
        signal[pulse.0] = pulse.1;
    }

    signal
}

pub fn rand_sparse_signal(size: usize, pulse_num: usize, max_value: f64) -> Array1<f64> {
    let mut rng = rand::thread_rng();
    let uni_distribution = Uniform::from(0..size);

    let mut signal = Array::zeros(size);

    for _ in 0..pulse_num {
        signal[uni_distribution.sample(&mut rng)] = (rng.gen::<f64>() - 0.5) * 2. * max_value;
    }

    signal
}
