use crate::prelude::*;

#[allow(dead_code)]
pub fn pulses_signal(size: usize, pulses: &[(usize, f64)]) -> Array1<f64> {
    let mut signal = Array::zeros(size);

    for pulse in pulses {
        signal[pulse.0] = pulse.1;
    }

    signal
}

#[allow(dead_code)]
pub fn rand_pulses_signal(size: usize, pulse_num: usize, min_abs: f64, max_abs: f64) -> Result<Array1<f64>> {
    if pulse_num > size { return Err(anyhow!(format!("pulse_num({}) is more than the size({})", pulse_num, size))); }

    let mut rng = rand::thread_rng();
    let uni_distribution = Uniform::from(0..size);

    let mut signal = Array::zeros(size);

    let mut filled_indexes = HashSet::new();
    while filled_indexes.len() < pulse_num {
        let fill_i = uni_distribution.sample(&mut rng);
        if !filled_indexes.contains(&fill_i) {
            signal[fill_i] = rng.gen_range(min_abs..max_abs);
            signal[fill_i] *= match rng.gen_range(0..2) {
                0 => 1.0,
                _ => -1.0,
            };
            filled_indexes.insert(fill_i);
        }
    }

    Ok(signal)
}
