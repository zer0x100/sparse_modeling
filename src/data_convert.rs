use crate::prelude::*;
use csv;

//データ変換関連
//csvデータを1次元配列に
pub fn csv_to_vec(file: fs::File) -> Vec<f64> {
    let mut reader = csv::Reader::from_reader(file);
    let mut arr : Vec<f64>= Vec::new();

    for record in reader.records() {
        let record = record.expect("failed to convert record");
        arr.push(record[0].parse().expect("failed to parse signal value"));
    }
    return arr;
}