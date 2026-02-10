use anyhow::{Context, Result};
use freeverb::Freeverb;
use glob::glob;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::sync::atomic::{AtomicUsize, Ordering};
use wavers::{Wav, write};

const SEED: u64 = 42;

#[derive(Debug, Clone, Copy)]
struct SplitFractions {
    train: f64,
    val: f64,
    test: f64,
}

fn train_test_split<T>(mut items: Vec<T>, test_size: f64, seed: u64) -> Result<(Vec<T>, Vec<T>)> {
    anyhow::ensure!(
        (0.0..=1.0).contains(&test_size),
        "test_size must be in [0, 1]"
    );

    let mut rng = StdRng::seed_from_u64(seed);
    items.shuffle(&mut rng);

    let n = items.len();
    let n_test = (((n as f64) * test_size).ceil() as usize).min(n);
    let split_idx = n - n_test;

    let test = items.split_off(split_idx);
    Ok((items, test))
}

fn per_item_seed(base_seed: u64, idx: usize) -> u64 {
    base_seed ^ (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((idx as u64).rotate_left(17))
}

fn read_wav(path: &str) -> Result<Wav<f32>> {
    let wav: Wav<f32> = Wav::from_path(path).context("failed to read wav")?;
    Ok(wav)
}

fn reverberate(wav: &mut Wav<f32>, output_path: &str, room_size: f32, wet: f32) -> Result<()> {
    let sr = wav.sample_rate();
    let mut verb: Freeverb<f32> = Freeverb::new(sr as usize);

    verb.set_dry(1.);
    verb.set_wet(wet);
    verb.set_room_size(room_size);

    let count = wav.frames().count();
    let mut output_samples: Vec<f32> = Vec::with_capacity(count);
    for frame in wav.frames() {
        let right = if frame.len() > 1 { frame[1] } else { frame[0] };
        let out = verb.tick((frame[0], right));
        output_samples.push(out.0);
    }

    write(output_path, &output_samples, sr, 1)?;
    Ok(())
}

fn calc_peaq(test_path: &str, reference_path: &str) -> Result<(f32, f32)> {
    let output = std::process::Command::new("/usr/bin/python")
        .arg("utils/peaq.py")
        .arg("--ref")
        .arg(reference_path)
        .arg("--test")
        .arg(test_path)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        anyhow::bail!(
            "peaq.py failed (status={}). stdout: {}\nstderr: {}",
            output.status,
            stdout,
            stderr
        );
    }

    let stdout = String::from_utf8(output.stdout).context("peaq.py stdout was not valid UTF-8")?;
    let mut values: HashMap<&str, &str> = HashMap::new();

    for line in stdout.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let (key, value) = line
            .split_once('=')
            .with_context(|| format!("expected KEY=VALUE line, got {line:?}"))?;
        values.insert(key.trim(), value.trim());
    }

    let odg: f32 = values
        .get("ODG")
        .context("missing ODG in PEAQ output")?
        .parse()
        .context("ODG is not a float")?;
    let di: f32 = values
        .get("DI")
        .context("missing DI in PEAQ output")?
        .parse()
        .context("DI is not a float")?;

    Ok((odg, di))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ResultEntry {
    original_path: String,
    reverberant_path: String,
    size: f32,
    wetness: f32,
    odg: f32,
    di: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct DatasetSplits {
    train: Vec<ResultEntry>,
    val: Vec<ResultEntry>,
    test: Vec<ResultEntry>,
}

fn process_one_file(input_path: &str, idx: usize) -> Result<ResultEntry> {
    let mut rng = StdRng::seed_from_u64(per_item_seed(SEED, idx));
    let room_size: f32 = rng.random_range(0.0..=1.0);
    let wet: f32 = rng.random_range(0.0..=1.0);

    let mut wav = read_wav(input_path)?;
    let output = format!("data/{}", input_path.split('/').last().unwrap());
    reverberate(&mut wav, &output, room_size, wet)?;
    let (odg, di) = calc_peaq(&output, input_path)?;

    Ok(ResultEntry {
        original_path: input_path.to_string(),
        reverberant_path: output,
        size: room_size,
        wetness: wet,
        odg,
        di,
    })
}

fn main() -> Result<()> {
    fs::create_dir_all("data")?;

    let args: Vec<String> = std::env::args().collect();
    let input = args.get(1).context("usage: make-data <input_dir>")?;

    let entries: Vec<String> = glob(&format!("{input}/*.wav"))
        .context("Failed to read glob pattern")?
        .map(|p| Ok(p?.display().to_string()))
        .collect::<Result<Vec<_>>>()?;

    anyhow::ensure!(
        !entries.is_empty(),
        "no .wav files found in input dir: {input}"
    );

    let total_files = entries.len();
    let processed = AtomicUsize::new(0);

    let results: Vec<ResultEntry> = entries
        .par_iter()
        .enumerate()
        .map(|(idx, path)| {
            let res = process_one_file(path, idx);
            let done = processed.fetch_add(1, Ordering::SeqCst) + 1;
            let pct = (done as f64) / (total_files as f64) * 100.0;
            println!("Progress: {pct:.1}% ({done}/{total_files})");
            res
        })
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    let split = SplitFractions {
        train: 0.7,
        val: 0.2,
        test: 0.1,
    };
    anyhow::ensure!(split.train > 0.0 && split.val >= 0.0 && split.test >= 0.0);
    anyhow::ensure!(
        (split.train + split.val + split.test - 1.0).abs() < 1e-9,
        "split fractions must sum to 1.0"
    );

    let total = results.len();

    let (train, test_val) = train_test_split(results, 1.0 - split.train, SEED)?;
    let val_ratio = split.val / (split.val + split.test);
    let (val, test) = train_test_split(test_val, 1.0 - val_ratio, SEED)?;

    println!(
        "Train set: {} files ({:.1}%)",
        train.len(),
        (train.len() as f64) / (total as f64) * 100.0
    );
    println!(
        "Eval set: {} files ({:.1}%)",
        val.len(),
        (val.len() as f64) / (total as f64) * 100.0
    );
    println!(
        "Test set: {} files ({:.1}%)",
        test.len(),
        (test.len() as f64) / (total as f64) * 100.0
    );

    let mut f = File::create("data/data.pkl")?;
    let splits = DatasetSplits { train, val, test };
    serde_pickle::to_writer(&mut f, &splits, Default::default())?;

    Ok(())
}
