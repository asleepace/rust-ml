//! nn-train: tiny MLP trainer with no dependencies.
//!
//! Subcommands:
//!   fit      train a network from a CSV
//!   predict  load weights, write predictions to a CSV
//!   eval     load weights, print loss + accuracy on a labeled CSV
//!   xor      built-in XOR sanity check (no CSV needed)

mod data;
mod matrix;
mod nn;

use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Write;
use std::process;

use crate::data::{
    build_target_matrix, load_csv, load_weights, save_weights, Encoded, LoadedModel,
};
use crate::matrix::Matrix;
use crate::nn::{Activation, Network, Task};

fn main() {
    let argv: Vec<String> = env::args().collect();
    if argv.len() < 2 {
        usage();
        process::exit(1);
    }
    let (cmd, args) = parse_args(&argv);
    let result = match cmd.as_str() {
        "fit" => run_fit(&args),
        "predict" => run_predict(&args),
        "eval" => run_eval(&args),
        "xor" => run_xor(&args),
        "help" | "--help" | "-h" => {
            usage();
            Ok(())
        }
        other => Err(format!("unknown subcommand: {}", other)),
    };
    if let Err(e) = result {
        eprintln!("error: {}", e);
        process::exit(1);
    }
}

fn usage() {
    eprintln!(
        "nn-train — train small MLPs (no deps)\n\
         \n\
         USAGE:\n\
           nn-train fit     --data train.csv --arch 2,8,2 --activations relu \\\n\
                            --task multiclass --epochs 200 --lr 0.05 --out weights.txt \\\n\
                            [--batch-size 32] [--seed 42] [--val val.csv] [--quiet]\n\
         \n\
           nn-train predict --weights weights.txt --data x.csv --out preds.csv\n\
         \n\
           nn-train eval    --weights weights.txt --data labeled.csv\n\
         \n\
           nn-train xor     [--epochs 4000] [--lr 0.1] [--seed 42]\n\
         \n\
         FLAGS\n\
           --arch         comma-separated layer sizes including input & output\n\
                          (e.g. 17,32,16,1 for the listenlabs net)\n\
           --activations  one per HIDDEN layer; final activation is implicit\n\
                          and chosen by --task. e.g. arch=17,32,16,1 needs\n\
                          activations=relu,relu (2 hidden layers).\n\
                          Choices: relu, sigmoid, tanh, identity\n\
           --task         binary | multiclass | regression\n\
           --batch-size   default 32. set 0 to use full-batch.\n\
           --val          optional held-out CSV for per-epoch validation loss\n\
           --quiet        suppress per-epoch output\n"
    );
}

fn parse_args(argv: &[String]) -> (String, HashMap<String, String>) {
    let cmd = argv[1].clone();
    let mut map = HashMap::new();
    let mut i = 2;
    while i < argv.len() {
        let a = &argv[i];
        if let Some(rest) = a.strip_prefix("--") {
            // allow --key=value or --key value
            if let Some((k, v)) = rest.split_once('=') {
                map.insert(k.to_string(), v.to_string());
                i += 1;
            } else if i + 1 < argv.len() && !argv[i + 1].starts_with("--") {
                map.insert(rest.to_string(), argv[i + 1].clone());
                i += 2;
            } else {
                map.insert(rest.to_string(), "true".to_string());
                i += 1;
            }
        } else {
            eprintln!("unexpected positional arg: {:?}", a);
            i += 1;
        }
    }
    (cmd, map)
}

fn require<'a>(args: &'a HashMap<String, String>, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .map(|s| s.as_str())
        .ok_or_else(|| format!("missing required flag --{}", key))
}

fn parse_arch(s: &str) -> Result<Vec<usize>, String> {
    s.split(',')
        .map(|p| p.parse::<usize>().map_err(|e| format!("arch parse: {}", e)))
        .collect()
}

fn parse_activations(s: &str) -> Result<Vec<Activation>, String> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    s.split(',').map(Activation::parse).collect()
}

// ---------- fit ----------

fn run_fit(args: &HashMap<String, String>) -> Result<(), String> {
    let data_path = require(args, "data")?;
    let arch_str = require(args, "arch")?;
    let acts_str = require(args, "activations")?;
    let task_str = require(args, "task")?;
    let out_path = require(args, "out")?;

    let arch = parse_arch(arch_str)?;
    let hidden_acts = parse_activations(acts_str)?;
    let task = Task::parse(task_str)?;
    let epochs: usize = args
        .get("epochs")
        .map(|s| s.parse::<usize>().map_err(|e| e.to_string()))
        .unwrap_or(Ok(100))?;
    let lr: f32 = args
        .get("lr")
        .map(|s| s.parse::<f32>().map_err(|e| e.to_string()))
        .unwrap_or(Ok(0.01))?;
    let batch_size: usize = args
        .get("batch-size")
        .map(|s| s.parse::<usize>().map_err(|e| e.to_string()))
        .unwrap_or(Ok(32))?;
    let seed: u64 = args
        .get("seed")
        .map(|s| s.parse::<u64>().map_err(|e| e.to_string()))
        .unwrap_or(Ok(42))?;
    let quiet = args.contains_key("quiet");
    let val_path = args.get("val").cloned();

    let train = load_csv(data_path, task, None)?;
    let n_train = train.features.rows;
    let n_features = train.features.cols;
    if n_features != arch[0] {
        return Err(format!(
            "csv has {} feature columns but arch[0] = {}",
            n_features, arch[0]
        ));
    }

    // For classification, expected output size is the label vocabulary.
    let labels: Vec<String> = match &train.targets {
        Encoded::Class { labels, .. } => labels.clone(),
        Encoded::Numeric(_) => Vec::new(),
    };
    let expected_out = match task {
        Task::Binary => 1,
        Task::MultiClass => labels.len().max(1),
        Task::Regression => 1,
    };
    if *arch.last().unwrap() != expected_out {
        return Err(format!(
            "arch[-1] = {} but task {} expects output size {}{}",
            arch.last().unwrap(),
            task.name(),
            expected_out,
            match task {
                Task::MultiClass => format!(" ({} classes)", labels.len()),
                _ => String::new(),
            }
        ));
    }

    let train_targets = build_target_matrix(&train.targets, task);

    let val = match val_path {
        Some(p) => Some(load_csv(&p, task, if labels.is_empty() { None } else { Some(&labels) })?),
        None => None,
    };

    let mut net = Network::new(arch.clone(), hidden_acts.clone(), task, seed)?;

    if !quiet {
        eprintln!(
            "fit: n={} features={} arch={:?} acts={} task={} epochs={} lr={} batch={}",
            n_train,
            n_features,
            arch,
            acts_str,
            task.name(),
            epochs,
            lr,
            batch_size
        );
    }

    // Simple shuffled-minibatch SGD using xorshift for deterministic perm.
    let mut rng = matrix::Xorshift::new(seed.wrapping_add(7));
    let mut indices: Vec<usize> = (0..n_train).collect();
    let effective_batch = if batch_size == 0 { n_train } else { batch_size.min(n_train) };

    for epoch in 0..epochs {
        // Fisher-Yates shuffle
        for i in (1..n_train).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            indices.swap(i, j);
        }

        let mut total_loss = 0.0f32;
        let mut steps = 0usize;
        let mut start = 0usize;
        while start < n_train {
            let end = (start + effective_batch).min(n_train);
            let x = gather_rows(&train.features, &indices[start..end]);
            let y = gather_rows(&train_targets, &indices[start..end]);
            total_loss += net.train_step(&x, &y, lr);
            steps += 1;
            start = end;
        }
        let avg_train = total_loss / steps.max(1) as f32;

        if !quiet {
            let val_str = if let Some(ref ds) = val {
                let yt = build_target_matrix(&ds.targets, task);
                let logits = net.forward(&ds.features);
                let (vloss, _) = nn::compute_loss(&logits, &yt, task);
                let preds = net.predict(&ds.features);
                let acc = compute_accuracy(&preds, &ds.targets, task);
                match acc {
                    Some(a) => format!("  val_loss={:.4} val_acc={:.3}", vloss, a),
                    None => format!("  val_loss={:.4}", vloss),
                }
            } else {
                String::new()
            };
            eprintln!("epoch {:>4}/{}: train_loss={:.4}{}", epoch + 1, epochs, avg_train, val_str);
        }
    }

    save_weights(&net, &labels, out_path)?;
    eprintln!("saved weights to {}", out_path);
    Ok(())
}

fn gather_rows(m: &Matrix, idx: &[usize]) -> Matrix {
    let mut out = Matrix::zeros(idx.len(), m.cols);
    for (r, &src) in idx.iter().enumerate() {
        let s = src * m.cols;
        let d = r * m.cols;
        out.data[d..d + m.cols].copy_from_slice(&m.data[s..s + m.cols]);
    }
    out
}

// ---------- predict ----------

fn run_predict(args: &HashMap<String, String>) -> Result<(), String> {
    let weights_path = require(args, "weights")?;
    let data_path = require(args, "data")?;
    let out_path = require(args, "out")?;

    let LoadedModel { mut net, labels } = load_weights(weights_path)?;
    // For predict we don't strictly need labels in the file. Read CSV as plain features:
    let (header, rows) = data::read_csv(data_path)?;
    if header.is_empty() {
        return Err("empty csv".into());
    }
    // Decide whether the last column looks like a label or a feature.
    // Convention: if header.len() == arch[0], all columns are features;
    // if header.len() == arch[0] + 1, the last column is ignored.
    let n_features = net.arch[0];
    let n_cols = if header.len() == n_features {
        n_features
    } else if header.len() == n_features + 1 {
        n_features
    } else {
        return Err(format!(
            "csv has {} columns but model expects {} features",
            header.len(),
            n_features
        ));
    };
    let mut feat_data = Vec::with_capacity(rows.len() * n_cols);
    for row in &rows {
        for i in 0..n_cols {
            let v: f32 = row[i]
                .parse()
                .map_err(|_| format!("non-numeric feature: {:?}", row[i]))?;
            feat_data.push(v);
        }
    }
    let x = Matrix::from_data(rows.len(), n_cols, feat_data);
    let preds = net.predict(&x);

    let mut out = String::new();
    match net.task {
        Task::Binary => {
            out.push_str("p_pos,pred\n");
            for i in 0..preds.rows {
                let p = preds.data[i];
                let label = if p >= 0.5 {
                    if labels.len() >= 2 { labels[1].as_str() } else { "1" }
                } else if !labels.is_empty() {
                    labels[0].as_str()
                } else {
                    "0"
                };
                out.push_str(&format!("{},{}\n", p, label));
            }
        }
        Task::MultiClass => {
            // header: prob_<label0>,prob_<label1>,...,pred
            for (i, l) in labels.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&format!("prob_{}", l));
            }
            out.push_str(",pred\n");
            let k = preds.cols;
            for i in 0..preds.rows {
                let row = i * k;
                let mut argmax = 0usize;
                let mut best = f32::NEG_INFINITY;
                for j in 0..k {
                    let v = preds.data[row + j];
                    if j > 0 {
                        out.push(',');
                    }
                    out.push_str(&format!("{}", v));
                    if v > best {
                        best = v;
                        argmax = j;
                    }
                }
                let label = if argmax < labels.len() {
                    labels[argmax].as_str()
                } else {
                    "?"
                };
                out.push_str(&format!(",{}\n", label));
            }
        }
        Task::Regression => {
            out.push_str("pred\n");
            for v in &preds.data {
                out.push_str(&format!("{}\n", v));
            }
        }
    }
    let mut f = fs::File::create(out_path).map_err(|e| format!("create {}: {}", out_path, e))?;
    f.write_all(out.as_bytes())
        .map_err(|e| format!("write: {}", e))?;
    eprintln!("wrote {} predictions to {}", preds.rows, out_path);
    Ok(())
}

// ---------- eval ----------

fn run_eval(args: &HashMap<String, String>) -> Result<(), String> {
    let weights_path = require(args, "weights")?;
    let data_path = require(args, "data")?;

    let LoadedModel { mut net, labels } = load_weights(weights_path)?;
    let task = net.task;
    let ds = load_csv(
        data_path,
        task,
        if labels.is_empty() { None } else { Some(&labels) },
    )?;
    let yt = build_target_matrix(&ds.targets, task);
    let logits = net.forward(&ds.features);
    let (loss, _) = nn::compute_loss(&logits, &yt, task);
    let preds = net.predict(&ds.features);
    let acc = compute_accuracy(&preds, &ds.targets, task);

    println!("loss={:.6}", loss);
    if let Some(a) = acc {
        println!("accuracy={:.6}", a);
    }
    Ok(())
}

fn compute_accuracy(preds: &Matrix, targets: &Encoded, task: Task) -> Option<f32> {
    match (task, targets) {
        (Task::Binary, Encoded::Class { indices, .. }) => {
            let mut correct = 0;
            for i in 0..preds.rows {
                let pred = if preds.data[i] >= 0.5 { 1 } else { 0 };
                if pred == indices[i] {
                    correct += 1;
                }
            }
            Some(correct as f32 / preds.rows as f32)
        }
        (Task::MultiClass, Encoded::Class { indices, .. }) => {
            let k = preds.cols;
            let mut correct = 0;
            for i in 0..preds.rows {
                let row = i * k;
                let mut argmax = 0usize;
                let mut best = f32::NEG_INFINITY;
                for j in 0..k {
                    if preds.data[row + j] > best {
                        best = preds.data[row + j];
                        argmax = j;
                    }
                }
                if argmax == indices[i] {
                    correct += 1;
                }
            }
            Some(correct as f32 / preds.rows as f32)
        }
        _ => None,
    }
}

// ---------- xor (built-in toy) ----------

fn run_xor(args: &HashMap<String, String>) -> Result<(), String> {
    let epochs: usize = args
        .get("epochs")
        .map(|s| s.parse::<usize>().map_err(|e| e.to_string()))
        .unwrap_or(Ok(4000))?;
    let lr: f32 = args
        .get("lr")
        .map(|s| s.parse::<f32>().map_err(|e| e.to_string()))
        .unwrap_or(Ok(0.1))?;
    let seed: u64 = args
        .get("seed")
        .map(|s| s.parse::<u64>().map_err(|e| e.to_string()))
        .unwrap_or(Ok(42))?;

    let x = Matrix::from_data(
        4,
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    );
    let y = Matrix::from_data(4, 1, vec![0.0, 1.0, 1.0, 0.0]);

    let mut net = Network::new(
        vec![2, 4, 1],
        vec![Activation::Tanh],
        Task::Binary,
        seed,
    )?;

    println!("XOR sanity check: arch=[2,4,1] tanh, BCE loss, full-batch SGD");
    for epoch in 0..epochs {
        let loss = net.train_step(&x, &y, lr);
        if epoch % (epochs / 10).max(1) == 0 {
            println!("  epoch {:>5}: loss={:.6}", epoch, loss);
        }
    }
    let preds = net.predict(&x);
    println!("\nfinal predictions:");
    for i in 0..4 {
        println!(
            "  ({},{}) -> {:.4}  (truth {})",
            x.data[i * 2],
            x.data[i * 2 + 1],
            preds.data[i],
            y.data[i]
        );
    }
    Ok(())
}
