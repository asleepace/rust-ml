//! Tiny CSV reader, label encoding, and plaintext weight save/load.
//!
//! CSV assumptions (v1):
//!   * UTF-8, comma-separated.
//!   * First row is the header.
//!   * No quoted fields, no embedded commas/newlines.
//!   * Last column is the label; all other columns are numeric features.
//!
//! Weight file format (`NN_WEIGHTS_V1`):
//!   line 1:   NN_WEIGHTS_V1
//!   line 2:   arch=2,4,1
//!   line 3:   activations=relu
//!   line 4:   task=binary|multiclass|regression
//!   line 5:   labels=label_a,label_b,...     (multiclass only; else empty after `=`)
//!   then per Linear layer:
//!     line:   linear in=2 out=4
//!     line:   weights=<space-separated floats, row-major in*out>
//!     line:   bias=<space-separated floats, length out>

use crate::matrix::Matrix;
use crate::nn::{Activation, Linear, Network, Task};
use std::collections::HashMap;
use std::fs;

// ---------- CSV reading ----------

/// Generic CSV read: returns (header, rows_of_strings).
pub fn read_csv(path: &str) -> Result<(Vec<String>, Vec<Vec<String>>), String> {
    let content = fs::read_to_string(path).map_err(|e| format!("read {}: {}", path, e))?;
    let mut lines = content.lines();
    let header_line = lines.next().ok_or("csv is empty")?;
    let header: Vec<String> = header_line
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();
    let mut rows = Vec::new();
    for (lineno, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
        if row.len() != header.len() {
            return Err(format!(
                "line {}: {} fields, expected {}",
                lineno + 2,
                row.len(),
                header.len()
            ));
        }
        rows.push(row);
    }
    Ok((header, rows))
}

/// Parse a row of strings as f32 features.
fn parse_features(row: &[String], up_to: usize) -> Result<Vec<f32>, String> {
    let mut out = Vec::with_capacity(up_to);
    for (i, s) in row.iter().take(up_to).enumerate() {
        let v: f32 = s
            .parse()
            .map_err(|_| format!("column {} ({}): not a number: {:?}", i, s, s))?;
        out.push(v);
    }
    Ok(out)
}

// ---------- Loaded dataset ----------

pub struct Dataset {
    pub features: Matrix,            // (n_rows, n_features)
    pub feature_names: Vec<String>,
    pub label_name: String,
    /// For Binary/MultiClass, encoded targets are `Encoded::Class(_)`.
    /// For Regression, `Encoded::Numeric(_)`.
    pub targets: Encoded,
}

pub enum Encoded {
    /// raw f32 label per row (regression)
    Numeric(Vec<f32>),
    /// integer class index per row, plus the label vocabulary
    Class { indices: Vec<usize>, labels: Vec<String> },
}

/// Load a CSV with task-aware label encoding. If `expected_labels` is
/// provided (e.g. when predicting), labels are mapped against that vocab
/// (raises an error on unseen labels). Otherwise the vocab is built from
/// the data.
pub fn load_csv(
    path: &str,
    task: Task,
    expected_labels: Option<&[String]>,
) -> Result<Dataset, String> {
    let (header, rows) = read_csv(path)?;
    if header.len() < 2 {
        return Err("csv must have at least one feature column and one label column".into());
    }
    let n_features = header.len() - 1;
    let feature_names: Vec<String> = header[..n_features].to_vec();
    let label_name = header[n_features].clone();

    let mut feat_data = Vec::with_capacity(rows.len() * n_features);
    let mut targets_numeric: Vec<f32> = Vec::new();
    let mut targets_class: Vec<usize> = Vec::new();
    let mut label_vocab: Vec<String> = expected_labels.map(|l| l.to_vec()).unwrap_or_default();
    let mut label_index: HashMap<String, usize> = label_vocab
        .iter()
        .enumerate()
        .map(|(i, s)| (s.clone(), i))
        .collect();

    for row in &rows {
        feat_data.extend(parse_features(row, n_features)?);
        let label_str = &row[n_features];
        match task {
            Task::Regression => {
                let v: f32 = label_str
                    .parse()
                    .map_err(|_| format!("regression label not numeric: {:?}", label_str))?;
                targets_numeric.push(v);
            }
            Task::Binary => {
                // accept "0"/"1", "true"/"false", "yes"/"no" or strings (mapped to vocab)
                let lower = label_str.to_ascii_lowercase();
                let idx = match lower.as_str() {
                    "0" | "false" | "no" => 0,
                    "1" | "true" | "yes" => 1,
                    _ => {
                        if let Some(&i) = label_index.get(label_str) {
                            i
                        } else if expected_labels.is_some() {
                            return Err(format!("unseen label {:?}", label_str));
                        } else {
                            if label_vocab.len() >= 2 {
                                return Err(format!(
                                    "binary task: more than 2 distinct labels seen ({:?})",
                                    label_vocab
                                ));
                            }
                            let i = label_vocab.len();
                            label_vocab.push(label_str.clone());
                            label_index.insert(label_str.clone(), i);
                            i
                        }
                    }
                };
                if idx > 1 {
                    return Err(format!("binary task: label index {} out of range", idx));
                }
                targets_class.push(idx);
            }
            Task::MultiClass => {
                let idx = if let Some(&i) = label_index.get(label_str) {
                    i
                } else if expected_labels.is_some() {
                    return Err(format!("unseen label {:?}", label_str));
                } else {
                    let i = label_vocab.len();
                    label_vocab.push(label_str.clone());
                    label_index.insert(label_str.clone(), i);
                    i
                };
                targets_class.push(idx);
            }
        }
    }

    let features = Matrix::from_data(rows.len(), n_features, feat_data);

    let targets = match task {
        Task::Regression => Encoded::Numeric(targets_numeric),
        Task::Binary | Task::MultiClass => Encoded::Class {
            indices: targets_class,
            labels: label_vocab,
        },
    };

    Ok(Dataset {
        features,
        feature_names,
        label_name,
        targets,
    })
}

/// Build the (B x out) target matrix the loss expects.
/// Binary -> (B x 1) of 0/1. MultiClass -> (B x K) one-hot. Regression -> (B x 1) numeric.
pub fn build_target_matrix(targets: &Encoded, task: Task) -> Matrix {
    match (task, targets) {
        (Task::Regression, Encoded::Numeric(v)) => {
            Matrix::from_data(v.len(), 1, v.clone())
        }
        (Task::Binary, Encoded::Class { indices, .. }) => {
            let data: Vec<f32> = indices.iter().map(|&i| i as f32).collect();
            Matrix::from_data(indices.len(), 1, data)
        }
        (Task::MultiClass, Encoded::Class { indices, labels }) => {
            let k = labels.len();
            let mut data = vec![0.0f32; indices.len() * k];
            for (row, &idx) in indices.iter().enumerate() {
                data[row * k + idx] = 1.0;
            }
            Matrix::from_data(indices.len(), k, data)
        }
        _ => panic!("encoding does not match task"),
    }
}

// ---------- Weight save / load ----------

pub fn save_weights(net: &Network, labels: &[String], path: &str) -> Result<(), String> {
    let mut s = String::new();
    s.push_str("NN_WEIGHTS_V1\n");
    s.push_str("arch=");
    for (i, v) in net.arch.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&v.to_string());
    }
    s.push('\n');
    s.push_str("activations=");
    for (i, a) in net.hidden_acts.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(a.name());
    }
    s.push('\n');
    s.push_str(&format!("task={}\n", net.task.name()));
    s.push_str("labels=");
    for (i, l) in labels.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(l);
    }
    s.push('\n');
    for layer in &net.layers {
        layer.dump(&mut s);
    }
    fs::write(path, s).map_err(|e| format!("write {}: {}", path, e))?;
    Ok(())
}

pub struct LoadedModel {
    pub net: Network,
    pub labels: Vec<String>,
}

pub fn load_weights(path: &str) -> Result<LoadedModel, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("read {}: {}", path, e))?;
    let mut lines = content.lines();

    let magic = lines.next().ok_or("empty weights file")?;
    if magic != "NN_WEIGHTS_V1" {
        return Err(format!("unexpected magic: {:?}", magic));
    }

    fn kv<'a>(line: &'a str, key: &str) -> Result<&'a str, String> {
        let prefix = format!("{}=", key);
        line.strip_prefix(&prefix)
            .ok_or_else(|| format!("expected line beginning with {:?}, got {:?}", prefix, line))
    }

    let arch: Vec<usize> = kv(lines.next().ok_or("missing arch")?, "arch")?
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().map_err(|e| format!("arch parse: {}", e)))
        .collect::<Result<_, _>>()?;

    let acts_str = kv(lines.next().ok_or("missing activations")?, "activations")?;
    let hidden_acts: Vec<Activation> = if acts_str.is_empty() {
        Vec::new()
    } else {
        acts_str
            .split(',')
            .map(Activation::parse)
            .collect::<Result<_, _>>()?
    };

    let task = Task::parse(kv(lines.next().ok_or("missing task")?, "task")?)?;

    let labels_str = kv(lines.next().ok_or("missing labels")?, "labels")?;
    let labels: Vec<String> = if labels_str.is_empty() {
        Vec::new()
    } else {
        labels_str.split(',').map(|s| s.to_string()).collect()
    };

    // Build a network with the right shape, then overwrite each Linear's params.
    let mut net = Network::new(arch.clone(), hidden_acts.clone(), task, 1)?;
    // Replace each Linear in net.layers with one constructed from the file.
    let mut layer_idx = 0;
    for i in 0..(arch.len() - 1) {
        let header = lines.next().ok_or("missing linear header")?;
        // expected: "linear in=N out=M"
        let parts: Vec<&str> = header.split_whitespace().collect();
        if parts.len() != 3 || parts[0] != "linear" {
            return Err(format!("bad linear header: {:?}", header));
        }
        let in_size: usize = parts[1]
            .strip_prefix("in=")
            .ok_or("missing in=")?
            .parse()
            .map_err(|e| format!("in= parse: {}", e))?;
        let out_size: usize = parts[2]
            .strip_prefix("out=")
            .ok_or("missing out=")?
            .parse()
            .map_err(|e| format!("out= parse: {}", e))?;
        if in_size != arch[i] || out_size != arch[i + 1] {
            return Err(format!(
                "linear layer {}: header says {}x{}, arch says {}x{}",
                i, in_size, out_size, arch[i], arch[i + 1]
            ));
        }

        let w_line = lines.next().ok_or("missing weights")?;
        let w_str = w_line.strip_prefix("weights=").ok_or("expected weights=")?;
        let w_data: Vec<f32> = w_str
            .split_whitespace()
            .map(|s| s.parse::<f32>().map_err(|e| format!("weight parse: {}", e)))
            .collect::<Result<_, _>>()?;
        if w_data.len() != in_size * out_size {
            return Err(format!(
                "weight count mismatch: got {}, expected {}",
                w_data.len(),
                in_size * out_size
            ));
        }

        let b_line = lines.next().ok_or("missing bias")?;
        let b_str = b_line.strip_prefix("bias=").ok_or("expected bias=")?;
        let b_data: Vec<f32> = b_str
            .split_whitespace()
            .map(|s| s.parse::<f32>().map_err(|e| format!("bias parse: {}", e)))
            .collect::<Result<_, _>>()?;
        if b_data.len() != out_size {
            return Err(format!(
                "bias count mismatch: got {}, expected {}",
                b_data.len(),
                out_size
            ));
        }

        let w = Matrix::from_data(in_size, out_size, w_data);
        let b = Matrix::from_data(1, out_size, b_data);
        net.layers[layer_idx] = Box::new(Linear::from_params(in_size, out_size, w, b));
        // skip the activation that was inserted between linears (except after the last)
        layer_idx += 1;
        if i < arch.len() - 2 {
            layer_idx += 1;
        }
    }

    Ok(LoadedModel { net, labels })
}
