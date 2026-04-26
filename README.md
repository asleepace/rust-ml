# nn-train

A tiny multilayer-perceptron trainer in pure Rust. **Zero dependencies** — just `cargo build` and run. Built for experimenting with small nets across different projects without pulling in `candle`, `burn`, `tch`, or PyTorch.

## What it does

Train an MLP from a CSV, save weights, run inference, evaluate on held-out data. Three task types:

- **binary** — sigmoid output + BCE loss
- **multiclass** — softmax output + cross-entropy
- **regression** — linear output + MSE

Hand-rolled matrix ops, hand-rolled backprop, hand-rolled CSV parser. If you want to learn how an MLP actually works, the source is ~1000 lines across 4 files and reads top-to-bottom.

## Build

```bash
cargo build --release
./target/release/nn-train help
```

## Quick sanity check (XOR)

No CSV needed — it's built in:

```bash
cargo run --release -- xor
```

You should see loss decrease toward zero and the four predictions land near 0/1/1/0.

## Train from a CSV

CSV format: header row, last column is the label, every other column is a numeric feature. No quoted fields, no embedded commas.

```csv
x1,x2,label
0.1,0.5,0
0.8,0.3,1
0.2,0.9,1
...
```

```bash
cargo run --release -- fit \
    --data train.csv \
    --arch 2,8,2 \
    --activations relu \
    --task multiclass \
    --epochs 200 \
    --lr 0.05 \
    --batch-size 32 \
    --val val.csv \
    --out weights.txt
```

### Architecture spec

`--arch` is comma-separated layer sizes including input and output. `--activations` is one entry per **hidden** layer (so `arch.len() - 2` of them). The final activation is implicit — chosen by `--task`.

Example: `--arch 17,32,16,1 --activations relu,relu --task binary` builds the listenlabs-style net (17 inputs → 32 ReLU → 16 ReLU → 1 sigmoid).

### Output sizes per task

| Task | Required output size |
|---|---|
| binary | 1 |
| multiclass | number of distinct labels in the CSV |
| regression | 1 |

If they don't match, fit fails fast with a clear error.

## Predict

```bash
cargo run --release -- predict \
    --weights weights.txt \
    --data unlabeled.csv \
    --out preds.csv
```

The input CSV may include a label column (it's ignored). Output format depends on task:

- **binary** → `p_pos,pred`
- **multiclass** → `prob_<label0>,prob_<label1>,...,pred`
- **regression** → `pred`

## Eval

```bash
cargo run --release -- eval --weights weights.txt --data labeled.csv
```

Prints loss and (for classification tasks) accuracy.

## Weights file

Plaintext, human-readable. Magic header `NN_WEIGHTS_V1`. You can `cat`, `diff`, and `grep` weights files. Format:

```
NN_WEIGHTS_V1
arch=2,4,1
activations=tanh
task=binary
labels=
linear in=2 out=4
weights=<8 floats space-separated>
bias=<4 floats>
linear in=4 out=1
weights=<4 floats>
bias=<1 float>
```

## Project layout

```
src/
├── matrix.rs    matrix type + matmul/transpose-variants + xorshift RNG
├── nn.rs        Layer trait, Linear, ActivationLayer, Network, losses
├── data.rs      CSV reader, label encoding, weight save/load
└── main.rs      CLI dispatch + training loop
```

Reading order if you want to understand it: `matrix.rs` → `nn.rs` → `data.rs` → `main.rs`.

## Design choices worth knowing

1. **Logits-out forward.** The network outputs raw logits during training; the loss function combines the final activation (sigmoid/softmax) with the loss for numerical stability. At predict-time we apply the activation explicitly.
2. **Gradient averaging.** The loss gradient is divided by batch size, so layers don't have to.
3. **`Box<dyn Layer>`** for heterogeneous layers in a `Vec`. Slight runtime cost over generics but trivially extensible.
4. **He initialization** for weights (good default for ReLU). Deterministic given `--seed`.
5. **Plain SGD**, no momentum/Adam in v1. Easy to add — just give Linear a velocity matrix and a momentum field on the optimizer.

## v2 ideas (intentionally left out)

- Adam / momentum
- Dropout, batch norm
- Learning-rate schedules
- Binary weight format (faster load)
- Quoted CSV fields
- Conv / RNN layers
- Autograd (vs. hand-derived backprop)

If you need any of these, they're each a small isolated addition rather than a rewrite.

## How to use?

```
  [Postgres]                                    [predictions]
      |                                              ^
      | export                                       |
      v                                              |
  train.csv  ─►  nn-train fit  ─►  weights.txt  ─►  nn-train predict  ─►  preds.csv
                                                    ^
                                                    |
                                                today.csv (1 row, today's snapshot)
```

### Step 1: Export labeled training data
You need a SQL query that produces one row per labeled trading day with all your features and a label column at the end. From your existing schema (after the ALTER TABLE from the data collection plan), something like:

```sql
COPY (
    SELECT
        -- features (all numeric, last column will be the label)
        (spot_price - max_pain) / spot_price       AS dist_to_max_pain,
        (spot_price - gamma_flip) / spot_price     AS dist_to_gamma_flip,
        net_gex / spot_price                       AS gex_normalized,
        put_call_ratio,
        atm_iv,
        iv_skew,
        net_delta_exposure / spot_price            AS delta_normalized,
        -- add more features here as you collect them
        -- ...
        -- label LAST
        CASE
            WHEN forward_return_1d > 0.003 THEN 'up'
            WHEN forward_return_1d < -0.003 THEN 'down'
            ELSE 'flat'
        END AS direction
    FROM snapshots
    WHERE ticker = 'spy'
      AND is_decision_snapshot = true
      AND forward_return_1d IS NOT NULL
    ORDER BY trading_date
) TO '/tmp/spy_train.csv' WITH (FORMAT CSV, HEADER);
```

Important: temporal split must happen here, not at training time. The CLI shuffles rows internally for SGD, which is fine, but you have to keep the last 15-20% of dates out of train.csv entirely. Otherwise the model trains on the future and "validates" on the past.)

```sql
-- determine split date (e.g. last 6 weeks held out)
-- then write three files
COPY ( ... WHERE trading_date <  '2026-XX-XX' ... ) TO '/tmp/spy_train.csv' ...
COPY ( ... WHERE trading_date >= '2026-XX-XX' AND trading_date < '2026-YY-YY' ... ) TO '/tmp/spy_val.csv' ...
COPY ( ... WHERE trading_date >= '2026-YY-YY' ... ) TO '/tmp/spy_test.csv' ...
```
Test set: never touch until you're done iterating. Val set: use during model selection.

### Step 2: Standardize features
Heads up — this is the thing I flagged earlier and it's about to bite you. Your features span wildly different scales (GEX in millions, IV in [0,1]). MLPs trained on raw values like that converge slowly or not at all because gradients are dominated by large-magnitude features.

Quickest fix without modifying nn-train: standardize in SQL using stats computed from the train set only. Compute mean and stddev over the training set, then subtract/divide for all three splits using those same numbers (don't recompute on val/test — that leaks).

```sql
-- compute scaling stats from train only
WITH stats AS (
    SELECT
        AVG(put_call_ratio) AS pcr_mean, STDDEV(put_call_ratio) AS pcr_std,
        AVG(atm_iv) AS iv_mean, STDDEV(atm_iv) AS iv_std
        -- ...
    FROM snapshots
    WHERE ticker = 'spy' AND is_decision_snapshot AND trading_date < '2026-XX-XX'
)
-- then in your COPY query:
SELECT
    (put_call_ratio - (SELECT pcr_mean FROM stats)) / (SELECT pcr_std FROM stats) AS pcr_z,
    ...
```
Save those stats somewhere (a small JSON file). You'll need them at predict time too — same transform on every new snapshot.

### Step #3. Train

```bash
nn-train fit \
    --data spy_train.csv \
    --val spy_val.csv \
    --arch 7,16,8,3 \
    --activations relu,relu \
    --task multiclass \
    --epochs 300 \
    --lr 0.01 \
    --batch-size 16 \
    --seed 42 \
    --out spy_direction.txt
```

The `--arch 7,16,8,3` reads as: 7 input features → 16 hidden → 8 hidden → 3 output classes (up/flat/down). Watch the val loss vs train loss in the output — if val loss starts climbing while train loss keeps falling, you're overfitting and should stop.

Sanity check before believing anything: train a baseline with `--arch 7,3 --activations` (empty — pure logistic regression, no hidden layers). If your MLP doesn't beat that, the depth isn't helping and you're just overfitting to noise. With 30 rows it almost certainly won't beat LR.

### Step 4: Predict on a new snapshot
When a new premarket snapshot lands, you produce a single-row CSV with the same feature columns in the same order, applying the same standardization you computed in step 2.

```sql
COPY (
    SELECT
        (spot_price - max_pain) / spot_price,
        -- ... same transforms in same order ...
    FROM snapshots
    WHERE ticker = 'spy'
      AND captured_at = (SELECT MAX(captured_at) FROM snapshots WHERE ticker = 'spy')
) TO '/tmp/today.csv' WITH (FORMAT CSV, HEADER);
```
Then:
```bash
nn-train predict \
    --weights spy_direction.txt \
    --data today.csv \
    --out today_pred.csv

cat today_pred.csv
# prob_up,prob_flat,prob_down,pred
# 0.42,0.31,0.27,up
```

### Step 5: Evaluate honestly
After collecting more data and you're ready to score the model on the held-out test set:

```bash
nn-train eval --weights spy_direction.txt --data spy_test.csv
# loss=1.0234
# accuracy=0.412
```

**Read the accuracy in context.** With 3 classes, random is 33%, and class-balance baselines (always predict majority class) might be 40-45% if your data is imbalanced. Beating those is the bar. If your test set is small (e.g. 12 days), the 95% CI on the accuracy is huge — bootstrap it before trusting any number.
What to do while data accumulates
Concrete suggestions for the next 3 months:

1. **Set up the export queries now**, so when you have enough data you press one button.
2. **Run the pipeline weekly** with whatever you have. The numbers will be meaningless but the pipeline gets debugged.
3. **Track baseline performance over time**. Each week, log: how often "predict up always" works, how often "predict same direction as yesterday" works. These are your null hypotheses to beat.
4. **Standardization in SQL is fine for v1** but if you find yourself rewriting it constantly, add a `--standardize` flag to nn-train that computes stats from train and writes them next to the weights file. ~50 lines.

## License

MIT
