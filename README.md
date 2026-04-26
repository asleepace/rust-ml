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

## License

MIT
