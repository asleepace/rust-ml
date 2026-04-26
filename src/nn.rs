//! Neural net core: Layer trait, Linear & Activation layers, Network, losses.
//!
//! Design notes:
//! * Layers form a `Vec<Box<dyn Layer>>` (heterogeneous, easy to extend).
//! * The network outputs RAW LOGITS during training. The loss function knows
//!   how to combine the final activation (sigmoid/softmax/identity) with the
//!   loss for numerical stability and a simpler gradient. At predict time we
//!   apply the appropriate final activation explicitly.
//! * Gradients are averaged over the batch by dividing by `batch_size`
//!   in the loss gradient. So Layer::step does NOT divide.

use crate::matrix::{he_init, Matrix, Xorshift};

// ---------- Activations ----------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Activation {
    Relu,
    Sigmoid,
    Tanh,
    Identity,
}

impl Activation {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "relu" => Ok(Self::Relu),
            "sigmoid" => Ok(Self::Sigmoid),
            "tanh" => Ok(Self::Tanh),
            "identity" | "linear" | "none" => Ok(Self::Identity),
            other => Err(format!("unknown activation: {}", other)),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Relu => "relu",
            Self::Sigmoid => "sigmoid",
            Self::Tanh => "tanh",
            Self::Identity => "identity",
        }
    }
}

// ---------- Task ----------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Task {
    Binary,     // sigmoid + BCE
    MultiClass, // softmax + cross-entropy
    Regression, // identity + MSE
}

impl Task {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "binary" => Ok(Self::Binary),
            "multiclass" | "multi" | "classification" => Ok(Self::MultiClass),
            "regression" | "regress" => Ok(Self::Regression),
            other => Err(format!("unknown task: {}", other)),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Binary => "binary",
            Self::MultiClass => "multiclass",
            Self::Regression => "regression",
        }
    }
}

// ---------- Layer trait ----------

pub trait Layer {
    fn forward(&mut self, x: &Matrix) -> Matrix;
    fn backward(&mut self, grad_y: &Matrix) -> Matrix;
    fn step(&mut self, lr: f32);
    /// Append a textual representation of this layer's parameters to `out`.
    /// Activation layers contribute nothing; Linear layers write their weights & biases.
    fn dump(&self, out: &mut String);
}

// ---------- Linear (Dense) ----------

pub struct Linear {
    pub w: Matrix,    // (in, out)
    pub b: Matrix,    // (1, out)
    pub in_size: usize,
    pub out_size: usize,
    last_input: Option<Matrix>,
    grad_w: Matrix,
    grad_b: Matrix,
}

impl Linear {
    pub fn new(in_size: usize, out_size: usize, rng: &mut Xorshift) -> Self {
        Self {
            w: he_init(in_size, out_size, in_size, rng),
            b: Matrix::zeros(1, out_size),
            in_size,
            out_size,
            last_input: None,
            grad_w: Matrix::zeros(in_size, out_size),
            grad_b: Matrix::zeros(1, out_size),
        }
    }

    /// Construct from already-loaded weights & biases (used by `Network::load`).
    pub fn from_params(in_size: usize, out_size: usize, w: Matrix, b: Matrix) -> Self {
        assert_eq!(w.rows, in_size);
        assert_eq!(w.cols, out_size);
        assert_eq!(b.rows, 1);
        assert_eq!(b.cols, out_size);
        Self {
            w,
            b,
            in_size,
            out_size,
            last_input: None,
            grad_w: Matrix::zeros(in_size, out_size),
            grad_b: Matrix::zeros(1, out_size),
        }
    }
}

impl Layer for Linear {
    fn forward(&mut self, x: &Matrix) -> Matrix {
        // y = x @ w + b   (b broadcast across rows)
        let mut y = Matrix::matmul(x, &self.w);
        y.add_row_broadcast(&self.b);
        self.last_input = Some(x.clone());
        y
    }

    fn backward(&mut self, grad_y: &Matrix) -> Matrix {
        let x = self
            .last_input
            .as_ref()
            .expect("Linear::backward called before forward");
        // grad_w = x^T @ grad_y    shape (in, out)
        // grad_b = sum_over_rows(grad_y)  shape (1, out)
        // grad_x = grad_y @ w^T    shape (B, in)
        self.grad_w = Matrix::matmul_a_t(x, grad_y);
        self.grad_b = grad_y.sum_rows_to_row();
        Matrix::matmul_b_t(grad_y, &self.w)
    }

    fn step(&mut self, lr: f32) {
        for i in 0..self.w.data.len() {
            self.w.data[i] -= lr * self.grad_w.data[i];
        }
        for i in 0..self.b.data.len() {
            self.b.data[i] -= lr * self.grad_b.data[i];
        }
    }

    fn dump(&self, out: &mut String) {
        out.push_str(&format!("linear in={} out={}\n", self.in_size, self.out_size));
        out.push_str("weights=");
        for (i, v) in self.w.data.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(&format!("{}", v));
        }
        out.push('\n');
        out.push_str("bias=");
        for (i, v) in self.b.data.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(&format!("{}", v));
        }
        out.push('\n');
    }
}

// ---------- Activation layer ----------

pub struct ActivationLayer {
    pub kind: Activation,
    last_output: Option<Matrix>,
}

impl ActivationLayer {
    pub fn new(kind: Activation) -> Self {
        Self { kind, last_output: None }
    }
}

impl Layer for ActivationLayer {
    fn forward(&mut self, x: &Matrix) -> Matrix {
        let mut y = x.clone();
        match self.kind {
            Activation::Relu => {
                for v in &mut y.data {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            Activation::Sigmoid => {
                for v in &mut y.data {
                    *v = 1.0 / (1.0 + (-*v).exp());
                }
            }
            Activation::Tanh => {
                for v in &mut y.data {
                    *v = v.tanh();
                }
            }
            Activation::Identity => {}
        }
        self.last_output = Some(y.clone());
        y
    }

    fn backward(&mut self, grad_y: &Matrix) -> Matrix {
        let y = self
            .last_output
            .as_ref()
            .expect("ActivationLayer::backward called before forward");
        let mut grad_x = Matrix::zeros(grad_y.rows, grad_y.cols);
        match self.kind {
            Activation::Relu => {
                for i in 0..grad_y.data.len() {
                    grad_x.data[i] = if y.data[i] > 0.0 { grad_y.data[i] } else { 0.0 };
                }
            }
            Activation::Sigmoid => {
                for i in 0..grad_y.data.len() {
                    let yv = y.data[i];
                    grad_x.data[i] = grad_y.data[i] * yv * (1.0 - yv);
                }
            }
            Activation::Tanh => {
                for i in 0..grad_y.data.len() {
                    let yv = y.data[i];
                    grad_x.data[i] = grad_y.data[i] * (1.0 - yv * yv);
                }
            }
            Activation::Identity => {
                grad_x = grad_y.clone();
            }
        }
        grad_x
    }

    fn step(&mut self, _lr: f32) {
        // no parameters
    }

    fn dump(&self, _out: &mut String) {
        // activations are recorded in the network header, not per-layer
    }
}

// ---------- Network ----------

pub struct Network {
    pub arch: Vec<usize>,           // [in, h1, h2, ..., out]
    pub hidden_acts: Vec<Activation>, // length = arch.len() - 2
    pub task: Task,
    pub layers: Vec<Box<dyn Layer>>,
}

impl Network {
    /// Build a fresh randomly-initialized network.
    /// `hidden_acts` has one entry per hidden layer (i.e. `arch.len() - 2`).
    /// The final linear layer outputs raw logits; the task determines the
    /// implicit final activation used at predict-time and inside the loss.
    pub fn new(
        arch: Vec<usize>,
        hidden_acts: Vec<Activation>,
        task: Task,
        seed: u64,
    ) -> Result<Self, String> {
        if arch.len() < 2 {
            return Err(format!("arch must have at least input and output (got {} sizes)", arch.len()));
        }
        let expected_acts = arch.len().saturating_sub(2);
        if hidden_acts.len() != expected_acts {
            return Err(format!(
                "expected {} hidden activations for arch of length {}, got {}",
                expected_acts,
                arch.len(),
                hidden_acts.len()
            ));
        }

        let mut rng = Xorshift::new(seed);
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();
        for i in 0..(arch.len() - 1) {
            layers.push(Box::new(Linear::new(arch[i], arch[i + 1], &mut rng)));
            // Activation after every linear EXCEPT the last
            if i < arch.len() - 2 {
                layers.push(Box::new(ActivationLayer::new(hidden_acts[i])));
            }
        }

        Ok(Self { arch, hidden_acts, task, layers })
    }

    /// Forward pass producing raw logits.
    pub fn forward(&mut self, x: &Matrix) -> Matrix {
        let mut out = x.clone();
        for layer in &mut self.layers {
            out = layer.forward(&out);
        }
        out
    }

    /// Backward pass given the gradient w.r.t. logits.
    pub fn backward(&mut self, grad_logits: &Matrix) {
        let mut grad = grad_logits.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
    }

    pub fn step(&mut self, lr: f32) {
        for layer in &mut self.layers {
            layer.step(lr);
        }
    }

    /// One full training step. Returns the scalar loss.
    pub fn train_step(&mut self, x: &Matrix, targets: &Matrix, lr: f32) -> f32 {
        let logits = self.forward(x);
        let (loss, grad_logits) = compute_loss(&logits, targets, self.task);
        self.backward(&grad_logits);
        self.step(lr);
        loss
    }

    /// Inference. Applies the appropriate final activation given the task.
    /// Binary -> sigmoid (B x 1).
    /// MultiClass -> softmax (B x K).
    /// Regression -> identity.
    pub fn predict(&mut self, x: &Matrix) -> Matrix {
        let logits = self.forward(x);
        match self.task {
            Task::Binary => apply_sigmoid(&logits),
            Task::MultiClass => apply_softmax(&logits),
            Task::Regression => logits,
        }
    }
}

// ---------- Final-activation helpers ----------

fn apply_sigmoid(logits: &Matrix) -> Matrix {
    let mut out = logits.clone();
    for v in &mut out.data {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
    out
}

fn apply_softmax(logits: &Matrix) -> Matrix {
    let b = logits.rows;
    let k = logits.cols;
    let mut out = Matrix::zeros(b, k);
    for i in 0..b {
        let row_start = i * k;
        let mut max_z = f32::NEG_INFINITY;
        for j in 0..k {
            let v = logits.data[row_start + j];
            if v > max_z {
                max_z = v;
            }
        }
        let mut sum_exp = 0.0;
        for j in 0..k {
            let e = (logits.data[row_start + j] - max_z).exp();
            out.data[row_start + j] = e;
            sum_exp += e;
        }
        let inv = 1.0 / sum_exp;
        for j in 0..k {
            out.data[row_start + j] *= inv;
        }
    }
    out
}

// ---------- Loss ----------

/// Returns (loss_value, grad_w.r.t._logits). Gradient is averaged over the
/// batch (i.e., divided by batch size) so that downstream layers don't need to.
pub fn compute_loss(logits: &Matrix, targets: &Matrix, task: Task) -> (f32, Matrix) {
    assert_eq!(logits.rows, targets.rows, "loss: batch size mismatch");
    assert_eq!(logits.cols, targets.cols, "loss: column count mismatch");
    let batch = logits.rows as f32;

    match task {
        Task::Binary => {
            // logits (B x 1), targets (B x 1) with values in {0, 1}.
            // Numerically stable BCE:
            //   loss = max(z, 0) - z*y + log(1 + exp(-|z|))
            //   grad = sigmoid(z) - y
            let mut loss = 0.0;
            let mut grad = Matrix::zeros(logits.rows, logits.cols);
            for i in 0..logits.data.len() {
                let z = logits.data[i];
                let y = targets.data[i];
                let l = z.max(0.0) - z * y + (1.0 + (-z.abs()).exp()).ln();
                loss += l;
                let p = 1.0 / (1.0 + (-z).exp());
                grad.data[i] = (p - y) / batch;
            }
            (loss / batch, grad)
        }
        Task::MultiClass => {
            // logits (B x K), targets (B x K) one-hot.
            // CE(softmax(z), y) = -sum_j y_j log(softmax(z)_j)
            // grad = softmax(z) - y
            let b = logits.rows;
            let k = logits.cols;
            let mut loss = 0.0;
            let mut grad = Matrix::zeros(b, k);
            for i in 0..b {
                let row = i * k;
                let mut max_z = f32::NEG_INFINITY;
                for j in 0..k {
                    let v = logits.data[row + j];
                    if v > max_z {
                        max_z = v;
                    }
                }
                let mut sum_exp = 0.0;
                for j in 0..k {
                    let e = (logits.data[row + j] - max_z).exp();
                    grad.data[row + j] = e; // store exp temporarily
                    sum_exp += e;
                }
                let inv = 1.0 / sum_exp;
                for j in 0..k {
                    let p = grad.data[row + j] * inv;
                    let y = targets.data[row + j];
                    if y > 0.0 {
                        // clamp log to avoid -inf when p is tiny
                        loss -= y * p.max(1e-12).ln();
                    }
                    grad.data[row + j] = (p - y) / batch;
                }
            }
            (loss / batch, grad)
        }
        Task::Regression => {
            // MSE summed over outputs, averaged over batch.
            // loss = sum_{i,j} (z_ij - y_ij)^2 / B
            // grad = 2*(z - y) / B
            let mut loss = 0.0;
            let mut grad = Matrix::zeros(logits.rows, logits.cols);
            for i in 0..logits.data.len() {
                let d = logits.data[i] - targets.data[i];
                loss += d * d;
                grad.data[i] = 2.0 * d / batch;
            }
            (loss / batch, grad)
        }
    }
}
