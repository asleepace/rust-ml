//! Row-major dense matrix and a tiny xorshift RNG.
//! Everything is f32. Layout: data[r * cols + c].

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn from_data(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), rows * cols, "data length mismatch");
        Self { rows, cols, data }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        self.data[r * self.cols + c] = v;
    }

    /// C = A * B  where A is (m x k), B is (k x n), C is (m x n).
    /// Loop ordering is i-k-j (cache-friendlier than i-j-k for row-major).
    pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
        assert_eq!(a.cols, b.rows, "matmul shape: ({}x{}) * ({}x{})", a.rows, a.cols, b.rows, b.cols);
        let m = a.rows;
        let k = a.cols;
        let n = b.cols;
        let mut c = Matrix::zeros(m, n);
        for i in 0..m {
            for kk in 0..k {
                let aik = a.data[i * k + kk];
                let b_row = kk * n;
                let c_row = i * n;
                for j in 0..n {
                    c.data[c_row + j] += aik * b.data[b_row + j];
                }
            }
        }
        c
    }

    /// C = A^T * B   where A is (m x k), B is (m x n), C is (k x n).
    /// We never materialize the transpose; instead we walk A by column.
    pub fn matmul_a_t(a: &Matrix, b: &Matrix) -> Matrix {
        assert_eq!(a.rows, b.rows, "matmul_a_t shape: A^T ({}x{}) * B ({}x{})", a.cols, a.rows, b.rows, b.cols);
        let m = a.rows;
        let k = a.cols;
        let n = b.cols;
        let mut c = Matrix::zeros(k, n);
        for ki in 0..k {
            for mi in 0..m {
                let v = a.data[mi * k + ki];
                let b_row = mi * n;
                let c_row = ki * n;
                for j in 0..n {
                    c.data[c_row + j] += v * b.data[b_row + j];
                }
            }
        }
        c
    }

    /// C = A * B^T   where A is (m x k), B is (n x k), C is (m x n).
    pub fn matmul_b_t(a: &Matrix, b: &Matrix) -> Matrix {
        assert_eq!(a.cols, b.cols, "matmul_b_t shape: A ({}x{}) * B^T ({}x{})", a.rows, a.cols, b.cols, b.rows);
        let m = a.rows;
        let k = a.cols;
        let n = b.rows;
        let mut c = Matrix::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                let a_row = i * k;
                let b_row = j * k;
                for kk in 0..k {
                    acc += a.data[a_row + kk] * b.data[b_row + kk];
                }
                c.data[i * n + j] = acc;
            }
        }
        c
    }

    /// In-place: self += rhs (must be same shape).
    pub fn add_inplace(&mut self, rhs: &Matrix) {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        for i in 0..self.data.len() {
            self.data[i] += rhs.data[i];
        }
    }

    /// In-place: each row of self += `bias_row`. `bias_row` must be (1 x cols).
    pub fn add_row_broadcast(&mut self, bias_row: &Matrix) {
        assert_eq!(bias_row.rows, 1);
        assert_eq!(bias_row.cols, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i * self.cols + j] += bias_row.data[j];
            }
        }
    }

    /// Sum each column into a (1 x cols) row vector.
    pub fn sum_rows_to_row(&self) -> Matrix {
        let mut out = Matrix::zeros(1, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.data[j] += self.data[i * self.cols + j];
            }
        }
        out
    }

    /// In-place: self *= scalar.
    pub fn scale(&mut self, s: f32) {
        for v in &mut self.data {
            *v *= s;
        }
    }
}

// ---------- RNG ----------

/// Minimal xorshift64 PRNG. Deterministic given the seed; good enough for weight init.
pub struct Xorshift(u64);

impl Xorshift {
    pub fn new(seed: u64) -> Self {
        // xorshift cannot have state 0
        Self(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        // Take top 24 bits, scale to [0, 1).
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 / (1u32 << 24) as f32
    }

    /// Standard normal via Box-Muller.
    pub fn next_normal(&mut self) -> f32 {
        // Avoid log(0) by clamping u1.
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }
}

/// He initialization (good for ReLU): N(0, sqrt(2/fan_in)).
pub fn he_init(rows: usize, cols: usize, fan_in: usize, rng: &mut Xorshift) -> Matrix {
    let std = (2.0 / fan_in as f32).sqrt();
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..(rows * cols) {
        data.push(rng.next_normal() * std);
    }
    Matrix::from_data(rows, cols, data)
}
