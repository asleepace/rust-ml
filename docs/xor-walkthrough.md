# XOR Neural Net — A Walkthrough From First Principles

The XOR example in `nn-train` is the canonical "smallest interesting neural net" — tiny but contains every essential idea. This walks through what's actually happening, what we're training, and why it works.

## Contents

- [What is XOR?](#what-is-xor)
- [Why XOR is the famous test case](#why-xor-is-the-famous-test-case)
- [What our XOR setup looks like](#what-our-xor-setup-looks-like)
- [What "training" actually does](#what-training-actually-does)
- [What the network is actually computing](#what-the-network-is-actually-computing)
- [What the loss function is](#what-the-loss-function-is)
- [Why backprop works](#why-backprop-works-the-part-that-feels-like-magic)
- [Why your XOR run worked](#why-your-xor-run-worked)
- [What carries over to your real problem](#what-carries-over-to-your-real-problem)

## What is XOR?

XOR ("exclusive or") is a function that takes two binary inputs and returns 1 if exactly one of them is 1, otherwise 0:

```
input        output
(0, 0)    →    0
(0, 1)    →    1
(1, 0)    →    1
(1, 1)    →    0
```

Simple, deterministic, four data points total.

## Why XOR is the famous test case

In 1969, Marvin Minsky and Seymour Papert published *Perceptrons*, which proved that a single-layer perceptron (one neuron, no hidden layer) **cannot** learn XOR. This is geometrically intuitive: a single neuron draws one straight line through the input space and classifies points based on which side they fall on. Look at the four XOR points:

```
y=1 │  (0,1)=1     (1,1)=0
    │
    │  (0,0)=0     (1,0)=1
    └──────────────────────── 
                              x
```

The "1" points are on opposite corners. There's **no straight line** that separates 1s from 0s. You'd need either two lines or a curve. This is what "linearly inseparable" means.

This result essentially killed neural network research for a decade ("AI winter"). The fix — adding a hidden layer with nonlinear activations — wasn't widely understood until the 1980s. So XOR became *the* canary test: if your net can learn XOR, your forward pass, backprop, and nonlinearity are all working. If it can't, something fundamental is broken.

## What our XOR setup looks like

From the code:

```rust
let x = Matrix::from_data(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
let y = Matrix::from_data(4, 1, vec![0.0, 1.0, 1.0, 0.0]);

let mut net = Network::new(
    vec![2, 4, 1],            // architecture
    vec![Activation::Tanh],   // hidden activation
    Task::Binary,              // sigmoid+BCE under the hood
    seed,
)?;
```

**Architecture: `[2, 4, 1]`**
- 2 input neurons (one per input dimension)
- 4 hidden neurons (the "hidden layer")
- 1 output neuron (probability of class 1)

**Total trainable parameters:** 17
- Layer 1 weights: 2×4 = 8 + 4 biases = 12
- Layer 2 weights: 4×1 = 4 + 1 bias = 5

That's it. 17 numbers that we're going to nudge until the net produces the right output.

## What "training" actually does

Training is a search through this 17-dimensional parameter space to find values that make the network's predictions match the truth. The search is guided by **gradient descent**.

Here's the loop, in plain English:

1. Take all 4 input pairs
2. Run them through the net with the current weights → get 4 predictions
3. Compare predictions to truth → compute a "loss" (a single number measuring how wrong we are)
4. Compute the **gradient** of the loss w.r.t. each of the 17 parameters (which way to nudge each one to make loss smaller)
5. Nudge each parameter slightly in the right direction (`weight -= learning_rate * gradient`)
6. Repeat 4000 times

After enough iterations, the loss is tiny and the predictions match.

## What the network is actually computing

Let's walk through one forward pass with random initial weights, conceptually. Take input `(0, 1)`:

**Layer 1** (the hidden layer, 4 neurons):

Each hidden neuron computes: `tanh(w₁·x₁ + w₂·x₂ + b)`

That's a weighted sum of the inputs, plus a bias, squashed through `tanh` to land in (-1, 1).

The crucial bit is the **`tanh`**. Without it, the whole network collapses into a single linear function — multiple linear layers stacked equal one big linear layer (matrix multiplication is associative). Linear functions can only draw straight lines through the input space, which is exactly why a single perceptron fails XOR. The nonlinearity is what gives the network the ability to bend.

You can think of each hidden neuron as a learned feature detector. After training, the four hidden neurons will have specialized into something like:

- "fires when x₁ is high"
- "fires when x₂ is high"
- "fires when both are high"
- "fires when both are low"

(The exact specialization depends on initialization and is rarely this clean, but conceptually that's what's happening.)

**Layer 2** (the output, 1 neuron):

Computes: `sigmoid(w₁·h₁ + w₂·h₂ + w₃·h₃ + w₄·h₄ + b)`

Takes the four hidden activations, weighted-sums them, and squashes the result through sigmoid to land in (0, 1) — interpretable as "probability of class 1".

So the output neuron is asking: *"given what the hidden layer detected, how should I combine those signals into a final answer?"* It might learn something like "fire if 'one is high' detector fires AND 'both are high' detector does NOT fire" — which is exactly XOR.

## What the loss function is

Binary cross-entropy:

```
loss = -[y·log(p) + (1-y)·log(1-p)]
```

where `y` is the true label (0 or 1) and `p` is the predicted probability.

Why this and not just "squared error"?

- If `y=1` and `p=0.99`, loss is tiny. Good.
- If `y=1` and `p=0.01`, loss is huge (`-log(0.01) ≈ 4.6`). Punishes confident wrongness hard.
- BCE pairs naturally with sigmoid output — the gradient simplifies to `(p - y)`, which is numerically clean.

## Why backprop works (the part that feels like magic)

Here's the key insight: **the chain rule of calculus tells us exactly how to compute the gradient of the loss w.r.t. every weight, no matter how deep the network**.

Conceptually, for each weight, we want to answer: "if I nudge this weight up by ε, how much does the final loss change?"

Backpropagation computes this efficiently by working **backwards** from the output:

1. Compute how much the loss changes if the output changes (`∂loss/∂output`)
2. Use that to compute how much the loss changes if each layer-2 weight changes
3. Use that to compute how much the loss changes if each hidden activation changes
4. Use that to compute how much the loss changes if each layer-1 weight changes

Each step is just one application of the chain rule. The "back" in backpropagation is literal: the gradient flows from output back to input. And it's exactly what's happening in `Network::backward` in the code — each layer's `backward` takes the gradient flowing in from the next layer and produces the gradient to send to the previous layer.

Once we have all 17 gradients, we update: `w := w - lr · gradient`. The learning rate `lr=0.1` controls step size. Too small and it takes forever. Too large and the training oscillates or diverges.

## Why your XOR run worked

Your output:

```
epoch     0: loss=0.703498
epoch  3600: loss=0.007570
final:
  (0,0) -> 0.0064  (truth 0)
  (0,1) -> 0.9982  (truth 1)
  (1,0) -> 0.9900  (truth 1)
  (1,1) -> 0.0084  (truth 0)
```

- Loss starts at 0.70, very close to `-log(0.5) ≈ 0.693` — that's the loss of a network making 50/50 random guesses, which is exactly what He-initialized weights with no training would do
- Loss drops to 0.007, meaning the network is putting ~99% probability on the correct answer for every input
- Predictions are all within 1% of truth

This is **proof** that your forward pass, backward pass, weight init, BCE loss, and SGD step are all correct. If any of them were broken:

- Wrong forward pass → predictions would be nonsense
- Wrong backward pass → loss wouldn't decrease (you'd see it stuck near 0.69)
- Bad init → could get stuck in a bad local minimum
- Bad loss → loss might decrease but predictions wouldn't match
- Bad SGD → loss would oscillate or diverge

XOR is small enough to converge from almost any reasonable starting point, but big enough to require every component to be right.

## What carries over to your real problem

Your options direction prediction is the same problem with bigger dimensions:

| | XOR | Options direction |
|---|---|---|
| Input | 2 features | 10+ features |
| Output | 1 number (binary) | 3 numbers (multiclass softmax) |
| Hidden | 4 neurons | 16, 8 neurons |
| Loss | BCE | Cross-entropy |
| Architecture | `[2,4,1]` | `[10,16,8,3]` |
| Why nonlinearity matters | XOR isn't linearly separable | options patterns probably aren't either |

The math, the training loop, the code — identical. Only the shapes change.

The differences that *do* matter for real problems:

- **Data quantity**: XOR has 4 samples and they're all you'll ever need. Options has 30 days, which isn't enough to learn anything.
- **Noise**: XOR is deterministic. Markets are noisy — even with infinite data, you can't perfectly predict next-day direction.
- **Generalization**: With 4 XOR samples, train and test are the same set. With markets, the model has to generalize from past to future, which is much harder.

So XOR teaches you **the mechanism**. It can't teach you what to expect on real, noisy, sparse data — that's the part you'll learn by collecting 6 months of snapshots and seeing how the model behaves. Both pieces are necessary.
