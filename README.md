# micrograd from scratch

A tiny scalar-valued autograd engine and a small neural network library, built from scratch in Python. It implements reverse-mode automatic differentiation (backpropagation) over a dynamically built computation graph, and uses it to train a multi-layer perceptron.

## What's implemented

**Autograd engine (`Value`)**
- A `Value` object wraps a scalar and tracks its gradient, the operation that produced it, and its child nodes
- Operations with their local derivatives: addition, multiplication, power, division, subtraction, negation, `exp`, `tanh` and `relu`
- Works with plain Python numbers on either side (`2 * a`, `a + 1`)
- `backward()` builds a topological ordering of the graph and applies the chain rule from the output back to every input
- Gradients accumulate (`+=`), so a node used more than once receives the sum of its gradients
- Computation graph visualisation with Graphviz

**Neural network**
- `Neuron`: weighted sum of inputs plus bias, passed through `tanh`
- `Layer`: a list of neurons
- `MLP`: a stack of layers, with `parameters()` exposing every weight and bias for training

## Gradient check against PyTorch

The same single neuron (`tanh(x1*w1 + x2*w2 + b)`) was built in both micrograd and PyTorch, and the gradients match:

| Parameter | Gradient |
| --- | --- |
| x1 | -1.5 |
| x2 | 0.5 |
| w1 | 1.0 |
| w2 | 0.0 |

## Tests

`test_engine.py` builds two larger expressions (using `+`, `*`, `**`, `/`, `-`, `relu`, and numbers on either side) in both micrograd and PyTorch, and checks that the forward values and gradients match.

```
pytest
```

## Training result

An `MLP(3, [4, 4, 1])` was trained on a small dataset of 4 examples using mean squared error loss and gradient descent (learning rate 0.01), resetting gradients to zero before every backward pass.

**Loss: 6.99 → 0.013**

| Target | Prediction after training |
| --- | --- |
| 1.0 | 0.944 |
| -1.0 | -0.938 |
| -1.0 | -0.952 |
| 1.0 | 0.939 |

## Requirements

```
torch      # only for the gradient checks and tests
graphviz   # for drawing the computation graph
pytest     # for running the tests
```

## Acknowledgements

This project was built while following Andrej Karpathy's lecture **"The spelled-out intro to neural networks and backpropagation: building micrograd"**, the first lecture of his [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) series. The design follows his original [micrograd](https://github.com/karpathy/micrograd). The graph-drawing utility is taken from the lecture, and the tests in `test_engine.py` are adapted from his original repo. Huge thanks to him for making this material freely available.