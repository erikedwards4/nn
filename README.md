# nn

nn: code for neural networks (NNs) computational neurons and in C

================================================

nn is a set of C functions and associated command-line tools in C++,
that implement model neurons of several types from computational neuroscience,
and typical neurons, activation functions, etc. from machine-learning NNs.

Either single neurons or layers of neurons can be used.

These are organized around "components", allowing modularity, flexibility and efficiency.
Each neuron or neural layer has 3 components in general:
1 (IN): the input side, usually an affine transformation (i.e., weights and biases)
2 (CELL): the middle part, often an identity, or complicated model such as LSTM recursions.
3 (OUT): the output side, usually an activation function (ReLU, etc.)

For example, a "ReLU layer" from any usual framework (PyTorch, TF, etc.) is:
1 (IN): affine transformation (weights and biases)
2 (CELL): identity (no change)
3 (OUT): ReLU activation function (static nonlinearity)

An "LSTM layer" is typically:
1 (IN): affine transformation (weights and biases)
2 (CELL): LSTM recursions
3 (OUT): tanh activation function (static nonlinearity)

The command-line programs are written in C++ with a consistent style and interface.
The low-level functions are written in C for very fast performance.

The C functions are meant only for the developer; the end-user only uses the C++ command-line tools.
The interface to each C function is BLAS-like, meaning that one specifies the input and/or output dimensions,
the matrix order as row-major or column-major, and so on.

The C++ command-line programs are written in a consistent style that was developed for command-line tools in general.
All of these command-line tools use argtable2 (http://argtable.sourceforge.net/) for parsing
inputs and option flags. All of them allow -h (--help) as a flag to give description and usage info.

Input/output is supported for NumPy tensors and several C++ tensor formats:
Armadillo (http://arma.sourceforge.net/), ArrayFire (https://arrayfire.com/), a minimal format
for Eigen (http://eigen.tuxfamily.org/) and NumPy (https://numpy.org/).
The later means that input/output can be piped directly to snippets of Python code that use NumPy!


## Dependencies
Requires argtable2, openBLAS.
For Ubuntu, these are available by apt-get:
```
sudo apt-get install libargtable2-0 libblas3 libopenblas-base
```


## Installation
```console
cd /opt/codee
git clone https://github.com/erikedwards4/nn
cd /opt/codee/nn
make
```

Each C function can also be compiled separately; see c subdirectory Makefile for details.
To make an archive library:
```console
cd /opt/codee/nn/c
make libnn.a CC=clang
```
This creates /opt/codee/nn/lib/libnn.a with all of the C object files.
This could be useful if trying to use the C functions in other applications.
Change clang to clang++ to compile for use with C++ applications.


## Usage
See each resulting command-line tool for help (use -h or --help option).
For example:
```console
/opt/codee/math/bin/log2 --help
```


## List of functions
All: IN CELL OUT

IN: Linear Conv
Linear: linear affine
Conv: conv conv_fft maxpool avgpool

CELL: Basic Model RNN
Basic: identity integrate fir
Model: fukushima fukushima2 hopfield grossberg grossberg2
RNN: elman jordan gru_min gru_min2 gru gru3 lstm lstm4 lstm_peephole lstm_peephole4

OUT: Static_Act Other_Act
Static_Act: step smoothstep logistic tanh atan asinh gudermann sqnl isru isrlu erf gelu relu prelu elu selu softclip softplus softsign plu silu swish sin
Other_Act: maxout softmax betamax


## Contributing
This is currently only to view the project in progress.


## License
[BSD 3-Clause](https://choosealicense.com/licenses/bsd-3-clause/)

