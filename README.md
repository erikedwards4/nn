# openN

openN: an open-source toolkit for neurons and neural networks in C

Erik Edwards (erik.edwards4@gmail.com)

================================================

openN is a set of command-line tools that implement model neurons of many types,
and also layers of these model neurons. Layers can be composed to form NNs.

The command-line programs are written in C++ with a consistent style and interface.
The low-level functions themselves are written in C, using openBLAS (very fast performance).

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
Requires argtable2, openBLAS, LAPACKE and FFTW3.
For Ubuntu, these are available by apt-get:
```
sudo apt-get install libargtable2-0 libblas3 libopenblas-base liblapack3 liblapacke fftw3
```


## Installation
```
cd /opt
git clone https://github.com/erikedwards4/openn
cd /opt/openn
make
```

Each C function can also be compiled separately; see c subdirectory Makefile for details.
To make an archive library, do:
```
cd /opt/openvoice/c
make libopenvoice.a CC=clang
```
This creates /opt/openvoice/lib/libopenvoice.a with all of the C object files.
This could be useful if trying to use the C functions in other applications.
Change clang to clang++ to compile for use with C++ applications.


## Usage
See each resulting command-line tool for help (use -h or --help option).
For example:
```
/opt/openn/bin/lstm --help
```


## Contributing
This is currently only to view the project in progress.


## License
[BSD 3-Clause](https://choosealicense.com/licenses/bsd-3-clause/)

