#@author Erik Edwards
#@date 2017-present
#@license BSD 3-clause

#nn is my own library of C functions for NNs (neural networks) and computational neurons.

#This is the Makefile used during development of the C functions.
#Since these are not stand-alone programs (no main functions), the only outputs are object files to obj.
#It is not necessary to run this when making the C++ command-line tools.

#Project is organized as:
#IN  : input side of neurons and layers of neurons
#CELL: middle part of neurons and layers of neurons
#OUT : output side of neurons and layers of neurons

SHELL=/bin/bash
CC=clang

ifeq ($(CC),$(filter $(CC),clang gcc))
	STD=-std=c11
else
	STD=-std=gnu++14
endif

ifeq ($(CC),clang++)
	STD=-std=c++11
	WFLAG=-Weverything -Wno-c++98-compat -Wno-gnu-imaginary-constant -Wno-deprecated -Wno-old-style-cast
else
	WFLAG=-Wall -Wextra
endif

CFLAGS=$(WFLAG) $(STD) -O3 -ffast-math -march=native -mfpmath=sse -fPIC


All: all
all: Dirs IN CELL OUT Lib Clean

Dirs:
	mkdir -pm 777 obj


#IN: input side of neurons/layers
IN: Transform Conv Pool

#Transform: linear (weights), affine (weights+biases) and bilinear
Transform: linear linear.cblas affine affine.cblas bilinear #bilinear.cblas
linear: linear.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
linear.cblas: linear.cblas.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
affine: affine.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
affine.cblas: affine.cblas.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
bilinear: bilinear.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
bilinear.cblas: bilinear.cblas.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)

#Conv: convolution
#conv1 versions have no groups and no dilation.
#conv1g versions have groups but no dilation.
#conv1d versions have groups and dilation.
#The .cblas versions use CBLAS for dot products.
#The .torch versions use PyTorch shape conventions for inputs/outputs.
Conv: conv1 conv1g conv1d conv1.cblas conv1g.cblas conv1d.cblas conv1.torch conv1g.torch conv1d.torch
conv1: conv1.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
conv1g: conv1g.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
conv1d: conv1d.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
conv1.cblas: conv1.cblas.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
conv1g.cblas: conv1g.cblas.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
conv1d.cblas: conv1d.cblas.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
conv1.torch: conv1.torch.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
conv1g.torch: conv1g.torch.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
conv1d.torch: conv1d.torch.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)

#Pool: pooling
#maxpool1d and avgpool1d have dilation; maxpool1 and avgpool1 do not.
Pool: maxpool1 maxpool1d avgpool1 avgpool1d lppool1 lppool1d #maxipool1 maxipool1d
maxpool1: maxpool1.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
maxpool1d: maxpool1d.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
maxipool1: maxipool1.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
maxipool1d: maxipool1d.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
avgpool1: avgpool1.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
avgpool1d: avgpool1d.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
lppool1: lppool1.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
lppool1d: lppool1d.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)


#CELL: middle part (~soma) of neurons/layers
CELL: Basic Model RNN

Basic: identity integrate fir 
identity: identity.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
integrate: integrate.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
fir: fir.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)

Model: fukushima fukushima2 hopfield grossberg grossberg2
fukushima: fukushima.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
fukushima2: fukushima2.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
hopfield: hopfield.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
grossberg: grossberg.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
grossberg2: grossberg2.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)

RNN: elman jordan gru_min gru_min2 gru gru3 lstm lstm4 lstm_peephole lstm_peephole4
elman: elman.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
jordan: jordan.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
gru_min: gru_min.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
gru_min2: gru_min2.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
gru: gru.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
gru3: gru3.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
lstm: lstm.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
lstm4: lstm4.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
lstm_peephole: lstm_peephole.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
lstm_peephole4: lstm_peephole4.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)



#OUT: output side of neurons/layers
OUT: Output_Act Other_Act

#Output Activation functions
#These are all element-wise static nonlinearities, so apply to neurons or layers
Output_Act: signum step smoothstep logistic sigmoid logsigmoid hardsigmoid hardshrink softshrink tanh tanhshrink hardtanh atan asinh gudermann sqnl isru isrlu erf gelu gelu_new relu relu6 prelu rrelu elu celu selu silu swish hardswish mish softclip softplus softsign plu sin threshold
signum: signum.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
step: step.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
smoothstep: smoothstep.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
logistic: logistic.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
sigmoid: sigmoid.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
logsigmoid: logsigmoid.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
hardsigmoid: hardsigmoid.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
hardshrink: hardshrink.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
softshrink: softshrink.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
tanh: tanh.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
tanhshrink: tanhshrink.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
hardtanh: hardtanh.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
atan: atan.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
asinh: asinh.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
gudermann: gudermann.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
sqnl: sqnl.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
isru: isru.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
isrlu: isrlu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
erf: erf.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
gelu: gelu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
gelu_new: gelu_new.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
relu: relu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
relu6: relu6.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
prelu: prelu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
rrelu: rrelu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
leaky_relu: leaky_relu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
elu: elu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
celu: celu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
selu: selu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
silu: silu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
swish: swish.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
hardswish: hardswish.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
mish: mish.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
softclip: softclip.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
softplus: softplus.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
softsign: softsign.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
plu: plu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
sin: sin.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
threshold: threshold.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)

#Activation functions applied layer-wise
#(Above Output_Acts can be applied without modification to layers, so are not repeated here)
Other_Act: maxout softmax softmin log_softmax betamax
maxout: maxout.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
softmax: softmax.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
softmin: softmin.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
log_softmax: log_softmax.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
betamax: betamax.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)

#CN_Neurons: output side of various neurons from history of computational neuroscience
# CN_Neurons: lapicque hill hodgkin_huxley 

#DEQ_Neurons: output side of various neurons based on systems of differential equations
# DEQ_Neurons: vanderpol fitzhugh nagumo izhikevich wang



#If want to make a library archive for C++ use, this works with clang++
Lib: lib
lib:
	ar crs ../lib/libnn.a obj/*.o
	gcc -shared -o ../lib/libnn.so ../lib/libnn.a


#make clean
Clean: clean
clean:
	find ./obj -type f -name *.o | xargs rm -f
	rm -f *.s tmp.*
