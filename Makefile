#@author Erik Edwards
#@date 2018-present
#@license BSD 3-clause

#nn is my own library of functions for neural networks (NNs)
#and computational neurons in C and C++.
#This is the makefile for the C++ wrappers to use at the command-line.

#Project is organized as:
#IN : input side (~synapses+dendrites) for neurons and layers of neurons
#CELL: middle part (~soma) for neurons and layers of neurons
#OUT: output side (~axon hillock) for neurons and layers of neurons

#That is, each neuron or each layer has 3 "components" -- IN, CELL and OUT.
#For most NNs, the IN side is an affine transform (weights and biases),
#the CELL is just the identity, and the OUT side is a static nonlinearity.
#Thus, a "ReLU neuron" is an affine transform (IN) followed by a ReLU nonlinearity (OUT).
#Here, the IN/OUT sides are decoupled to allow better flexibility and coding focus/modularity.
#The CELL part is required for RNNs (LSTMs, etc.) and for model neurons of computational neurosci.

SHELL=/bin/bash
ss=../util/bin/srci2src
CC=g++
CCC=g++-7

ifeq ($(CC),clang++)
	STD=-std=c++11
	WFLAG=-Weverything -Wno-c++98-compat -Wno-old-style-cast -Wno-gnu-imaginary-constant
else
	STD=-std=gnu++14
	WFLAG=-Wall -Wextra
endif

INCLS=-Ic -I../util
CFLAGS=$(WFLAG) $(STD) -O3 -ffast-math -march=native -mfpmath=sse $(INCLS)
CCFLAGS=-Wall -Wextra -std=gnu++14 -O3 -ffast-math -march=native -mfpmath=sse $(INCLS)
#LIBS=-largtable2 -lopenblas -llapacke -llapack -lfftw3f -lfftw3 -lm


All: all
all: Dirs IN CELL OUT Clean
	rm -f 7 obj/*.o

Dirs:
	mkdir -pm 777 bin obj


#IN: input side of neurons (~dendrites)
#The great majority of neuron models use weights+biases (affine) or convolution.
#Later, this could include biologically-motivated dendrites and synapses.
IN: Transform Conv Pool

#Transform
Transform: linear linear.cblas affine affine.cblas bilinear #bilinear.cblas
linear: srci/linear.cpp c/linear.c
	$(ss) -tvd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
linear.cblas: srci/linear.cblas.cpp c/linear.cblas.c
	$(ss) -tvd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lopenblas -lm
affine: srci/affine.cpp c/affine.c
	$(ss) -tvd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
affine.cblas: srci/affine.cblas.cpp c/affine.cblas.c
	$(ss) -tvd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lopenblas -lm
bilinear: srci/bilinear.cpp c/bilinear.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
bilinear.cblas: srci/bilinear.cblas.cpp c/bilinear.cblas.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CCC) -c src/$@.cpp -oobj/$@.o $(CCFLAGS); $(CCC) obj/$@.o -obin/$@ -largtable2 -lopenblas -lm

#Conv: convolution
#conv1 and conv1.cblas do not have dilation; conv1d and conv1d.cblas do.
Conv: conv1 conv1.cblas conv1d conv1d.cblas
conv1: srci/conv1.cpp c/conv1.c
	$(ss) -tvd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
conv1.cblas: srci/conv1.cblas.cpp c/conv1.cblas.c
	$(ss) -tvd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lopenblas -lm
conv1d: srci/conv1d.cpp c/conv1d.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
conv1d.cblas: srci/conv1d.cblas.cpp c/conv1d.cblas.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CCC) -c src/$@.cpp -oobj/$@.o $(CCFLAGS); $(CCC) obj/$@.o -obin/$@ -largtable2 -lopenblas -lm

#Pool: pooling
#maxpool1 and avgpool1 do not have dilation; maxpool1d and avgpool1d do.
Pool: maxpool1 maxpool1d maxipool1 maxipool1d avgpool1 avgpool1d lppool1 lppool1d
maxpool1: srci/maxpool1.cpp c/maxpool1.c
	$(ss) -tvd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
maxpool1d: srci/maxpool1d.cpp c/maxpool1d.c
	$(ss) -tvd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
maxipool1: srci/maxpool1.cpp c/maxpool1.c
	$(ss) -tvd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
maxipool1d: srci/maxpool1d.cpp c/maxpool1d.c
	$(ss) -tvd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
avgpool1: srci/avgpool1.cpp c/avgpool1.c
	$(ss) -tvd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
avgpool1d: srci/avgpool1d.cpp c/avgpool1d.c
	$(ss) -tvd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
lppool1: srci/lppool1.cpp c/lppool1.c
	$(ss) -tvd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
lppool1d: srci/lppool1d.cpp c/lppool1d.c
	$(ss) -tvd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm


#CELL: middle part of neurons (~soma)
#For most NNs in ML, this is just an identity (so can be also be skipped).
#However, for RNNs (LSTMs, etc.), this is the main part of the neuron (or layer of neurons).
#For computational neuroscience, many other possibilities exist, and a few critical ones are implemented here.
CELL: Basic Model RNN

Basic: identity integrate fir
identity: srci/identity.cpp c/identity.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
integrate: srci/integrate.cpp c/integrate.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
fir: srci/fir.cpp c/fir.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lopenblas

Model: fukushima fukushima2 hopfield grossberg grossberg2
fukushima: srci/fukushima.cpp c/fukushima.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lopenblas
fukushima2: srci/fukushima2.cpp c/fukushima2.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
hopfield: srci/hopfield.cpp c/hopfield.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
grossberg: srci/grossberg.cpp c/grossberg.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
grossberg2: srci/grossberg2.cpp c/grossberg2.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm

RNN: elman jordan gru_min gru_min2 gru gru3 lstm lstm4 lstm_peephole lstm_peephole4
elman: srci/elman.cpp c/elman.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lopenblas -lm
jordan: srci/jordan.cpp c/jordan.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lopenblas -lm
gru_min: srci/gru_min.cpp c/gru_min.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lopenblas -lm
gru_min2: srci/gru_min2.cpp c/gru_min2.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lopenblas -lm
gru: srci/gru.cpp c/gru.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lopenblas -lm
gru3: srci/gru3.cpp c/gru3.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lopenblas -lm
lstm: srci/lstm.cpp c/lstm.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lopenblas -lm
lstm4: srci/lstm4.cpp c/lstm4.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lopenblas -lm
lstm_peephole: srci/lstm_peephole.cpp c/lstm_peephole.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lopenblas -lm
lstm_peephole4: srci/lstm_peephole4.cpp c/lstm_peephole4.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lopenblas -lm



#OUT: output side of neurons (~axon hillock and axon)
OUT: Static_Act Other_Act #CN_Neurons DEQ_Neurons

#Output Activation functions
#These are all element-wise static nonlinearities, so apply without modification to single neurons or to layers of neurons.
Static_Act: step smoothstep logistic logsigmoid tanh tanhshrink hardtanh atan asinh gudermann sqnl isru isrlu erf gelu gelu_new relu relu6 prelu elu celu selu softclip softplus softsign plu silu swish hardswish mish sin
step: srci/step.cpp c/step.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
signum: srci/signum.cpp c/signum.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
smoothstep: srci/smoothstep.cpp c/smoothstep.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
logistic: srci/logistic.cpp c/logistic.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
logsigmoid: srci/logsigmoid.cpp c/logsigmoid.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
tanh: srci/tanh.cpp c/tanh.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
tanhshrink: srci/tanhshrink.cpp c/tanhshrink.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
hardtanh: srci/hardtanh.cpp c/hardtanh.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
atan: srci/atan.cpp c/atan.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
asinh: srci/asinh.cpp c/asinh.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
gudermann: srci/gudermann.cpp c/gudermann.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
sqnl: srci/sqnl.cpp c/sqnl.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
isru: srci/isru.cpp c/isru.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
isrlu: srci/isrlu.cpp c/isrlu.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
erf: srci/erf.cpp c/erf.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
gelu: srci/gelu.cpp c/gelu.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
gelu_new: srci/gelu_new.cpp c/gelu_new.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
relu: srci/relu.cpp c/relu.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
relu6: srci/relu6.cpp c/relu6.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
prelu: srci/prelu.cpp c/prelu.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
leaky_relu: srci/leaky_relu.cpp c/leaky_relu.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
elu: srci/elu.cpp c/elu.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
celu: srci/celu.cpp c/celu.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
selu: srci/selu.cpp c/selu.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
softclip: srci/softclip.cpp c/softclip.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
softplus: srci/softplus.cpp c/softplus.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
softsign: srci/softsign.cpp c/softsign.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
plu: srci/plu.cpp c/plu.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
silu: srci/silu.cpp c/silu.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
swish: srci/swish.cpp c/swish.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
hardswish: srci/hardswish.cpp c/hardswish.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2
mish: srci/mish.cpp c/mish.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
sin: srci/sin.cpp c/sin.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm


#Other OUT Activation functions 
#Softmax is inherently layer-wise (output of any single neuron depends on the whole layer).
#Maxout can be applied without modification to layers or single neurons, but it is not a static nonlinearity.
Other_Act: maxout softmax betamax
maxout: srci/maxout.cpp c/maxout.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lopenblas -lm
softmax: srci/softmax.cpp c/softmax.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm
betamax: srci/betamax.cpp c/betamax.c
	$(ss) -vd srci/$@.cpp > src/$@.cpp; $(CC) -c src/$@.cpp -oobj/$@.o $(CFLAGS); $(CC) obj/$@.o -obin/$@ -largtable2 -lm



#CN_Neurons: output side of various neurons/layers from history of computational neuroscience
#CN_Neurons: lapicque hill hodgkin_huxley 

#DEQ_Neurons: output side of various neurons/layers based on systems of differential equations
#DEQ_Neurons: vanderpol fitzhugh nagumo izhikevich wang


#make clean
Clean: clean
clean:
	find ./obj -type f -name *.o | xargs rm -f
	rm -f 7 X* Y* x* y*
