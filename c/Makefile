#@author Erik Edwards
#@date 2017-present
#@license BSD 3-clause

#nn is my own library of C functions for computational neurons in C.

#This is the Makefile used during development of the C functions.
#Since these are not stand-alone programs (no main functions), the only outputs are object files to obj.

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

CFLAGS=$(WFLAG) $(STD) -O3 -march=native -fPIC


All: all
all: IN CELL OUT Clean


#IN: input side of neurons/layers
IN: Linear Conv

#linear applies weights only; affine applies weights and biases.
Linear: linear affine
linear: linear.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
affine: affine.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)

#Conv: convolution
Conv: #conv1d conv2d conv1d_fft conv2d_fft maxpool avgpool
conv1d: conv1d.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
conv2d: conv2d.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
conv1d_fft: conv1d_fft.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
conv2d_fft: conv2d_fft.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
maxpool: maxpool.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
avgpool: avgpool.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)


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
Output_Act: signum step smoothstep logistic tanh atan asinh gudermann sqnl isru isrlu erf gelu relu prelu elu selu silu swish softclip softplus softsign plu sin
signum: signum.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
step: step.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
smoothstep: smoothstep.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
logistic: logistic.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
tanh: tanh.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
atan: atan.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
asinh: asinh.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
gudermann: gudermann.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
sqnl: sqnl.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
isru: isru.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
isrlu: isrlu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
erf: erf.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
gelu: gelu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
relu: relu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
prelu: prelu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
elu: elu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
selu: selu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
silu: silu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
swish: swish.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
softclip: softclip.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
softplus: softplus.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
softsign: softsign.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
plu: plu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
sin: sin.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)

#Activation functions applied layer-wise
#(Above Output_Acts can be applied without modification to layers, so are not repeated here)
Other_Act: maxout softmax betamax
maxout: maxout.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
softmax: softmax.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
betamax: betamax.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)

#CN_Neurons: output side of various neurons from history of computational neuroscience
# CN_Neurons: lapicque hill hodgkin_huxley 

#DEQ_Neurons: output side of various neurons based on systems of differential equations
# DEQ_Neurons: vanderpol fitzhugh nagumo izhikevich wang



#If want to make a library archive for C++ use, this works with clang++
#ar crs ../lib/libnn.a obj/*.o


#make clean
Clean: clean
clean:
	find ./obj -type f -name *.o | xargs rm -f
	rm -f *.s tmp.*
