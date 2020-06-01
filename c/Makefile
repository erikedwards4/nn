#@author Erik Edwards
#@date 2017-2020

#openN is my own library of C functions for computational neurons in C.

#This is the Makefile used during development of the C functions.
#Since these are not stand-alone programs (no main functions), the only outputs are object files to obj.

#Project is organized as:
#Math  : some useful math functions
#Input : input side of neurons
#INPUT : input side for layers of neurons
#Output: output side of neurons
#OUTPUT: output side for layers of neurons

SHELL=/bin/bash

CC=clang++

ifeq ($(CC),$(filter $(CC),clang gcc))
	STD=-std=c11
else
	STD=-std=gnu++14
endif

ifeq ($(CC),clang++)
	STD=-std=c++11
	WFLAG=-Weverything -Wno-c++98-compat -Wno-padded -Wno-old-style-cast -Wno-gnu-imaginary-constant -Wno-deprecated
else
	WFLAG=-Wall -Wextra
endif

CFLAGS=$(WFLAG) -O3 $(STD) -march=native -fPIC


#This works with clang++
libopenn.a: all
	ar crs ../lib/libopenn.a obj/*.o


all: Math Input INPUT Output OUTPUT


#Math: some useful math functions
Math: Nonlin

#Nonlin: various static nonlinearities used in NNs and computational neuroscience
#Each of these is applied element-wise to the input X.
Nonlin: abs square sqrt cbrt log log2 log10 pow exp
abs: abs.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
square: square.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
sqrt: sqrt.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
cbrt: cbrt.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
log: log.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
log2: log2.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
log10: log10.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
pow: pow.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
exp: exp.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)


#Input: input side of neurons/layers
#For now, only the usual weights and bias (wb) is implemented
Input: WB #WV_filt
WB: WB.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)


#Output: output side of neurons/layers
Output: Output_Acts Layer_Acts ML_Neurons CN_Neurons

#Output Activation functions
#These are all element-wise static nonlinearities, so apply to neurons or layers
Output_Acts: identity step smoothstep logistic tanh atan asinh gudermann sqnl isru isrlu erf gelu relu prelu elu selu silu swish softclip softplus softsign plu
identity: identity.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
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

#Activation functions applied layer-wise
#Recall that above Output_Acts can be applied without modification to layers, so are not repeated here
Layer_Acts: maxout softmax
maxout: maxout.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
softmax: softmax.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)

#ML_Neurons: output side of various neurons/layers from history of ML and pattern recognit
#Note that the input must come from WB for the full neuron as usually described
ML_Neurons: perceptron elman jordan gru_min gru_full lstm lstm_peephole
perceptron: perceptron.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
elman: elman.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
jordan: jordan.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
gru_min: gru_min.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
gru_full: gru_full.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
lstm: lstm.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
lstm_peephole: lstm_peephole.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)

#CN_Neurons: output side of various neurons/layers from history of computational neuroscience
CN_Neurons: lapicque hill hodgkin_huxley 

#DEQ_Neurons: output side of various neurons/layers based on systems of differential equations
DEQ_Neurons: vanderpol fitzhugh nagumo izhikevich wang


clean:
	find ./obj -type f -name *.o | xargs rm -f

