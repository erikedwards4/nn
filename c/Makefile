#@author Erik Edwards
#@date 2017-2020

#openN is my own library of C functions for computational neurons in C.

#This is the Makefile used during development of the C functions.
#Since these are not stand-alone programs (no main functions),
#the only outputs are object files to obj.

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

CFLAGS=$(WFLAG) -O3 $(STD) -march=native -fopenmp -fPIC


#This works with clang++
libopenn.a: all
	ar crs ../lib/libopenn.a obj/*.o


all: Nonlin

#Various static nonlinearities used in NNs and computational neuroscience
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

Acts: identity step logistic relu
identity: identity.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
step: step.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
logistic: logistic.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)
relu: relu.c; $(CC) -c $@.c -o obj/$@.o $(CFLAGS)


clean:
	find ./obj -type f -name *.o | xargs rm -f

