//This gets randomized ReLU (RReLU) activation function for each element of X.
//Xu B, Wang N, Chen T, Li M. 2015. Empirical evaluation of rectified activations in convolutional network. arXiv. 1505.00853[v.2]: 1-5.

//For each element: y = alpha*x,  if x<0. \n";
//                  y = x,        if x>=0. \n
//where alpha is drawn from a uniform distribution in [lower upper]

//The random numbers are generated here directly using method of the PCG random library.

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int rrelu_s (float *Y, const float *X, const size_t N, const float lower, const float upper)
{
    if (lower>=upper) { fprintf(stderr,"error in rrelu_s: upper must be > lower\n"); return 1; }

    //Init random num generator
    const float sc = upper - lower;
    float alpha;
    uint32_t r, xorshifted, rot;
    uint64_t state = 0u;
    const uint64_t mul = 6364136223846793005u;
    const uint64_t inc = ((uint64_t)(&state) << 1u) | 1u;
    struct timespec ts;

    //Init generator state
    if (timespec_get(&ts,TIME_UTC)==0) { fprintf(stderr, "error in rrelu_s: timespec_get.\n"); perror("timespec_get"); return 1; }
    state = (uint64_t)(ts.tv_nsec^ts.tv_sec) + inc;

    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        if (*X<0.0f)
        {
            state = state*mul + inc;
            xorshifted = (uint32_t)(((state >> 18u) ^ state) >> 27u);
            rot = state >> 59u;
            r = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
            alpha = ldexp((float)r,-32)*sc + lower;
            *Y = alpha * *X;
        }
        else
        {
            *Y = *X;
        }
    }

    return 0;
}


int rrelu_d (double *Y, const double *X, const size_t N, const double lower, const double upper)
{
    if (lower>=upper) { fprintf(stderr,"error in rrelu_d: upper must be > lower\n"); return 1; }

    //Init random num generator
    const double sc = upper - lower;
    double alpha;
    uint32_t r, xorshifted, rot;
    uint64_t state = 0u;
    const uint64_t mul = 6364136223846793005u;
    const uint64_t inc = ((uint64_t)(&state) << 1u) | 1u;
    struct timespec ts;

    //Init generator state
    if (timespec_get(&ts,TIME_UTC)==0) { fprintf(stderr, "error in rrelu_d: timespec_get.\n"); perror("timespec_get"); return 1; }
    state = (uint64_t)(ts.tv_nsec^ts.tv_sec) + inc;

    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        if (*X<0.0f)
        {
            state = state*mul + inc;
            xorshifted = (uint32_t)(((state >> 18u) ^ state) >> 27u);
            rot = state >> 59u;
            r = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
            alpha = ldexp((double)r,-32)*sc + lower;
            *Y = alpha * *X;
        }
        else
        {
            *Y = *X;
        }
    }

    return 0;
}


int rrelu_inplace_s (float *X, const size_t N, const float lower, const float upper)
{
    if (lower>=upper) { fprintf(stderr,"error in rrelu_inplace_s: upper must be > lower\n"); return 1; }

    //Init random num generator
    const float sc = upper - lower;
    float alpha;
    uint32_t r, xorshifted, rot;
    uint64_t state = 0u;
    const uint64_t mul = 6364136223846793005u;
    const uint64_t inc = ((uint64_t)(&state) << 1u) | 1u;
    struct timespec ts;

    //Init generator state
    if (timespec_get(&ts,TIME_UTC)==0) { fprintf(stderr, "error in rrelu_inplace_s: timespec_get.\n"); perror("timespec_get"); return 1; }
    state = (uint64_t)(ts.tv_nsec^ts.tv_sec) + inc;

    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X<0.0f)
        {
            state = state*mul + inc;
            xorshifted = (uint32_t)(((state >> 18u) ^ state) >> 27u);
            rot = state >> 59u;
            r = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
            alpha = ldexp((float)r,-32)*sc + lower;
            *X *= alpha;
        }
    }

    return 0;
}


int rrelu_inplace_d (double *X, const size_t N, const double lower, const double upper)
{
    if (lower>=upper) { fprintf(stderr,"error in rrelu_inplace_d: upper must be > lower\n"); return 1; }

    //Init random num generator
    const double sc = upper - lower;
    double alpha;
    uint32_t r, xorshifted, rot;
    uint64_t state = 0u;
    const uint64_t mul = 6364136223846793005u;
    const uint64_t inc = ((uint64_t)(&state) << 1u) | 1u;
    struct timespec ts;

    //Init generator state
    if (timespec_get(&ts,TIME_UTC)==0) { fprintf(stderr, "error in rrelu_inplace_d: timespec_get.\n"); perror("timespec_get"); return 1; }
    state = (uint64_t)(ts.tv_nsec^ts.tv_sec) + inc;

    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X<0.0f)
        {
            state = state*mul + inc;
            xorshifted = (uint32_t)(((state >> 18u) ^ state) >> 27u);
            rot = state >> 59u;
            r = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
            alpha = ldexp((double)r,-32)*sc + lower;
            *X *= alpha;
        }
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
