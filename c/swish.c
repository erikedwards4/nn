//This gets Swish activation function [Ramachandran et al. 2017] for each element of X.

#include <stdio.h>
#include <math.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int swish_s (float *Y, const float *X, const size_t N, const float beta)
{
    if (beta<0.0f) { fprintf(stderr,"error in swish_s: beta must be nonnegative\n"); return 1; }

    if (beta==0.0f)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = 0.5f**X; }
    }
    else if (beta==1.0f)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X/(1.0f+expf(-*X)); }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X/(1.0f+expf(-beta**X)); }
    }

    return 0;
}


int swish_d (double *Y, const double *X, const size_t N, const double beta)
{
    if (beta<0.0) { fprintf(stderr,"error in swish_d: beta must be nonnegative\n"); return 1; }

    if (beta==0.0)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = 0.5**X; }
    }
    else if (beta==1.0)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X/(1.0+exp(-*X)); }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X/(1.0+exp(-beta**X)); }
    }
    
    return 0;
}


int swish_inplace_s (float *X, const size_t N, const float beta)
{
    if (beta<0.0f) { fprintf(stderr,"error in swish_inplace_s: beta must be nonnegative\n"); return 1; }

    if (beta==0.0f)
    {
        for (size_t n=N; n>0u; --n, ++X) { *X *= 0.5f; }
    }
    else if (beta==1.0f)
    {
        for (size_t n=N; n>0u; --n, ++X) { *X /= 1.0f + expf(-*X); }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X) { *X /= 1.0f + expf(-beta**X); }
    }

    return 0;
}


int swish_inplace_d (double *X, const size_t N, const double beta)
{
    if (beta<0.0) { fprintf(stderr,"error in swish_inplace_d: beta must be nonnegative\n"); return 1; }

    if (beta==0.0)
    {
        for (size_t n=N; n>0u; --n, ++X) { *X *= 0.5; }
    }
    else if (beta==1.0)
    {
        for (size_t n=N; n>0u; --n, ++X) { *X /= 1.0 + exp(-*X); }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X) { *X /= 1.0 + exp(-beta**X); }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
