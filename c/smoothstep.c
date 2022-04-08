//This gets the smoothstep activation function for each element of X.
//Only p=0 (clamp) and p=1 (Hermite interpolation of clamp) supported here.

#include <stdio.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int smoothstep_s (float *Y, const float *X, const size_t N, const int p)
{
    if (p==0)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0f) ? 0.0f : (*X>1.0f) ? 1.0f : *X; }
    }
    else if (p==1)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y)
        {
            if (*X<0.0f) { *Y = 0.0f; }
            else if (*X>1.0f) { *Y = 1.0f; }
            else { *Y = *X**X*(3.0f-2.0f**X); }
        }
    }
    else
    {
        fprintf(stderr,"error in smoothstep_s: n must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int smoothstep_d (double *Y, const double *X, const size_t N, const int p)
{
    if (p==0)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0) ? 0.0 : (*X>1.0) ? 1.0 : *X; }
    }
    else if (p==1)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y)
        {
            if (*X<0.0) { *Y = 0.0; }
            else if (*X>1.0) { *Y = 1.0; }
            else { *Y = *X * *X * (3.0-2.0**X); }
        }
    }
    else
    {
        fprintf(stderr,"error in smoothstep_d: n must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


int smoothstep_inplace_s (float *X, const size_t N, const int p)
{
    if (p==0)
    {
        for (size_t n=N; n>0u; --n, ++X)
        {
            if (*X<0.0f) { *X = 0.0f; }
            else if (*X>1.0f) { *X = 1.0f; }
        }
    }
    else if (p==1)
    {
        for (size_t n=N; n>0u; --n, ++X)
        {
            if (*X<0.0f) { *X = 0.0f; }
            else if (*X>1.0f) { *X = 1.0f; }
            else { *X *= *X * (3.0f-2.0f**X); }
        }
    }
    else
    {
        fprintf(stderr,"error in smoothstep_inplace_s: n must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int smoothstep_inplace_d (double *X, const size_t N, const int p)
{
    if (p==0)
    {
        for (size_t n=N; n>0u; --n, ++X)
        {
            if (*X<0.0) { *X = 0.0; }
            else if (*X>1.0) { *X = 1.0; }
        }
    }
    else if (p==1)
    {
        for (size_t n=N; n>0u; --n, ++X)
        {
            if (*X<0.0) { *X = 0.0; }
            else if (*X>1.0) { *X = 1.0; }
            else { *X *= *X * (3.0-2.0**X); }
        }
    }
    else
    {
        fprintf(stderr,"error in smoothstep_inplace_d: n must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
