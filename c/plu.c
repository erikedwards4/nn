//Gets the Piecewise Linear Unit (PLU) function [Nicolae 2018] of input X element-wise.
//This has in-place and not-in-place versions.

#include <stdio.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int plu_s (float *Y, const float *X, const size_t N, float a, float c)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        if (*X>c) { *Y = a*(*X-c) + c; }
        else if (*X<-c) { *Y = a*(*X+c) - c; }
        else { *Y = *X; }
    }

    return 0;
}


int plu_d (double *Y, const double *X, const size_t N, double a, double c)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        if (*X>c) { *Y = a*(*X-c) + c; }
        else if (*X<-c) { *Y = a*(*X+c) - c; }
        else { *Y = *X; }
    }
    
    return 0;
}


int plu_inplace_s (float *X, const size_t N, float a, float c)
{
    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X>c) { *X = a*(*X-c) + c; }
        else if (*X<-c) { *X = a*(*X+c) - c; }
    }

    return 0;
}


int plu_inplace_d (double *X, const size_t N, double a, double c)
{
    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X>c) { *X = a*(*X-c) + c; }
        else if (*X<-c) { *X = a*(*X+c) - c; }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
