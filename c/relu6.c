//This gets ReLU6 activation function for each element of X.
//Same as ReLU, except saturates at 6.

#include <stdio.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int relu6_s (float *Y, const float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        *Y = (*X<0.0f) ? 0.0f : (*X>6.0f) ? 6.0f : *X;
    }

    return 0;
}


int relu6_d (double *Y, const double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        *Y = (*X<0.0) ? 0.0 : (*X>6.0) ? 6.0 : *X;
    }
    
    return 0;
}


int relu6_inplace_s (float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X<0.0f) { *X = 0.0f; }
        else if (*X>6.0f) { *X = 6.0f; }
    }
    
    return 0;
}


int relu6_inplace_d (double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X<0.0) { *X = 0.0; }
        else if (*X>6.0) { *X = 6.0; }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
