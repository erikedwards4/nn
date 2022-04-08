//This gets sigmoid activation function for each element of X.
//For each element: y = 1/(1+exp(-x)).

#include <stdio.h>
#include <math.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int sigmoid_s (float *Y, const float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        *Y = 1.0f / (1.0f+expf(-*X));
    }

    return 0;
}


int sigmoid_d (double *Y, const double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        *Y = 1.0 / (1.0+exp(-*X));
    }
    
    return 0;
}


int sigmoid_inplace_s (float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X)
    {
        *X = 1.0f / (1.0f+expf(-*X));
    }

    return 0;
}


int sigmoid_inplace_d (double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X)
    {
        *X = 1.0 / (1.0+exp(-*X));
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
