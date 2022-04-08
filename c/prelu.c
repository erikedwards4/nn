//This gets parametric ReLU activation function for each element of X.
//This is also called the leaky ReLU (where default alpha is 0.01).
//For compatibility to PyTorch, this function is identical to leaky ReLU.

#include <stdio.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int prelu_s (float *Y, const float *X, const size_t N, const float alpha)
{
    if (alpha==0.0f)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0f) ? 0.0f : *X; }
    }
    else if (alpha==1.0f)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X; }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0f) ? alpha**X : *X; }
    }

    return 0;
}


int prelu_d (double *Y, const double *X, const size_t N, const double alpha)
{
    if (alpha==0.0)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0) ? 0.0 : *X; }
    }
    else if (alpha==1.0)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X; }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0) ? alpha**X : *X; }
    }
    
    return 0;
}


int prelu_inplace_s (float *X, const size_t N, const float alpha)
{
    if (alpha==0.0f)
    {
        for (size_t n=N; n>0u; --n, ++X) { if (*X<0.0f) { *X = 0.0f; } }
    }
    else if (alpha==1.0f) {}
    else
    {
        for (size_t n=N; n>0u; --n, ++X) { if(*X<0.0f) { *X *= alpha; } }
    }

    return 0;
}


int prelu_inplace_d (double *X, const size_t N, const double alpha)
{
    if (alpha==0.0)
    {
        for (size_t n=N; n>0u; --n, ++X) { if (*X<0.0) { *X = 0.0; } }
    }
    else if (alpha==1.0) {}
    else
    {
        for (size_t n=N; n>0u; --n, ++X) { if(*X<0.0) { *X *= alpha; } }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
