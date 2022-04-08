//This gets the threshold function for each element of X.
//y = x,   if x>thresh
//y = val, otherwise

//For thresh=0.0 and val=0.0, this is the usual ReLU.

#include <stdio.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int threshold_s (float *Y, const float *X, const size_t N, const float thresh, const float val)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        *Y = (*X>thresh) ? *X : val;
    }

    return 0;
}


int threshold_d (double *Y, const double *X, const size_t N, const double thresh, const double val)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        *Y = (*X>thresh) ? *X : val;
    }

    return 0;
}


int threshold_inplace_s (float *X, const size_t N, const float thresh, const float val)
{
    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X<thresh) { *X = val; }
    }

    return 0;
}


int threshold_inplace_d (double *X, const size_t N, const double thresh, const double val)
{
    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X<thresh) { *X = val; }
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
