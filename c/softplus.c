//Gets the softplus function (derivative of logistic) of input X element-wise.
//This has in-place and not-in-place versions.

#include <stdio.h>
#include <math.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int softplus_s (float *Y, const float *X, const size_t N, const float beta, const float thresh)
{
    float bx;

    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        bx = *X * beta;
        if (bx>thresh) { *Y = *X; }
        else { *Y = logf(1.0f+expf(bx)) / beta; }
    }

    return 0;
}


int softplus_d (double *Y, const double *X, const size_t N, const double beta, const double thresh)
{
    double bx;

    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        bx = *X * beta;
        if (bx>thresh) { *Y = *X; }
        else { *Y = log(1.0+exp(bx)) / beta; }
    }

    return 0;
}


int softplus_inplace_s (float *X, const size_t N, const float beta, const float thresh)
{
    float bx;

    for (size_t n=N; n>0u; --n, ++X)
    {
        bx = *X * beta;
        if (bx<thresh) { *X = logf(1.0f+expf(bx)) / beta; }
    }

    return 0;
}


int softplus_inplace_d (double *X, const size_t N, const double beta, const double thresh)
{
    double bx;

    for (size_t n=N; n>0u; --n, ++X)
    {
        bx = *X * beta;
        if (bx<thresh) { *X = log(1.0+exp(bx)) / beta; }
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
