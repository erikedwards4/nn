//This gets CELU activation function for each element of X.
//This is from Continuously Differentiable Exponential Linear Units [Barron 2017].
//Barron JT. 2017. Continuously differentiable exponential linear units. arXiv. 1704.07483[v.1]: 1-2.

//For each element: y = max(0,x) + min(0,alpha*(exp(x/alpha)−1))

#include <stdio.h>
#include <math.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int celu_s (float *Y, const float *X, const size_t N, const float alpha)
{
    float tmp;

    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        tmp = alpha * (expf(*X/alpha)-1.0f);

        if (*X>0.0f)
        {
            if (tmp<0.0f) { *Y = *X + tmp; }
            else { *Y = *X; }
        }
        else
        {
            if (tmp<0.0f) { *Y = tmp; }
            else { *Y = 0.0f; }
        }
    }

    return 0;
}


int celu_d (double *Y, const double *X, const size_t N, const double alpha)
{
    double tmp;

    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        tmp = alpha * (exp(*X/alpha)-1.0);

        if (*X>0.0)
        {
            if (tmp<0.0) { *Y = *X + tmp; }
            else { *Y = *X; }
        }
        else
        {
            if (tmp<0.0) { *Y = tmp; }
            else { *Y = 0.0; }
        }
    }

    return 0;
}


int celu_inplace_s (float *X, const size_t N, const float alpha)
{
    float tmp;

    for (size_t n=N; n>0u; --n, ++X)
    {
        tmp = alpha * (expf(*X/alpha)-1.0f);

        if (*X>0.0f)
        {
            if (tmp<0.0f) { *X += tmp; }
        }
        else
        {
            if (tmp<0.0f) { *X = tmp; }
            else { *X = 0.0f; }
        }
    }

    return 0;
}


int celu_inplace_d (double *X, const size_t N, const double alpha)
{
    double tmp;

    for (size_t n=N; n>0u; --n, ++X)
    {
        tmp = alpha * (exp(*X/alpha)-1.0);

        if (*X>0.0)
        {
            if (tmp<0.0) { *X += tmp; }
        }
        else
        {
            if (tmp<0.0) { *X = tmp; }
            else { *X = 0.0; }
        }
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
