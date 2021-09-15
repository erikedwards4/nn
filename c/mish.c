//Gets the Mish function [Misra 2019] of input X element-wise.
//Misra D. Mish: a self regularized non-monotonic neural activation function. 2019. arXiv. 1908.08681(v.4): 1-13.

//For each element: y = x*tanh(softplus(x)) = x*tanh(ln(1+exp(x))).

//This has in-place and not-in-place versions.

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int mish_s (float *Y, const float *X, const size_t N);
int mish_d (double *Y, const double *X, const size_t N);

int mish_inplace_s (float *X, const size_t N);
int mish_inplace_d (double *X, const size_t N);


int mish_s (float *Y, const float *X, const size_t N)
{
    float xp, xm;

    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        xp = expf(*X);
        xm = expf(-logf(1.0f+xp));
        xp += 1.0f;
        *Y = *X * (xp-xm)/(xp+xm);
        //*Y = *X * tanhf(logf(1.0f+expf(*X)));
    }

    return 0;
}


int mish_d (double *Y, const double *X, const size_t N)
{
    double xp, xm;

    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        xp = exp(*X);
        xm = exp(-log(1.0+xp));
        xp += 1.0;
        *Y = *X * (xp-xm)/(xp+xm);
        //*Y = *X * tanh(log(1.0+exp(*X)));
    }

    return 0;
}


int mish_inplace_s (float *X, const size_t N)
{
    float xp, xm;

    for (size_t n=N; n>0u; --n, ++X)
    {
        xp = expf(*X);
        xm = expf(-logf(1.0f+xp));
        xp += 1.0f;
        *X *= (xp-xm)/(xp+xm);
        //*X *= tanhf(logf(1.0f+expf(*X)));
    }

    return 0;
}


int mish_inplace_d (double *X, const size_t N)
{
    double xp, xm;

    for (size_t n=N; n>0u; --n, ++X)
    {
        xp = exp(*X);
        xm = exp(-log(1.0+xp));
        xp += 1.0;
        *X *= (xp-xm)/(xp+xm);
        //*X *= tanh(log(1.0+exp(*X)));
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
