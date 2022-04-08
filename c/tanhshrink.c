//Gets Tanhshrink of input X element-wise.
// y = x - tanh(x) = x - (exp(x)-exp(-x))/(exp(x)+exp(-x))
//This has in-place and not-in-place versions.

#include <stdio.h>
#include <math.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int tanhshrink_s (float *Y, const float *X, const size_t N)
{
    float xp, xm;

    //for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X - tanhf(*X); }
    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        xp = expf(*X); xm = expf(-*X);
        *Y = *X - (xp-xm)/(xp+xm);
    }

    return 0;
}


int tanhshrink_d (double *Y, const double *X, const size_t N)
{
    double xp, xm;
    
    //for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X - tanh(*X); }
    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        xp = exp(*X); xm = exp(-*X);
        *Y = *X - (xp-xm)/(xp+xm);
    }
    
    return 0;
}


int tanhshrink_inplace_s (float *X, const size_t N)
{
    float xp, xm;

    for (size_t n=N; n>0u; --n, ++X)
    {
        xp = expf(*X); xm = expf(-*X);
        *X -= (xp-xm)/(xp+xm);
    }

    return 0;
}


int tanhshrink_inplace_d (double *X, const size_t N)
{
    double xp, xm;

    for (size_t n=N; n>0u; --n, ++X)
    {
        xp = exp(*X); xm = exp(-*X);
        *X -= (xp-xm)/(xp+xm);
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
