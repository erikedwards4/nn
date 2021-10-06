//Gets the Gaussian Error Linear Unit (GELU) [Hendrycks & Gimpel 2018] of input X element-wise.
//This has in-place and not-in-place versions.

#include <stdio.h>
#include <math.h>

#ifndef M_SQRT1_2
    #define M_SQRT1_2 0.707106781186547524401
#endif

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int gelu_s (float *Y, const float *X, const size_t N);
int gelu_d (double *Y, const double *X, const size_t N);

int gelu_inplace_s (float *X, const size_t N);
int gelu_inplace_d (double *X, const size_t N);


int gelu_s (float *Y, const float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = 0.5f * *X * (1.0f + erff(*X*(float)M_SQRT1_2)); }

    return 0;
}


int gelu_d (double *Y, const double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = 0.5 * *X * (1.0 + erf(*X*M_SQRT1_2)); }
    
    return 0;
}


int gelu_inplace_s (float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X) { *X *= 0.5f * (1.0f + erff(*X*(float)M_SQRT1_2)); }

    return 0;
}


int gelu_inplace_d (double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X) { *X *= 0.5 * (1.0 + erf(*X*M_SQRT1_2)); }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
