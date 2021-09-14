//Gets the Gaussian Error Linear Unit (GELU) [Hendrycks & Gimpel 2018] of input X element-wise.
//This has in-place and not-in-place versions.

#include <stdio.h>
#include <math.h>

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
    const float sc = 1.0f/sqrtf(2.0f);

    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = 0.5f * *X * (1.0f+sc*erff(*X)); }

    return 0;
}


int gelu_d (double *Y, const double *X, const size_t N)
{
    const double sc = 1.0/sqrt(2.0);

    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = 0.5 * *X * (1.0+sc*erf(*X)); }
    
    return 0;
}


int gelu_inplace_s (float *X, const size_t N)
{
    const float sc = 0.5f/sqrtf(2.0f);

    for (size_t n=N; n>0u; --n, ++X) { *X *= 0.5f * (1.0f+sc*erff(*X)); }

    return 0;
}


int gelu_inplace_d (double *X, const size_t N)
{
    const double sc = 0.5/sqrt(2.0);

    for (size_t n=N; n>0u; --n, ++X) { *X *= 0.5 * (1.0+sc*erf(*X)); }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
