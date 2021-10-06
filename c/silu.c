//This gets sigmoid-weighted linear unit (SiLU) [Elfwing et al. 2017] for each element of X.
//For each element: y = x * logistic_sigmoid(x);

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int silu_s (float *Y, const float *X, const size_t N);
int silu_d (double *Y, const double *X, const size_t N);

int silu_inplace_s (float *X, const size_t N);
int silu_inplace_d (double *X, const size_t N);


int silu_s (float *Y, const float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X / (1.0f+expf(-*X)); }

    return 0;
}


int silu_d (double *Y, const double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X / (1.0+exp(-*X)); }
    
    return 0;
}


int silu_inplace_s (float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X) { *X /= 1.0f + expf(-*X); }

    return 0;
}


int silu_inplace_d (double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X) { *X /= 1.0 + exp(-*X); }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
