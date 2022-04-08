//This gets scaled exponential lineary unit (SELU) [Klambauer et al. 2017] activation function for each element of X.

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int selu_s (float *Y, const float *X, const size_t N);
int selu_d (double *Y, const double *X, const size_t N);

int selu_inplace_s (float *X, const size_t N);
int selu_inplace_d (double *X, const size_t N);


int selu_s (float *Y, const float *X, const size_t N)
{
    const float lam = 1.0507f, sc =  1.0507f*1.67326f;


    for (size_t n=0; n<N; ++n, ++X, ++Y) { *Y = (*X<0.0f) ? sc*(expf(*X)-1.0f) : lam**X; }

    return 0;
}


int selu_d (double *Y, const double *X, const size_t N)
{
    const double lam = 1.0507, sc =  1.0507*1.67326;


    for (size_t n=0; n<N; ++n, ++X, ++Y) { *Y = (*X<0.0) ? sc*(exp(*X)-1.0) : lam**X; }
    
    return 0;
}


int selu_inplace_s (float *X, const size_t N)
{
    const float lam = 1.0507f, sc =  1.0507f*1.67326f;


    for (size_t n=0; n<N; ++n, ++X) { *X = (*X<0.0f) ? sc*(expf(*X)-1.0f) : lam**X; }

    return 0;
}


int selu_inplace_d (double *X, const size_t N)
{
    const double lam = 1.0507, sc =  1.0507*1.67326;


    for (size_t n=0; n<N; ++n, ++X) { *X = (*X<0.0) ? sc*(exp(*X)-1.0) : lam**X; }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
