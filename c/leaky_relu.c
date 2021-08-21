//This gets the leaky ReLU activation function for each element of X.
//This is also called the parametric ReLU with alpha=0.01.

#include <stdio.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int leaky_relu_s (float *Y, const float *X, const size_t N);
int leaky_relu_d (double *Y, const double *X, const size_t N);

int leaky_relu_inplace_s (float *X, const size_t N);
int leaky_relu_inplace_d (double *X, const size_t N);


int leaky_relu_s (float *Y, const float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0f) ? 0.01f**X : *X; }

    return 0;
}


int leaky_relu_d (double *Y, const double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0) ? 0.01**X : *X; }
    
    return 0;
}


int leaky_relu_inplace_s (float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X) { if(*X<0.0f) { *X *= 0.01f; } }

    return 0;
}


int leaky_relu_inplace_d (double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X) { if(*X<0.0) { *X *= 0.01; } }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
