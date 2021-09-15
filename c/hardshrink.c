//This gets the HardShrink activation function for each element of X.
//This is similar to the dead-zone, but different outside of the dead-zone.

#include <stdio.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int hardshrink_s (float *Y, const float *X, const size_t N, const float lambda);
int hardshrink_d (double *Y, const double *X, const size_t N, const double lambda);

int hardshrink_inplace_s (float *X, const size_t N, const float lambda);
int hardshrink_inplace_d (double *X, const size_t N, const double lambda);


int hardshrink_s (float *Y, const float *X, const size_t N, const float lambda)
{
    if (lambda<0.0f) { fprintf(stderr,"error in hardshrink_s: lambda must be non-negative\n"); return 1; }

    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        *Y = (*X<-lambda || *X>lambda) ? *X : 0.0f;
    }

    return 0;
}


int hardshrink_d (double *Y, const double *X, const size_t N, const double lambda)
{
    if (lambda<0.0) { fprintf(stderr,"error in hardshrink_d: lambda must be non-negative\n"); return 1; }

    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        *Y = (*X<-lambda || *X>lambda) ? *X : 0.0;
    }

    return 0;
}


int hardshrink_inplace_s (float *X, const size_t N, const float lambda)
{
    if (lambda<0.0f) { fprintf(stderr,"error in hardshrink_inplace_s: lambda must be non-negative\n"); return 1; }

    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X>=-lambda && *X<=lambda) { *X = 0.0f; }
    }

    return 0;
}


int hardshrink_inplace_d (double *X, const size_t N, const double lambda)
{
    if (lambda<0.0) { fprintf(stderr,"error in hardshrink_inplace_d: lambda must be non-negative\n"); return 1; }

    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X>=-lambda && *X<=lambda) { *X = 0.0; }
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
