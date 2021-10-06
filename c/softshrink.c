//This gets the SoftShrink activation function for each element of X.
//This is similar to the dead-zone, but different outside of the dead-zone.

#include <stdio.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int softshrink_s (float *Y, const float *X, const size_t N, const float lambda);
int softshrink_d (double *Y, const double *X, const size_t N, const double lambda);

int softshrink_inplace_s (float *X, const size_t N, const float lambda);
int softshrink_inplace_d (double *X, const size_t N, const double lambda);


int softshrink_s (float *Y, const float *X, const size_t N, const float lambda)
{
    if (lambda<0.0f) { fprintf(stderr,"error in softshrink_s: lambda must be non-negative\n"); return 1; }

    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        *Y = (*X>lambda) ? *X-lambda : (*X<-lambda) ? *X+lambda : 0.0f;
    }

    return 0;
}


int softshrink_d (double *Y, const double *X, const size_t N, const double lambda)
{
    if (lambda<0.0) { fprintf(stderr,"error in softshrink_d: lambda must be non-negative\n"); return 1; }

    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        *Y = (*X>lambda) ? *X-lambda : (*X<-lambda) ? *X+lambda : 0.0;
    }

    return 0;
}


int softshrink_inplace_s (float *X, const size_t N, const float lambda)
{
    if (lambda<0.0f) { fprintf(stderr,"error in softshrink_inplace_s: lambda must be non-negative\n"); return 1; }

    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X>lambda) { *X -= lambda; }
        else if (*X<-lambda) { *X += lambda; }
        else { *X = 0.0f; }
    }

    return 0;
}


int softshrink_inplace_d (double *X, const size_t N, const double lambda)
{
    if (lambda<0.0) { fprintf(stderr,"error in softshrink_inplace_d: lambda must be non-negative\n"); return 1; }

    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X>lambda) { *X -= lambda; }
        else if (*X<-lambda) { *X += lambda; }
        else { *X = 0.0; }
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
