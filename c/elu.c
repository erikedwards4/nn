//This gets exponential linear unit (ELU) activation function [Clevert et al. 2015] for each element of X.

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int elu_s (float *Y, const float *X, const size_t N, const float alpha);
int elu_d (double *Y, const double *X, const size_t N, const double alpha);

int elu_inplace_s (float *X, const size_t N, const float alpha);
int elu_inplace_d (double *X, const size_t N, const double alpha);


int elu_s (float *Y, const float *X, const size_t N, const float alpha)
{
    if (alpha<0.0f) { fprintf(stderr,"error in elu_s: alpha must be nonnegative\n"); return 1; }

    if (alpha==0.0f)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0f) ? 0.0f : *X; }
    }
    else if (alpha==1.0f)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0f) ? expf(*X)-1.0f : *X; }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0f) ? alpha*(expf(*X)-1.0f) : *X; }
    }
    

    return 0;
}


int elu_d (double *Y, const double *X, const size_t N, const double alpha)
{
    if (alpha<0.0) { fprintf(stderr,"error in elu_d: alpha must be nonnegative\n"); return 1; }

    if (alpha==0.0)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0) ? 0.0 : *X; }
    }
    else if (alpha==1.0)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0) ? exp(*X)-1.0 : *X; }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0) ? alpha*(exp(*X)-1.0) : *X; }
    }
    
    return 0;
}


int elu_inplace_s (float *X, const size_t N, const float alpha)
{
    if (alpha<0.0f) { fprintf(stderr,"error in elu_inplace_s: alpha must be nonnegative\n"); return 1; }

    if (alpha==0.0f)
    {
        for (size_t n=N; n>0u; --n, ++X) { if (*X<0.0f) { *X = 0.0f; } }
    }
    else if (alpha==1.0f)
    {
        for (size_t n=N; n>0u; --n, ++X) { if (*X<0.0f) { *X = expf(*X)-1.0f; } }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X) { if (*X<0.0f) { *X = alpha*(expf(*X)-1.0f); } }
    }

    return 0;
}


int elu_inplace_d (double *X, const size_t N, const double alpha)
{
    if (alpha<0.0) { fprintf(stderr,"error in elu_inplace_d: alpha must be nonnegative\n"); return 1; }

    if (alpha==0.0)
    {
        for (size_t n=N; n>0u; --n, ++X) { if (*X<0.0) { *X = 0.0; } }
    }
    else if (alpha==1.0)
    {
        for (size_t n=N; n>0u; --n, ++X) { if (*X<0.0) { *X = exp(*X)-1.0; } }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X) { if (*X<0.0) { *X = alpha*(exp(*X)-1.0); } }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
