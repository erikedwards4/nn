//This gets softclip activation function [Klimek & Perelstein 2020] for each element of X.

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int softclip_s (float *Y, const float *X, const size_t N, const float p);
int softclip_d (double *Y, const double *X, const size_t N, const double p);

int softclip_inplace_s (float *X, const size_t N, const float p);
int softclip_inplace_d (double *X, const size_t N, const double p);


int softclip_s (float *Y, const float *X, const size_t N, const float p)
{
    if (p<=0.0f) { fprintf(stderr,"error in softclip_s: p must be positive\n"); return 1; }

    const float ip = 1.0f/p;
    float x;

    for (size_t n=0; n<N; ++n, ++X, ++Y)
    {
        x = p * *X;
        if (x>88.72f) { *Y = 1.0f; }
        else { *Y = ip * logf((1.0f+expf(x))/(1.0f+expf(x-1.0f))); }
    }

    return 0;
}


int softclip_d (double *Y, const double *X, const size_t N, const double p)
{
    if (p<=0.0) { fprintf(stderr,"error in softclip_d: p must be positive\n"); return 1; }

    const double ip = 1.0/p;
    double x;

    for (size_t n=0; n<N; ++n, ++X, ++Y)
    {
        x = p * *X;
        if (x>709.77) { *Y = 1.0; }
        else { *Y = ip * log((1.0+exp(x))/(1.0+exp(x-1.0))); }
    }
    
    return 0;
}


int softclip_inplace_s (float *X, const size_t N, const float p)
{
    if (p<=0.0f) { fprintf(stderr,"error in softclip_inplace_s: p must be positive\n"); return 1; }

    const float ip = 1.0f/p;
    float x;

    for (size_t n=0; n<N; ++n, ++X)
    {
        x = p * *X;
        if (x>88.72f) { *X = 1.0f; }
        else { *X = ip * logf((1.0f+expf(x))/(1.0f+expf(x-1.0f))); }
    }

    return 0;
}


int softclip_inplace_d (double *X, const size_t N, const double p)
{
    if (p<=0.0) { fprintf(stderr,"error in softclip_inplace_d: p must be positive\n"); return 1; }

    const double ip = 1.0/p;
    double x;

    for (size_t n=0; n<N; ++n, ++X)
    {
        x = p * *X;
        if (x>709.77) { *X = 1.0; }
        else { *X = ip * log((1.0+exp(x))/(1.0+exp(x-1.0))); }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
