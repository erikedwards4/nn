//This gets softclip activation function [Klimek & Perelstein 2020] for each element of X.

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int softclip_s (float *Y, const float *X, const int N, const float p);
int softclip_d (double *Y, const double *X, const int N, const double p);

int softclip_inplace_s (float *X, const int N, const float p);
int softclip_inplace_d (double *X, const int N, const double p);


int softclip_s (float *Y, const float *X, const int N, const float p)
{
    int n;
    const float ip = 1.0f/p;
    float x;

    //Checks
    if (N<0) { fprintf(stderr,"error in softclip_s: N (num elements X) must be nonnegative\n"); return 1; }
    if (p<=0.0f) { fprintf(stderr,"error in softclip_s: p must be positive\n"); return 1; }

    for (n=0; n<N; n++)
    {
        x = p*X[n];
        if (x>88.72f) { Y[n] = 1.0f; }
        else { Y[n] = ip * logf((1.0f+expf(x))/(1.0f+expf(x-1.0f))); }
    }

    return 0;
}


int softclip_d (double *Y, const double *X, const int N, const double p)
{
    int n;
    const double ip = 1.0/p;
    double x;

    //Checks
    if (N<0) { fprintf(stderr,"error in softclip_d: N (num elements X) must be nonnegative\n"); return 1; }
    if (p<=0.0) { fprintf(stderr,"error in softclip_d: p must be positive\n"); return 1; }

    for (n=0; n<N; n++)
    {
        x = p*X[n];
        if (x>709.77) { Y[n] = 1.0; }
        else { Y[n] = ip * log((1.0+exp(x))/(1.0+exp(x-1.0))); }
    }
    
    return 0;
}


int softclip_inplace_s (float *X, const int N, const float p)
{
    int n;
    const float ip = 1.0f/p;
    float x;

    //Checks
    if (N<0) { fprintf(stderr,"error in softclip_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }
    if (p<=0.0f) { fprintf(stderr,"error in softclip_inplace_s: p must be positive\n"); return 1; }

    for (n=0; n<N; n++)
    {
        x = p*X[n];
        if (x>88.72f) { X[n] = 1.0f; }
        else { X[n] = ip * logf((1.0f+expf(x))/(1.0f+expf(x-1.0f))); }
    }

    return 0;
}


int softclip_inplace_d (double *X, const int N, const double p)
{
    int n;
    const double ip = 1.0/p;
    double x;

    //Checks
    if (N<0) { fprintf(stderr,"error in softclip_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }
    if (p<=0.0) { fprintf(stderr,"error in softclip_inplace_d: p must be positive\n"); return 1; }

    for (n=0; n<N; n++)
    {
        x = p*X[n];
        if (x>709.77) { X[n] = 1.0; }
        else { X[n] = ip * log((1.0+exp(x))/(1.0+exp(x-1.0))); }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
