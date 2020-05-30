//This gets Swish activation function [Ramachandran et al. 2017] for each element of X.

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int swish_s (float *Y, const float *X, const int N, const float beta);
int swish_d (double *Y, const double *X, const int N, const double beta);

int swish_inplace_s (float *X, const int N, const float beta);
int swish_inplace_d (double *X, const int N, const double beta);


int swish_s (float *Y, const float *X, const int N, const float beta)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in swish_s: N (num elements X) must be nonnegative\n"); return 1; }
    if (beta<0.0f) { fprintf(stderr,"error in swish_s: beta must be nonnegative\n"); return 1; }

    if (beta==0.0f)
    {
        for (n=0; n<N; n++) { Y[n] = 0.5f*X[n]; }
    }
    else if (beta==1.0f)
    {
        for (n=0; n<N; n++) { Y[n] = X[n]/(1.0f+expf(-X[n])); }
    }
    else
    {
        for (n=0; n<N; n++) { Y[n] = X[n]/(1.0f+expf(-beta*X[n])); }
    }

    return 0;
}


int swish_d (double *Y, const double *X, const int N, const double beta)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in swish_d: N (num elements X) must be nonnegative\n"); return 1; }
    if (beta<0.0) { fprintf(stderr,"error in swish_d: beta must be nonnegative\n"); return 1; }

    if (beta==0.0)
    {
        for (n=0; n<N; n++) { Y[n] = 0.5*X[n]; }
    }
    else if (beta==1.0)
    {
        for (n=0; n<N; n++) { Y[n] = X[n]/(1.0+exp(-X[n])); }
    }
    else
    {
        for (n=0; n<N; n++) { Y[n] = X[n]/(1.0+exp(-beta*X[n])); }
    }
    
    return 0;
}


int swish_inplace_s (float *X, const int N, const float beta)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in swish_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }
    if (beta<0.0f) { fprintf(stderr,"error in swish_inplace_s: beta must be nonnegative\n"); return 1; }

    if (beta==0.0f)
    {
        for (n=0; n<N; n++) { X[n] *= 0.5f; }
    }
    else if (beta==1.0f)
    {
        for (n=0; n<N; n++) { X[n] /= 1.0f + expf(-X[n]); }
    }
    else
    {
        for (n=0; n<N; n++) { X[n] /= 1.0f + expf(-beta*X[n]); }
    }

    return 0;
}


int swish_inplace_d (double *X, const int N, const double beta)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in swish_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }
    if (beta<0.0) { fprintf(stderr,"error in swish_inplace_d: beta must be nonnegative\n"); return 1; }

    if (beta==0.0)
    {
        for (n=0; n<N; n++) { X[n] *= 0.5; }
    }
    else if (beta==1.0)
    {
        for (n=0; n<N; n++) { X[n] /= 1.0 + exp(-X[n]); }
    }
    else
    {
        for (n=0; n<N; n++) { X[n] /= 1.0 + exp(-beta*X[n]); }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
