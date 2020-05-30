//This gets exponential linear unit (ELU) activation function [Clevert et al. 2015] for each element of X.

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int elu_s (float *Y, const float *X, const int N, const float alpha);
int elu_d (double *Y, const double *X, const int N, const double alpha);

int elu_inplace_s (float *X, const int N, const float alpha);
int elu_inplace_d (double *X, const int N, const double alpha);


int elu_s (float *Y, const float *X, const int N, const float alpha)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in elu_s: N (num elements X) must be nonnegative\n"); return 1; }
    if (alpha<0.0f) { fprintf(stderr,"error in elu_s: alpha must be nonnegative\n"); return 1; }

    if (alpha==0.0f)
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<0.0f) ? 0.0f : X[n]; }
    }
    else if (alpha==1.0f)
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<0.0f) ? expf(X[n]-1.0f) : X[n]; }
    }
    else
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<0.0f) ? alpha*expf(X[n]-1.0f) : X[n]; }
    }
    

    return 0;
}


int elu_d (double *Y, const double *X, const int N, const double alpha)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in elu_d: N (num elements X) must be nonnegative\n"); return 1; }
    if (alpha<0.0) { fprintf(stderr,"error in elu_d: alpha must be nonnegative\n"); return 1; }

    if (alpha==0.0)
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<0.0) ? 0.0 : X[n]; }
    }
    else if (alpha==1.0)
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<0.0) ? exp(X[n]-1.0) : X[n]; }
    }
    else
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<0.0) ? alpha*exp(X[n]-1.0) : X[n]; }
    }
    
    return 0;
}


int elu_inplace_s (float *X, const int N, const float alpha)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in elu_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }
    if (alpha<0.0f) { fprintf(stderr,"error in elu_inplace_s: alpha must be nonnegative\n"); return 1; }

    if (alpha==0.0f)
    {
        for (n=0; n<N; n++) { if (X[n]<0.0f) { X[n] = 0.0f; } }
    }
    else if (alpha==1.0f)
    {
        for (n=0; n<N; n++) { if (X[n]<0.0f) { X[n] = expf(X[n]-1.0f); } }
    }
    else
    {
        for (n=0; n<N; n++) { if (X[n]<0.0f) { X[n] = alpha*expf(X[n]-1.0f); } }
    }

    return 0;
}


int elu_inplace_d (double *X, const int N, const double alpha)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in elu_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }
    if (alpha<0.0) { fprintf(stderr,"error in elu_inplace_d: alpha must be nonnegative\n"); return 1; }

    if (alpha==0.0)
    {
        for (n=0; n<N; n++) { if (X[n]<0.0) { X[n] = 0.0; } }
    }
    else if (alpha==1.0)
    {
        for (n=0; n<N; n++) { if (X[n]<0.0) { X[n] = exp(X[n]-1.0); } }
    }
    else
    {
        for (n=0; n<N; n++) { if (X[n]<0.0) { X[n] = alpha*exp(X[n]-1.0); } }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
