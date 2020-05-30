//Gets the Piecewise Linear Unit (PLU) function [Nicolae 2018] of input X element-wise.
//This has in-place and not-in-place versions.

#include <stdio.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int plu_s (float *Y, const float *X, const int N, float a, float c);
int plu_d (double *Y, const double *X, const int N, double a, double c);

int plu_inplace_s (float *X, const int N, float a, float c);
int plu_inplace_d (double *X, const int N, double a, double c);


int plu_s (float *Y, const float *X, const int N, float a, float c)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in plu_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++)
    {
        if (X[n]>c) { Y[n] = a*(X[n]-c) + c; }
        else if (X[n]<-c) { Y[n] = a*(X[n]+c) - c; }
        else { Y[n] = X[n]; }
    }

    return 0;
}


int plu_d (double *Y, const double *X, const int N, double a, double c)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in plu_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++)
    {
        if (X[n]>c) { Y[n] = a*(X[n]-c) + c; }
        else if (X[n]<-c) { Y[n] = a*(X[n]+c) - c; }
        else { Y[n] = X[n]; }
    }
    
    return 0;
}


int plu_inplace_s (float *X, const int N, float a, float c)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in plu_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++)
    {
        if (X[n]>c) { X[n] = a*(X[n]-c) + c; }
        else if (X[n]<-c) { X[n] = a*(X[n]+c) - c; }
    }

    return 0;
}


int plu_inplace_d (double *X, const int N, double a, double c)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in plu_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++)
    {
        if (X[n]>c) { X[n] = a*(X[n]-c) + c; }
        else if (X[n]<-c) { X[n] = a*(X[n]+c) - c; }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
