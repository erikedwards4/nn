//Gets the SQNL function [Wuraola & Patel 2018] of input X element-wise.
//This has in-place and not-in-place versions.

#include <stdio.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int sqnl_s (float *Y, const float *X, const int N);
int sqnl_d (double *Y, const double *X, const int N);

int sqnl_inplace_s (float *X, const int N);
int sqnl_inplace_d (double *X, const int N);


int sqnl_s (float *Y, const float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in sqnl_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++)
    {
        if (X[n]>0.0f)
        {
            if (X[n]<2.0f) { Y[n] = X[n] - 0.25f*X[n]*X[n]; }
            else { Y[n] = 1.0f; }
        }
        else
        {
            if (X[n]>-2.0f) { Y[n] = X[n] + 0.25f*X[n]*X[n]; }
            else { Y[n] = -1.0f; }
        }
    }

    return 0;
}


int sqnl_d (double *Y, const double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in sqnl_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++)
    {
        if (X[n]>0.0)
        {
            if (X[n]<2.0) { Y[n] = X[n] - 0.25*X[n]*X[n]; }
            else { Y[n] = 1.0; }
        }
        else
        {
            if (X[n]>-2.0) { Y[n] = X[n] + 0.25*X[n]*X[n]; }
            else { Y[n] = -1.0; }
        }
    }
    
    return 0;
}


int sqnl_inplace_s (float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in sqnl_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++)
    {
        if (X[n]>0.0f)
        {
            if (X[n]<2.0f) { X[n] -= 0.25f*X[n]*X[n]; }
            else { X[n] = 1.0f; }
        }
        else
        {
            if (X[n]>-2.0f) { X[n] += 0.25f*X[n]*X[n]; }
            else { X[n] = -1.0f; }
        }
    }

    return 0;
}


int sqnl_inplace_d (double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in sqnl_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++)
    {
        if (X[n]>0.0)
        {
            if (X[n]<2.0) { X[n] -= 0.25*X[n]*X[n]; }
            else { X[n] = 1.0; }
        }
        else
        {
            if (X[n]>-2.0) { X[n] += 0.25*X[n]*X[n]; }
            else { X[n] = -1.0; }
        }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
