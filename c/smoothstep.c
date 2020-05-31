//This gets the smoothstep activation function for each element of X.
//Only p=0 (clamp) and p=1 (Hermite interpolation of clamp) supported here.

#include <stdio.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int smoothstep_s (float *Y, const float *X, const int N, const int p);
int smoothstep_d (double *Y, const double *X, const int N, const int p);

int smoothstep_inplace_s (float *X, const int N, const int p);
int smoothstep_inplace_d (double *X, const int N, const int p);


int smoothstep_s (float *Y, const float *X, const int N, const int p)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in smoothstep_s: N (num elements X) must be nonnegative\n"); return 1; }

    if (p==0)
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<0.0f) ? 0.0f : (X[n]>1.0f) ? 1.0f : X[n]; }
    }
    else if (p==1)
    {
        for (n=0; n<N; n++)
        {
            if (X[n]<0.0f) { Y[n] = 0.0f; }
            else if (X[n]>1.0f) { Y[n] = 1.0f; }
            else { Y[n] = X[n]*X[n]*(3.0f-2.0f*X[n]); }
        }
    }
    else
    {
        fprintf(stderr,"error in smoothstep_s: n must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int smoothstep_d (double *Y, const double *X, const int N, const int p)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in smoothstep_d: N (num elements X) must be nonnegative\n"); return 1; }

    if (p==0)
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<0.0) ? 0.0 : (X[n]>1.0) ? 1.0 : X[n]; }
    }
    else if (p==1)
    {
        for (n=0; n<N; n++)
        {
            if (X[n]<0.0) { Y[n] = 0.0; }
            else if (X[n]>1.0) { Y[n] = 1.0; }
            else { Y[n] = X[n]*X[n]*(3.0-2.0*X[n]); }
        }
    }
    else
    {
        fprintf(stderr,"error in smoothstep_d: n must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


int smoothstep_inplace_s (float *X, const int N, const int p)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in smoothstep_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }

    if (p==0)
    {
        for (n=0; n<N; n++)
        {
            if (X[n]<0.0f) { X[n] = 0.0f; }
            else if (X[n]>1.0f) { X[n] = 1.0f; }
        }
    }
    else if (p==1)
    {
        for (n=0; n<N; n++)
        {
            if (X[n]<0.0f) { X[n] = 0.0f; }
            else if (X[n]>1.0f) { X[n] = 1.0f; }
            else { X[n] = X[n]*X[n]*(3.0f-2.0f*X[n]); }
        }
    }
    else
    {
        fprintf(stderr,"error in smoothstep_inplace_s: n must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int smoothstep_inplace_d (double *X, const int N, const int p)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in smoothstep_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }

    if (p==0)
    {
        for (n=0; n<N; n++)
        {
            if (X[n]<0.0) { X[n] = 0.0; }
            else if (X[n]>1.0) { X[n] = 1.0; }
        }
    }
    else if (p==1)
    {
        for (n=0; n<N; n++)
        {
            if (X[n]<0.0) { X[n] = 0.0; }
            else if (X[n]>1.0) { X[n] = 1.0; }
            else { X[n] = X[n]*X[n]*(3.0-2.0*X[n]); }
        }
    }
    else
    {
        fprintf(stderr,"error in smoothstep_inplace_d: n must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
