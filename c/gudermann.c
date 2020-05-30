//Gets Gudermannian function of input X element-wise.
//This has in-place and not-in-place versions.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int gudermann_s (float *Y, const float *X, const int N);
int gudermann_d (double *Y, const double *X, const int N);
int gudermann_c (float *Y, const float *X, const int N);
int gudermann_z (double *Y, const double *X, const int N);

int gudermann_inplace_s (float *X, const int N);
int gudermann_inplace_d (double *X, const int N);
int gudermann_inplace_c (float *X, const int N);
int gudermann_inplace_z (double *X, const int N);


int gudermann_s (float *Y, const float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in gudermann_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = 2.0f*atanf(tanhf(0.5f*X[n])); }

    return 0;
}


int gudermann_d (double *Y, const double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in gudermann_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = 2.0*atan(tanh(0.5*X[n])); }
    
    return 0;
}


int gudermann_c (float *Y, const float *X, const int N)
{
    int n2;
    _Complex float y;

    //Checks
    if (N<0) { fprintf(stderr,"error in gudermann_c: N (num elements X) must be nonnegative\n"); return 1; }

    for (n2=0; n2<2*N; n2+=2)
    {
        y = 2.0f*catanf(ctanhf(0.5f*X[n2]+1.0if*0.5f*X[n2+1]));
        memcpy(&Y[n2],(float *)&y,2*sizeof(float));
    }
    
    return 0;
}


int gudermann_z (double *Y, const double *X, const int N)
{
    int n2;
    _Complex double y;

    //Checks
    if (N<0) { fprintf(stderr,"error in gudermann_z: N (num elements X) must be nonnegative\n"); return 1; }

    for (n2=0; n2<2*N; n2+=2)
    {
        y = 2.0*catan(ctanh(0.5*X[n2]+1.0i*0.5*X[n2+1]));
        memcpy(&Y[n2],(double *)&y,2*sizeof(double));
    }
    
    return 0;
}


int gudermann_inplace_s (float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in gudermann_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] = 2.0f*atanf(tanhf(0.5f*X[n])); }

    return 0;
}


int gudermann_inplace_d (double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in gudermann_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] = 2.0*atan(tanh(0.5*X[n])); }
    
    return 0;
}


int gudermann_inplace_c (float *X, const int N)
{
    int n2;
    _Complex float x;

    //Checks
    if (N<0) { fprintf(stderr,"error in gudermann_inplace_c: N (num elements X) must be nonnegative\n"); return 1; }

    for (n2=0; n2<2*N; n2+=2)
    {
        x = 2.0f*catanf(ctanhf(0.5f*X[n2]+1.0if*0.5f*X[n2+1]));
        memcpy(&X[n2],(float *)&x,2*sizeof(float));
    }
    
    return 0;
}


int gudermann_inplace_z (double *X, const int N)
{
    int n2;
    _Complex double x;

    //Checks
    if (N<0) { fprintf(stderr,"error in gudermann_inplace_z: N (num elements X) must be nonnegative\n"); return 1; }

    for (n2=0; n2<2*N; n2+=2)
    {
        x = 2.0*catan(ctanh(0.5*X[n2]+1.0i*0.5*X[n2+1]));
        memcpy(&X[n2],(double *)&x,2*sizeof(double));
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
