//Gets cube-root of input X element-wise.
//This has in-place and not-in-place versions.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int cbrt_s (float *Y, const float *X, const int N);
int cbrt_d (double *Y, const double *X, const int N);
int cbrt_c (float *Y, const float *X, const int N);
int cbrt_z (double *Y, const double *X, const int N);

int cbrt_inplace_s (float *X, const int N);
int cbrt_inplace_d (double *X, const int N);
int cbrt_inplace_c (float *X, const int N);
int cbrt_inplace_z (double *X, const int N);


int cbrt_s (float *Y, const float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in cbrt_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = cbrtf(X[n]); }

    return 0;
}


int cbrt_d (double *Y, const double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in cbrt_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = cbrt(X[n]); }
    
    return 0;
}


int cbrt_c (float *Y, const float *X, const int N)
{
    int n2;
    _Complex float y;
    const float p = 1.0f/3.0f;

    //Checks
    if (N<0) { fprintf(stderr,"error in cbrt_c: N (num elements X) must be nonnegative\n"); return 1; }

    for (n2=0; n2<2*N; n2+=2)
    {
        y = cpowf(X[n2]+1.0if*X[n2+1],p);
        memcpy(&Y[n2],(float *)&y,2*sizeof(float));
    }
    
    return 0;
}


int cbrt_z (double *Y, const double *X, const int N)
{
    int n2;
    _Complex double y;
    const double p = 1.0/3.0;

    //Checks
    if (N<0) { fprintf(stderr,"error in cbrt_z: N (num elements X) must be nonnegative\n"); return 1; }

    for (n2=0; n2<2*N; n2+=2)
    {
        y = cpow(X[n2]+1.0i*X[n2+1],p);
        memcpy(&Y[n2],(double *)&y,2*sizeof(double));
    }
    
    return 0;
}


int cbrt_inplace_s (float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in cbrt_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] = cbrtf(X[n]); }

    return 0;
}


int cbrt_inplace_d (double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in cbrt_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] = cbrt(X[n]); }
    
    return 0;
}


int cbrt_inplace_c (float *X, const int N)
{
    int n2;
    _Complex float x;
    const float p = 1.0f/3.0f;

    //Checks
    if (N<0) { fprintf(stderr,"error in cbrt_inplace_c: N (num elements X) must be nonnegative\n"); return 1; }

    for (n2=0; n2<2*N; n2+=2)
    {
        x = cpowf(X[n2]+1.0if*X[n2+1],p);
        memcpy(&X[n2],(float *)&x,2*sizeof(float));
    }
    
    return 0;
}


int cbrt_inplace_z (double *X, const int N)
{
    int n2;
    _Complex double x;
    const double p = 1.0/3.0;

    //Checks
    if (N<0) { fprintf(stderr,"error in cbrt_inplace_z: N (num elements X) must be nonnegative\n"); return 1; }

    for (n2=0; n2<2*N; n2+=2)
    {
        x = cpow(X[n2]+1.0i*X[n2+1],p);
        memcpy(&X[n2],(double *)&x,2*sizeof(double));
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
