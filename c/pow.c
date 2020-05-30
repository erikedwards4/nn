//Raises each element of X to power p.
//This has in-place and not-in-place versions.

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int pow_s (float *Y, const float *X, const int N, const float p);
int pow_d (double *Y, const double *X, const int N, const double p);
int pow_c (float *Y, const float *X, const int N, const float p);
int pow_z (double *Y, const double *X, const int N, const double p);

int pow_inplace_s (float *X, const int N, const float p);
int pow_inplace_d (double *X, const int N, const double p);
int pow_inplace_c (float *X, const int N, const float p);
int pow_inplace_z (double *X, const int N, const double p);


int pow_s (float *Y, const float *X, const int N, const float p)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in pow_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = powf(X[n],p); }

    return 0;
}


int pow_d (double *Y, const double *X, const int N, const double p)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in pow_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = pow(X[n],p); }
    
    return 0;
}


int pow_c (float *Y, const float *X, const int N, const float p)
{
    int n2;
    _Complex float y;

    //Checks
    if (N<0) { fprintf(stderr,"error in pow_c: N (num elements X) must be nonnegative\n"); return 1; }

    for (n2=0; n2<2*N; n2+=2)
    {
        y = cpowf(X[n2]+1.0if*X[n2+1],(_Complex float)p);
        memcpy(&Y[n2],(float *)&y,2*sizeof(float));
    }
    
    return 0;
}


int pow_z (double *Y, const double *X, const int N, const double p)
{
    int n2;
    _Complex double y;

    //Checks
    if (N<0) { fprintf(stderr,"error in pow_z: N (num elements X) must be nonnegative\n"); return 1; }

    for (n2=0; n2<2*N; n2+=2)
    {
        y = cpow(X[n2]+1.0i*X[n2+1],(_Complex double)p);
        memcpy(&Y[n2],(double *)&y,2*sizeof(double));
    }
    
    return 0;
}


int pow_inplace_s (float *X, const int N, const float p)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in pow_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] = powf(X[n],p); }

    return 0;
}


int pow_inplace_d (double *X, const int N, const double p)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in pow_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] = pow(X[n],p); }
    
    return 0;
}


int pow_inplace_c (float *X, const int N, const float p)
{
    int n2;
    _Complex float x;

    //Checks
    if (N<0) { fprintf(stderr,"error in pow_inplace_c: N (num elements X) must be nonnegative\n"); return 1; }

    for (n2=0; n2<2*N; n2+=2)
    {
        x = cpowf(X[n2]+1.0if*X[n2+1],(_Complex float)p);
        memcpy(&X[n2],(float *)&x,2*sizeof(float));
    }
    
    return 0;
}


int pow_inplace_z (double *X, const int N, const double p)
{
    int n2;
    _Complex double x;

    //Checks
    if (N<0) { fprintf(stderr,"error in pow_inplace_z: N (num elements X) must be nonnegative\n"); return 1; }

    for (n2=0; n2<2*N; n2+=2)
    {
        x = cpow(X[n2]+1.0i*X[n2+1],(_Complex double)p);
        memcpy(&X[n2],(double *)&x,2*sizeof(double));
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif

