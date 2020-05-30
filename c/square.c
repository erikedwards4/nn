//This just squares input X element-wise.
//For complex data, this is |X|.^2, i.e. Xr*Xr + Xi*Xi.

//This has in-place and not-in-place versions.

#include <stdio.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int square_s (float *Y, const float *X, const int N);
int square_d (double *Y, const double *X, const int N);
int square_c (float *Y, const float *X, const int N);
int square_z (double *Y, const double *X, const int N);

int square_inplace_s (float *X, const int N);
int square_inplace_d (double *X, const int N);
int square_inplace_c (float *X, const int N);
int square_inplace_z (double *X, const int N);


int square_s (float *Y, const float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in square_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = X[n]*X[n]; }

    return 0;
}


int square_d (double *Y, const double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in square_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = X[n]*X[n]; }
    
    return 0;
}


int square_c (float *Y, const float *X, const int N)
{
    int n, n2;

    //Checks
    if (N<0) { fprintf(stderr,"error in square_c: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0, n2=0; n<N; n++, n2+=2) { Y[n] = X[n2]*X[n2] + X[n2+1]*X[n2+1]; }
    
    return 0;
}


int square_z (double *Y, const double *X, const int N)
{
    int n, n2;

    //Checks
    if (N<0) { fprintf(stderr,"error in square_z: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0, n2=0; n<N; n++, n2+=2) { Y[n] = X[n2]*X[n2] + X[n2+1]*X[n2+1]; }
    
    return 0;
}


int square_inplace_s (float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in square_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] *= X[n]; }

    return 0;
}


int square_inplace_d (double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in square_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] *= X[n]; }
    
    return 0;
}


int square_inplace_c (float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in square_inplace_c: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<2*N; n++) { X[n] *= X[n]; }
    for (n=0; n<2*N; n+=2) { X[n] += X[n+1]; X[n+1] = 0.0f; }
    
    return 0;
}


int square_inplace_z (double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in square_inplace_z: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<2*N; n++) { X[n] *= X[n]; }
    for (n=0; n<2*N; n+=2) { X[n] += X[n+1]; X[n+1] = 0.0; }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
