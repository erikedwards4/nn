//This outputs X unchanged.
//Could be useful for debugging or for place-holder in a chain.

#include <stdio.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int identity_s (float *Y, const float *X, const int N);
int identity_d (double *Y, const double *X, const int N);
int identity_c (float *Y, const float *X, const int N);
int identity_z (double *Y, const double *X, const int N);

int identity_inplace_s (float *X, const int N);
int identity_inplace_d (double *X, const int N);
int identity_inplace_c (float *X, const int N);
int identity_inplace_z (double *X, const int N);


int identity_s (float *Y, const float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in identity_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = X[n]; }

    return 0;
}


int identity_d (double *Y, const double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in identity_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = X[n]; }
    
    return 0;
}


int identity_c (float *Y, const float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in identity_c: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<2*N; n++) { Y[n] = X[n]; }

    return 0;
}


int identity_z (double *Y, const double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in identity_z: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<2*N; n++) { Y[n] = X[n]; }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
