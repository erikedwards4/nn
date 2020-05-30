//Gets the Gauss error function (erf) of input X element-wise.
//This has in-place and not-in-place versions.

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int erf_s (float *Y, const float *X, const int N);
int erf_d (double *Y, const double *X, const int N);

int erf_inplace_s (float *X, const int N);
int erf_inplace_d (double *X, const int N);


int erf_s (float *Y, const float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in erf_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = erff(X[n]); }

    return 0;
}


int erf_d (double *Y, const double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in erf_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = erf(X[n]); }
    
    return 0;
}


int erf_inplace_s (float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in erf_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] = erff(X[n]); }

    return 0;
}


int erf_inplace_d (double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in erf_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] = erf(X[n]); }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
