//This gets sigmoid-weighted linear unit (SiLU) [Elfwing et al. 2017] for each element of X.

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int silu_s (float *Y, const float *X, const int N);
int silu_d (double *Y, const double *X, const int N);

int silu_inplace_s (float *X, const int N);
int silu_inplace_d (double *X, const int N);


int silu_s (float *Y, const float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in silu_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = X[n]/(1.0f+expf(-X[n])); }

    return 0;
}


int silu_d (double *Y, const double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in silu_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = X[n]/(1.0+exp(-X[n])); }
    
    return 0;
}


int silu_inplace_s (float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in silu_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] /= 1.0f + expf(-X[n]); }

    return 0;
}


int silu_inplace_d (double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in silu_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] /= 1.0 + exp(-X[n]); }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
