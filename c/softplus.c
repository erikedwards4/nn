//Gets the softplus function (derivative of logistic) of input X element-wise.
//This has in-place and not-in-place versions.

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int softplus_s (float *Y, const float *X, const int N);
int softplus_d (double *Y, const double *X, const int N);

int softplus_inplace_s (float *X, const int N);
int softplus_inplace_d (double *X, const int N);


int softplus_s (float *Y, const float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in softplus_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = logf(1.0f+expf(X[n])); }

    return 0;
}


int softplus_d (double *Y, const double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in softplus_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = log(1.0+exp(X[n])); }
    
    return 0;
}


int softplus_inplace_s (float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in softplus_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] = logf(1.0f+expf(X[n])); }

    return 0;
}


int softplus_inplace_d (double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in softplus_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] = log(1.0+exp(X[n])); }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
