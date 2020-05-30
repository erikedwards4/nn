//Gets the Gaussian Error Linear Unit (GELU) [Hendrycks & Gimpel 2018] of input X element-wise.
//This has in-place and not-in-place versions.

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int gelu_s (float *Y, const float *X, const int N);
int gelu_d (double *Y, const double *X, const int N);

int gelu_inplace_s (float *X, const int N);
int gelu_inplace_d (double *X, const int N);


int gelu_s (float *Y, const float *X, const int N)
{
    int n;
    const float sc = 1.0f/sqrtf(2.0f);

    //Checks
    if (N<0) { fprintf(stderr,"error in gelu_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = 0.5f*X[n]*(1.0f+sc*erff(X[n])); }

    return 0;
}


int gelu_d (double *Y, const double *X, const int N)
{
    int n;
    const double sc = 1.0/sqrt(2.0);

    //Checks
    if (N<0) { fprintf(stderr,"error in gelu_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = 0.5*X[n]*(1.0+sc*erf(X[n])); }
    
    return 0;
}


int gelu_inplace_s (float *X, const int N)
{
    int n;
    const float sc = 0.5f/sqrtf(2.0f);

    //Checks
    if (N<0) { fprintf(stderr,"error in gelu_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] *= 0.5f + sc*erff(X[n]); }

    return 0;
}


int gelu_inplace_d (double *X, const int N)
{
    int n;
    const double sc = 0.5/sqrt(2.0);

    //Checks
    if (N<0) { fprintf(stderr,"error in gelu_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] = 0.5*X[n]*(1.0+sc*erf(X[n])); }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
