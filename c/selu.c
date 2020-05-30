//This gets scaled exponential lineary unit (SELU) [Klambauer et al. 2017] activation function for each element of X.

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int selu_s (float *Y, const float *X, const int N);
int selu_d (double *Y, const double *X, const int N);

int selu_inplace_s (float *X, const int N);
int selu_inplace_d (double *X, const int N);


int selu_s (float *Y, const float *X, const int N)
{
    int n;
    const float lam = 1.0507f, sc =  1.0507f*1.67326f;

    //Checks
    if (N<0) { fprintf(stderr,"error in selu_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = (X[n]<0.0f) ? sc*(expf(X[n])-1.0f) : lam*X[n]; }

    return 0;
}


int selu_d (double *Y, const double *X, const int N)
{
    int n;
    const double lam = 1.0507, sc =  1.0507*1.67326;

    //Checks
    if (N<0) { fprintf(stderr,"error in selu_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = (X[n]<0.0) ? sc*(exp(X[n])-1.0) : lam*X[n]; }
    
    return 0;
}


int selu_inplace_s (float *X, const int N)
{
    int n;
    const float lam = 1.0507f, sc =  1.0507f*1.67326f;

    //Checks
    if (N<0) { fprintf(stderr,"error in selu_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] = (X[n]<0.0f) ? sc*(expf(X[n])-1.0f) : lam*X[n]; }

    return 0;
}


int selu_inplace_d (double *X, const int N)
{
    int n;
    const double lam = 1.0507, sc =  1.0507*1.67326;

    //Checks
    if (N<0) { fprintf(stderr,"error in selu_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] = (X[n]<0.0) ? sc*(exp(X[n])-1.0) : lam*X[n]; }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
