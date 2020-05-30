//This gets ReLU activation function for each element of X.

#include <stdio.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int relu_s (float *Y, const float *X, const int N);
int relu_d (double *Y, const double *X, const int N);

int relu_inplace_s (float *X, const int N);
int relu_inplace_d (double *X, const int N);


int relu_s (float *Y, const float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in relu_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = (X[n]<0.0f) ? 0.0f : X[n]; }

    return 0;
}


int relu_d (double *Y, const double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in relu_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = (X[n]<0.0) ? 0.0 : X[n]; }
    
    return 0;
}


int relu_inplace_s (float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in relu_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { if (X[n]<0.0f) { X[n] = 0.0f; } }

    return 0;
}


int relu_inplace_d (double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in relu_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { if (X[n]<0.0) { X[n] = 0.0; } }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
