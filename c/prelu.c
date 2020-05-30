//This gets parametric ReLU activation function for each element of X.
//For alpha=0.01, this is also called the leaky ReLU.

#include <stdio.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int prelu_s (float *Y, const float *X, const int N, const float alpha);
int prelu_d (double *Y, const double *X, const int N, const double alpha);

int prelu_inplace_s (float *X, const int N, const float alpha);
int prelu_inplace_d (double *X, const int N, const double alpha);


int prelu_s (float *Y, const float *X, const int N, const float alpha)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in prelu_s: N (num elements X) must be nonnegative\n"); return 1; }

    if (alpha==0.0f)
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<0.0f) ? 0.0f : X[n]; }
    }
    else if (alpha==1.0f)
    {
        for (n=0; n<N; n++) { Y[n] = X[n]; }
    }
    else
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<0.0f) ? alpha*X[n] : X[n]; }
    }
    

    return 0;
}


int prelu_d (double *Y, const double *X, const int N, const double alpha)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in prelu_d: N (num elements X) must be nonnegative\n"); return 1; }

    if (alpha==0.0)
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<0.0) ? 0.0 : X[n]; }
    }
    else if (alpha==1.0)
    {
        for (n=0; n<N; n++) { Y[n] = X[n]; }
    }
    else
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<0.0) ? alpha*X[n] : X[n]; }
    }
    
    return 0;
}


int prelu_inplace_s (float *X, const int N, const float alpha)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in prelu_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }

    if (alpha==0.0f)
    {
        for (n=0; n<N; n++) { if (X[n]<0.0f) { X[n] = 0.0f; } }
    }
    else if (alpha==1.0f) {}
    else
    {
        for (n=0; n<N; n++) { if(X[n]<0.0f) { X[n] *= alpha; } }
    }

    return 0;
}


int prelu_inplace_d (double *X, const int N, const double alpha)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in prelu_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }

    if (alpha==0.0)
    {
        for (n=0; n<N; n++) { if (X[n]<0.0) { X[n] = 0.0; } }
    }
    else if (alpha==1.0) {}
    else
    {
        for (n=0; n<N; n++) { if(X[n]<0.0) { X[n] *= alpha; } }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
