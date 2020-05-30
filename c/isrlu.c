//This gets inverse square-root unit (ISRU) function for each element of X.
//This is an activation function and an algebraic sigmoid function.

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int isrlu_s (float *Y, const float *X, const int N, const float alpha);
int isrlu_d (double *Y, const double *X, const int N, const double alpha);

int isrlu_inplace_s (float *X, const int N, const float alpha);
int isrlu_inplace_d (double *X, const int N, const double alpha);


int isrlu_s (float *Y, const float *X, const int N, const float alpha)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in isrlu_s: N (num elements X) must be nonnegative\n"); return 1; }
    if (alpha<=0.0f) { fprintf(stderr,"error in isrlu_s: alpha must be positive\n"); return 1; }

    if (alpha==1.0f)
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<0.0f) ? X[n]/sqrtf(1.0f+X[n]) : X[n]; }
    }
    else
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<0.0f) ? X[n]/sqrtf(1.0f+alpha*X[n]) : X[n]; }
    }

    return 0;
}


int isrlu_d (double *Y, const double *X, const int N, const double alpha)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in isrlu_d: N (num elements X) must be nonnegative\n"); return 1; }
    if (alpha<=0.0) { fprintf(stderr,"error in isrlu_d: alpha must be positive\n"); return 1; }

    if (alpha==1.0)
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<0.0) ? X[n]/sqrt(1.0+X[n]) : X[n]; }
    }
    else
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<0.0) ? X[n]/sqrt(1.0+alpha*X[n]) : X[n]; }
    }
    
    return 0;
}


int isrlu_inplace_s (float *X, const int N, const float alpha)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in isrlu_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }
    if (alpha<=0.0f) { fprintf(stderr,"error in isrlu_inplace_s: alpha must be positive\n"); return 1; }

    if (alpha==1.0f)
    {
        for (n=0; n<N; n++) { if (X[n]<0.0f) { X[n] /= sqrtf(1.0f+X[n]); } }
    }
    else
    {
        for (n=0; n<N; n++) { if (X[n]<0.0f) { X[n] /= sqrtf(1.0f+alpha*X[n]); } }
    }

    return 0;
}


int isrlu_inplace_d (double *X, const int N, const double alpha)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in isrlu_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }
    if (alpha<=0.0) { fprintf(stderr,"error in isrlu_inplace_d: alpha must be positive\n"); return 1; }

    if (alpha==1.0)
    {
        for (n=0; n<N; n++) { if (X[n]<0.0) { X[n] /= sqrt(1.0+X[n]); } }
    }
    else
    {
        for (n=0; n<N; n++) { if (X[n]<0.0) { X[n] /= sqrt(1.0+alpha*X[n]); } }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
