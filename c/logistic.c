//This gets logistic activation function for each element of X.

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int logistic_s (float *Y, const float *X, const int N, const float alpha);
int logistic_d (double *Y, const double *X, const int N, const double alpha);

int logistic_inplace_s (float *X, const int N, const float alpha);
int logistic_inplace_d (double *X, const int N, const double alpha);


int logistic_s (float *Y, const float *X, const int N, const float alpha)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in logistic_s: N (num elements X) must be nonnegative\n"); return 1; }
    if (alpha<=0.0f) { fprintf(stderr,"error in logistic_s: alpha must be positive\n"); return 1; }

    if (alpha==1.0f)
    {
        for (n=0; n<N; n++) { Y[n] = 1.0f/(1.0f+expf(-X[n])); }
    }
    else
    {
        for (n=0; n<N; n++) { Y[n] = powf(1.0f+expf(-X[n]),-alpha); }
    }

    return 0;
}


int logistic_d (double *Y, const double *X, const int N, const double alpha)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in logistic_d: N (num elements X) must be nonnegative\n"); return 1; }
    if (alpha<=0.0) { fprintf(stderr,"error in logistic_d: alpha must be positive\n"); return 1; }

    if (alpha==1.0)
    {
        for (n=0; n<N; n++) { Y[n] = 1.0/(1.0+exp(-X[n])); }
    }
    else
    {
        for (n=0; n<N; n++) { Y[n] = pow(1.0+exp(-X[n]),-alpha); }
    }
    
    return 0;
}


int logistic_inplace_s (float *X, const int N, const float alpha)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in logistic_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }
    if (alpha<=0.0f) { fprintf(stderr,"error in logistic_inplace_s: alpha must be positive\n"); return 1; }

    if (alpha==1.0f)
    {
        for (n=0; n<N; n++) { X[n] = 1.0f/(1.0f+expf(-X[n])); }
    }
    else
    {
        for (n=0; n<N; n++) { X[n] = powf(1.0f+expf(-X[n]),-alpha); }
    }

    return 0;
}


int logistic_inplace_d (double *X, const int N, const double alpha)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in logistic_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }
    if (alpha<=0.0) { fprintf(stderr,"error in logistic_inplace_d: alpha must be positive\n"); return 1; }

    if (alpha==1.0)
    {
        for (n=0; n<N; n++) { X[n] = 1.0/(1.0+exp(-X[n])); }
    }
    else
    {
        for (n=0; n<N; n++) { X[n] = pow(1.0+exp(-X[n]),-alpha); }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
