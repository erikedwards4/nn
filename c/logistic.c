//This gets logistic activation function for each element of X.

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int logistic_s (float *Y, const float *X, const size_t N, const float alpha);
int logistic_d (double *Y, const double *X, const size_t N, const double alpha);

int logistic_inplace_s (float *X, const size_t N, const float alpha);
int logistic_inplace_d (double *X, const size_t N, const double alpha);


int logistic_s (float *Y, const float *X, const size_t N, const float alpha)
{

    if (alpha<=0.0f) { fprintf(stderr,"error in logistic_s: alpha must be positive\n"); return 1; }

    if (alpha==1.0f)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = 1.0f/(1.0f+expf(-*X)); }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = powf(1.0f+expf(-*X),-alpha); }
    }

    return 0;
}


int logistic_d (double *Y, const double *X, const size_t N, const double alpha)
{

    if (alpha<=0.0) { fprintf(stderr,"error in logistic_d: alpha must be positive\n"); return 1; }

    if (alpha==1.0)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = 1.0/(1.0+exp(-*X)); }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = pow(1.0+exp(-*X),-alpha); }
    }
    
    return 0;
}


int logistic_inplace_s (float *X, const size_t N, const float alpha)
{

    if (alpha<=0.0f) { fprintf(stderr,"error in logistic_inplace_s: alpha must be positive\n"); return 1; }

    if (alpha==1.0f)
    {
        for (size_t n=N; n>0u; --n, ++X) { *X = 1.0f/(1.0f+expf(-*X)); }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X) { *X = powf(1.0f+expf(-*X),-alpha); }
    }

    return 0;
}


int logistic_inplace_d (double *X, const size_t N, const double alpha)
{

    if (alpha<=0.0) { fprintf(stderr,"error in logistic_inplace_d: alpha must be positive\n"); return 1; }

    if (alpha==1.0)
    {
        for (size_t n=N; n>0u; --n, ++X) { *X = 1.0/(1.0+exp(-*X)); }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X) { *X = pow(1.0+exp(-*X),-alpha); }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
