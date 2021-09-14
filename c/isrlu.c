//This gets inverse square-root unit (ISRU) function for each element of X.
//This is an activation function and an algebraic sigmoid function.

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int isrlu_s (float *Y, const float *X, const size_t N, const float alpha);
int isrlu_d (double *Y, const double *X, const size_t N, const double alpha);

int isrlu_inplace_s (float *X, const size_t N, const float alpha);
int isrlu_inplace_d (double *X, const size_t N, const double alpha);


int isrlu_s (float *Y, const float *X, const size_t N, const float alpha)
{

    if (alpha<=0.0f) { fprintf(stderr,"error in isrlu_s: alpha must be positive\n"); return 1; }

    if (alpha==1.0f)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0f) ? *X/sqrtf(1.0f+*X) : *X; }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0f) ? *X/sqrtf(1.0f+alpha**X) : *X; }
    }

    return 0;
}


int isrlu_d (double *Y, const double *X, const size_t N, const double alpha)
{

    if (alpha<=0.0) { fprintf(stderr,"error in isrlu_d: alpha must be positive\n"); return 1; }

    if (alpha==1.0)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0) ? *X/sqrt(1.0+*X) : *X; }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0) ? *X/sqrt(1.0+alpha**X) : *X; }
    }
    
    return 0;
}


int isrlu_inplace_s (float *X, const size_t N, const float alpha)
{

    if (alpha<=0.0f) { fprintf(stderr,"error in isrlu_inplace_s: alpha must be positive\n"); return 1; }

    if (alpha==1.0f)
    {
        for (size_t n=N; n>0u; --n, ++X) { if (*X<0.0f) { *X /= sqrtf(1.0f+*X); } }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X) { if (*X<0.0f) { *X /= sqrtf(1.0f+alpha**X); } }
    }

    return 0;
}


int isrlu_inplace_d (double *X, const size_t N, const double alpha)
{

    if (alpha<=0.0) { fprintf(stderr,"error in isrlu_inplace_d: alpha must be positive\n"); return 1; }

    if (alpha==1.0)
    {
        for (size_t n=N; n>0u; --n, ++X) { if (*X<0.0) { *X /= sqrt(1.0+*X); } }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X) { if (*X<0.0) { *X /= sqrt(1.0+alpha**X); } }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
