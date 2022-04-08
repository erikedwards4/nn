//This gets inverse square-root unit (ISRU) function for each element of X.
//This is an activation function and an algebraic sigmoid function.

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int isru_s (float *Y, const float *X, const size_t N, const float alpha);
int isru_d (double *Y, const double *X, const size_t N, const double alpha);

int isru_inplace_s (float *X, const size_t N, const float alpha);
int isru_inplace_d (double *X, const size_t N, const double alpha);


int isru_s (float *Y, const float *X, const size_t N, const float alpha)
{

    if (alpha<=0.0f) { fprintf(stderr,"error in isru_s: alpha must be positive\n"); return 1; }

    if (alpha==1.0f)
    {
        for (size_t n=0; n<N; ++n, ++X, ++Y) { *Y = *X/sqrtf(1.0f+*X); }
    }
    else
    {
        for (size_t n=0; n<N; ++n, ++X, ++Y) { *Y = *X/sqrtf(1.0f+alpha**X); }
    }

    return 0;
}


int isru_d (double *Y, const double *X, const size_t N, const double alpha)
{

    if (alpha<=0.0) { fprintf(stderr,"error in isru_d: alpha must be positive\n"); return 1; }

    if (alpha==1.0)
    {
        for (size_t n=0; n<N; ++n, ++X, ++Y) { *Y = *X/sqrt(1.0+*X); }
    }
    else
    {
        for (size_t n=0; n<N; ++n, ++X, ++Y) { *Y = *X/sqrt(1.0+alpha**X); }
    }
    
    return 0;
}


int isru_inplace_s (float *X, const size_t N, const float alpha)
{

    if (alpha<=0.0f) { fprintf(stderr,"error in isru_inplace_s: alpha must be positive\n"); return 1; }

    if (alpha==1.0f)
    {
        for (size_t n=0; n<N; ++n, ++X) { *X = *X/sqrtf(1.0f+*X); }
    }
    else
    {
        for (size_t n=0; n<N; ++n, ++X) { *X = *X/sqrtf(1.0f+alpha**X); }
    }

    return 0;
}


int isru_inplace_d (double *X, const size_t N, const double alpha)
{

    if (alpha<=0.0) { fprintf(stderr,"error in isru_inplace_d: alpha must be positive\n"); return 1; }

    if (alpha==1.0)
    {
        for (size_t n=0; n<N; ++n, ++X) { *X = *X/sqrt(1.0+*X); }
    }
    else
    {
        for (size_t n=0; n<N; ++n, ++X) { *X = *X/sqrt(1.0+alpha**X); }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
