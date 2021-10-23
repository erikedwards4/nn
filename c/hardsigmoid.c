//Gets the hardsigmoid function for each element in X.

//For each element: y = 0,         if x < -3
//                  y = 1,         if x > +3
//                  y = (x+3)/6,   otherwise

#include <stdio.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int hardsigmoid_s (float *Y, const float *X, const size_t N);
int hardsigmoid_d (double *Y, const double *X, const size_t N);

int hardsigmoid_inplace_s (float *X, const size_t N);
int hardsigmoid_inplace_d (double *X, const size_t N);


int hardsigmoid_s (float *Y, const float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        if (*X>3.0f) { *Y = 1.0f; }
        else if (*X<-3.0f) { *Y = 0.0f; }
        else { *Y = (*X+3.0f)/6.0f; }
    }

    return 0;
}


int hardsigmoid_d (double *Y, const double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        if (*X>3.0) { *Y = 1.0; }
        else if (*X<-3.0) { *Y = 0.0; }
        else { *Y = (*X+3.0)/6.0; }
    }
    
    return 0;
}


int hardsigmoid_inplace_s (float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X>3.0f) { *X = 1.0f; }
        else if (*X<-3.0f) { *X = 0.0f; }
        else { *X = (*X+3.0f)/6.0f; }
    }
    return 0;
}


int hardsigmoid_inplace_d (double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X>3.0) { *X = 1.0; }
        else if (*X<-3.0) { *X = 0.0; }
        else { *X = (*X+3.0)/6.0; }
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
