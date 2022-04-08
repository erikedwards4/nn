//This gets hardtanh function, which is:
//y = min,  if x<min
//y = max,  if x>max
//y = x,    otherwise

#include <stdio.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int hardtanh_s (float *Y, const float *X, const size_t N, const float a, const float b)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        *Y = (*X<a) ? a : (*X>b) ? b : *X;
    }

    return 0;
}


int hardtanh_d (double *Y, const double *X, const size_t N, const double a, const double b)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        *Y = (*X<a) ? a : (*X>b) ? b : *X;
    }
    
    return 0;
}


int hardtanh_inplace_s (float *X, const size_t N, const float a, const float b)
{
    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X<a) { *X = a; }
        else if (*X>b) { *X = b; }
    }

    return 0;
}


int hardtanh_inplace_d (double *X, const size_t N, const double a, const double b)
{
    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X<a) { *X = a; }
        else if (*X>b) { *X = b; }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
