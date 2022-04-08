//Gets the hardswish function [Howard et al. 2019].
//Howard A, Sandler M, Chu G, Chen LC, Chen B, Tan M, Wang W, Zhu Y, Pang R, Vasudevan V, Le QV. 2019. Searching for MobileNetV3. Proc IEEE/CVF ICCV: 1314-24.

//For each element: y = 0,         if x<=-3
//For each element: y = x,         if x>=+3
//                  y = x*(x+3)/6, otherwise
#include <stdio.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int hardswish_s (float *Y, const float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        if (*X>3.0f) { *Y = *X; }
        else if (*X<=-3.0f) { *Y = 0.0f; }
        else { *Y = *X * (*X+3.0f)/6.0f; }
    }

    return 0;
}


int hardswish_d (double *Y, const double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        if (*X>3.0) { *Y = *X; }
        else if (*X<=-3.0) { *Y = 0.0; }
        else { *Y = *X * (*X+3.0)/6.0; }
    }
    
    return 0;
}


int hardswish_inplace_s (float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X<=-3.0) { *X = 0.0; }
        else if (*X<3.0) { *X *= (*X+3.0)/6.0; }
    }
    return 0;
}


int hardswish_inplace_d (double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X<=-3.0) { *X = 0.0; }
        else if (*X<3.0) { *X *= (*X+3.0)/6.0; }
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
