//Step activation function for each element of X (a.k.a., comparator).
//I include a thresh (which is 0 by usual definition) for more general use.

#include <stdio.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int step_s (float *Y, const float *X, const size_t N, const float thresh)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<thresh) ? 0.0f : 1.0f; }

    return 0;
}


int step_d (double *Y, const double *X, const size_t N, const double thresh)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<thresh) ? 0.0 : 1.0; }
    
    return 0;
}


int step_inplace_s (float *X, const size_t N, const float thresh)
{
    for (size_t n=N; n>0u; --n, ++X) { *X = (*X<thresh) ? 0.0f : 1.0f; }

    return 0;
}


int step_inplace_d (double *X, const size_t N, const double thresh)
{
    for (size_t n=N; n>0u; --n, ++X) { *X = (*X<thresh) ? 0.0 : 1.0; }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
