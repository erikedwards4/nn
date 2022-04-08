//This outputs X unchanged.
//Could be useful for debugging or for place-holder in a chain.

#include <stdio.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int identity_s (float *Y, const float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X; }

    return 0;
}


int identity_d (double *Y, const double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X; }
    
    return 0;
}


int identity_c (float *Y, const float *X, const size_t N)
{
    for (size_t n=2u*N; n>0; --n, ++X, ++Y) { *Y = *X; }

    return 0;
}


int identity_z (double *Y, const double *X, const size_t N)
{
    for (size_t n=2u*N; n>0; --n, ++X, ++Y) { *Y = *X; }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
