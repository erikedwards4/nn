//Gets the Gauss error-function (special function) of each element of X.
//This has in-place and not-in-place versions.

//For complex input, the cerf function is not usually available for complex.h,
//so I use libcerf from: https://jugit.fz-juelich.de/mlz/libcerf.
//See math repo for the complex case.

#include <stdio.h>
#include <math.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int erf_s (float *Y, const float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = erff(*X); }

    return 0;
}


int erf_d (double *Y, const double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = erf(*X); }
    
    return 0;
}


int erf_inplace_s (float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X) { *X = erff(*X); }

    return 0;
}


int erf_inplace_d (double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X) { *X = erf(*X); }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
