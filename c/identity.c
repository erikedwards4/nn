//This outputs X unchanged.
//Could be useful for debugging or for place-holder in a chain.

#include <stdio.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int identity_s (float *Y, const float *X, const size_t N);
int identity_d (double *Y, const double *X, const size_t N);
int identity_c (float *Y, const float *X, const size_t N);
int identity_z (double *Y, const double *X, const size_t N);


int identity_s (float *Y, const float *X, const size_t N)
{
    for (size_t n=0; n<N; ++n, ++X, ++Y) { *Y = *X; }

    return 0;
}


int identity_d (double *Y, const double *X, const size_t N)
{
    for (size_t n=0; n<N; ++n, ++X, ++Y) { *Y = *X; }
    
    return 0;
}


int identity_c (float *Y, const float *X, const size_t N)
{
    for (size_t n=0; n<2*N; ++n, ++X, ++Y) { *Y = *X; }

    return 0;
}


int identity_z (double *Y, const double *X, const size_t N)
{
    for (size_t n=0; n<2*N; ++n, ++X, ++Y) { *Y = *X; }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
