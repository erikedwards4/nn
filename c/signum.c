//This gets signum (sign) activation function for each element of X.
//I also include a thresh (which is 0 by usual definition) for more general use.

//This is a.k.a. the hard limiter.
//This is used in the Hopfield network, for example.

#include <stdio.h>
#include "codee_nn.h"
//#include <time.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int signum_s (float *Y, const float *X, const size_t N, const float thresh)
{
    //struct timespec tic, toc; clock_gettime(CLOCK_REALTIME,&tic);

    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X>thresh) - (*X<thresh); }
    // for (size_t n=N; n>0u; --n, ++X, ++Y)
    // {
    //     if (*X<thresh) { *Y = -1.0f; }
    //     else if (*X>thresh) { *Y = 1.0f; }
    //     else { *Y = 0.0f; }
    // }

    //clock_gettime(CLOCK_REALTIME,&toc); fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);

    return 0;
}


int signum_d (double *Y, const double *X, const size_t N, const double thresh)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X>thresh) - (*X<thresh); }
    
    return 0;
}


int signum_inplace_s (float *X, const size_t N, const float thresh)
{
    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X<thresh) { *X = -1.0f; }
        else if (*X>thresh) { *X = 1.0f; }
    }

    return 0;
}


int signum_inplace_d (double *X, const size_t N, const double thresh)
{
    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X<thresh) { *X = -1.0; }
        else if (*X>thresh) { *X = 1.0; }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
