//Gets sine (sin) of input X element-wise.
//This has in-place and not-in-place versions.

#include <stdio.h>
#include "codee_nn.h"
//#include <string.h>
#include <math.h>
#include <complex.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int sin_s (float *Y, const float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = sinf(*X); }

    return 0;
}


int sin_d (double *Y, const double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = sin(*X); }
    
    return 0;
}


int sin_c (float *Y, const float *X, const size_t N)
{
    _Complex float y;
    
    for (size_t n=N; n>0u; --n, X+=2, ++Y)
    {
        y = csinf(*X + 1.0if**(X+1));
        *Y = *(float *)&y; *++Y = *((float *)&y+1);
    }
    
    return 0;
}


int sin_z (double *Y, const double *X, const size_t N)
{
    _Complex double y;
    
    for (size_t n=N; n>0u; --n, X+=2, ++Y)
    {
        y = csin(*X + 1.0i**(X+1));
        *Y = *(double *)&y; *++Y = *((double *)&y+1);
    }
    
    return 0;
}


int sin_inplace_s (float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X) { *X = sinf(*X); }

    return 0;
}


int sin_inplace_d (double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X) { *X = sin(*X); }
    
    return 0;
}


int sin_inplace_c (float *X, const size_t N)
{
    _Complex float y;

    //struct timespec tic, toc; clock_gettime(CLOCK_REALTIME,&tic);
    
    // for (size_t n2=0; n2<2*N; n2+=2)
    // {
    //     y = csinf(X[n2]+1.0if*X[n2+1]);
    //     memcpy(&X[n2],(float *)&y,2*sizeof(float));
    // }

    for (size_t n=N; n>0u; --n, ++X)
    {
        y = csinf(*X + 1.0if**(X+1));
        *X = *(float *)&y; *++X = *((float *)&y+1);
    }

    //clock_gettime(CLOCK_REALTIME,&toc); fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);
    
    return 0;
}


int sin_inplace_z (double *X, const size_t N)
{
    _Complex double y;

    for (size_t n=N; n>0u; --n, ++X)
    {
        y = csin(*X + 1.0i**(X+1));
        *X = *(double *)&y; *++X = *((double *)&y+1);
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
