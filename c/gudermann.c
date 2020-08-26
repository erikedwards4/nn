//Gets Gudermannian function of input X element-wise.
//This has in-place and not-in-place versions.

#include <stdio.h>
#include <math.h>
#include <complex.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int gudermann_s (float *Y, const float *X, const size_t N);
int gudermann_d (double *Y, const double *X, const size_t N);
int gudermann_c (float *Y, const float *X, const size_t N);
int gudermann_z (double *Y, const double *X, const size_t N);

int gudermann_inplace_s (float *X, const size_t N);
int gudermann_inplace_d (double *X, const size_t N);
int gudermann_inplace_c (float *X, const size_t N);
int gudermann_inplace_z (double *X, const size_t N);


int gudermann_s (float *Y, const float *X, const size_t N)
{
    for (size_t n=0; n<N; ++n, ++X, ++Y) { *Y = 2.0f * atanf(tanhf(0.5f**X)); }

    return 0;
}


int gudermann_d (double *Y, const double *X, const size_t N)
{
    for (size_t n=0; n<N; ++n, ++X, ++Y) { *Y = 2.0 * atan(tanh(0.5**X)); }
    
    return 0;
}


int gudermann_c (float *Y, const float *X, const size_t N)
{
    _Complex float y;

    for (size_t n=0; n<N; ++n, X+=2, ++Y)
    {
        y = 2.0f * catanf(ctanhf(0.5f**X+1.0if*0.5f**(X+1)));
        *Y = *(float *)&y; *++Y = *((float *)&y+1);
    }
    
    return 0;
}


int gudermann_z (double *Y, const double *X, const size_t N)
{
    _Complex double y;
    
    for (size_t n=0; n<N; ++n, X+=2, ++Y)
    {
        y = 2.0 * catan(ctanh(0.5**X+1.0i*0.5**(X+1)));
        *Y = *(double *)&y; *++Y = *((double *)&y+1);
    }
    
    return 0;
}


int gudermann_inplace_s (float *X, const size_t N)
{
    for (size_t n=0; n<N; ++n, ++X) { *X = 2.0f * atanf(tanhf(0.5f**X)); }

    return 0;
}


int gudermann_inplace_d (double *X, const size_t N)
{
    for (size_t n=0; n<N; ++n, ++X) { *X = 2.0 * atan(tanh(0.5**X)); }
    
    return 0;
}


int gudermann_inplace_c (float *X, const size_t N)
{
    _Complex float x;

    for (size_t n=0; n<N; ++n, ++X)
    {
        x = 2.0f * catanf(ctanhf(0.5f**X+1.0if*0.5f**(X+1)));
        *X = *(float *)&x; *++X = *((float *)&x+1);
    }
    
    return 0;
}


int gudermann_inplace_z (double *X, const size_t N)
{
    _Complex double x;

    for (size_t n=0; n<N; ++n, ++X)
    {
        x = 2.0 * catan(ctanh(0.5**X+1.0i*0.5**(X+1)));
        *X = *(double *)&x; *++X = *((double *)&x+1);
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
