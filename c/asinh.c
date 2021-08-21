//Gets inverse hyberbolic sine (asinh) of input X element-wise.
//This has in-place and not-in-place versions.

#include <stdio.h>
#include <math.h>
#include <complex.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int asinh_s (float *Y, const float *X, const size_t N);
int asinh_d (double *Y, const double *X, const size_t N);
int asinh_c (float *Y, const float *X, const size_t N);
int asinh_z (double *Y, const double *X, const size_t N);

int asinh_inplace_s (float *X, const size_t N);
int asinh_inplace_d (double *X, const size_t N);
int asinh_inplace_c (float *X, const size_t N);
int asinh_inplace_z (double *X, const size_t N);


int asinh_s (float *Y, const float *X, const size_t N)
{
    //for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = asinhf(*X); }
    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = logf(*X + sqrtf(fmaf(*X,*X,1.0f))); }

    return 0;
}


int asinh_d (double *Y, const double *X, const size_t N)
{
    //for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = asinh(*X); }
    for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = log(*X + sqrt(fma(*X,*X,1.0))); }
    
    return 0;
}


int asinh_c (float *Y, const float *X, const size_t N)
{
    _Complex float x, y;

    for (size_t n=N; n>0u; --n, X+=2, ++Y)
    {
        //y = casinhf(X[n2]+1.0if*X[n2+1]);
        x = *X + 1.0if**(X+1);
        y = clogf(x + csqrtf(x*x+1.0f));
        *Y = *(float *)&y; *++Y = *((float *)&y+1);
    }
    
    return 0;
}


int asinh_z (double *Y, const double *X, const size_t N)
{
    _Complex double x, y;

    for (size_t n=N; n>0u; --n, X+=2, ++Y)
    {
        //y = casinh(X[n2]+1.0if*X[n2+1]);
        x = *X + 1.0i**(X+1);
        y = clog(x + csqrt(x*x+1.0));
        *Y = *(double *)&y; *++Y = *((double *)&y+1);
    }
    
    return 0;
}


int asinh_inplace_s (float *X, const size_t N)
{
    //for (size_t n=N; n>0u; --n, ++X) { *X = asinhf(*X); }
    for (size_t n=N; n>0u; --n, ++X) { *X = logf(*X + sqrtf(fmaf(*X,*X,1.0f))); }

    return 0;
}


int asinh_inplace_d (double *X, const size_t N)
{
    //for (size_t n=N; n>0u; --n, ++X) { *X = asinh(*X); }
    for (size_t n=N; n>0u; --n, ++X) { *X = log(*X + sqrt(fma(*X,*X,1.0))); }
    
    return 0;
}


int asinh_inplace_c (float *X, const size_t N)
{
    _Complex float x, y;

    for (size_t n=N; n>0u; --n, ++X)
    {
        //y = casinhf(X[n2]+1.0if*X[n2+1]);
        x = *X + 1.0if**(X+1);
        y = clogf(x + csqrtf(x*x+1.0f));
        *X = *(float *)&y; *++X = *((float *)&y+1);
    }
    
    return 0;
}


int asinh_inplace_z (double *X, const size_t N)
{
    _Complex double x, y;

    for (size_t n=N; n>0u; --n, ++X)
    {
        //y = casinh(X[n2]+1.0i*X[n2+1]);
        x = *X + 1.0i**(X+1);
        y = clog(x + csqrt(x*x+1.0));
        *X = *(double *)&y; *++X = *((double *)&y+1);
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
