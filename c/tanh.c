//Gets hyberbolic tangent (tanh) of input X element-wise.
// y = tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))
//This has in-place and not-in-place versions.

#include <stdio.h>
#include <math.h>
#include <complex.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int tanh_s (float *Y, const float *X, const size_t N);
int tanh_d (double *Y, const double *X, const size_t N);
int tanh_c (float *Y, const float *X, const size_t N);
int tanh_z (double *Y, const double *X, const size_t N);

int tanh_inplace_s (float *X, const size_t N);
int tanh_inplace_d (double *X, const size_t N);
int tanh_inplace_c (float *X, const size_t N);
int tanh_inplace_z (double *X, const size_t N);


int tanh_s (float *Y, const float *X, const size_t N)
{
    float xp, xm;

    //for (size_t n=0; n<N; ++n, ++X, ++Y) { *Y = tanhf(*X); }
    for (size_t n=0; n<N; ++n, ++X, ++Y) { xp = expf(*X); xm = expf(-*X); *Y = (xp-xm)/(xp+xm); }

    return 0;
}


int tanh_d (double *Y, const double *X, const size_t N)
{
    double xp, xm;
    
    //for (size_t n=0; n<N; ++n, ++X, ++Y) { *Y = tanh(*X); }
    for (size_t n=0; n<N; ++n, ++X, ++Y) { xp = exp(*X); xm = exp(-*X); *Y = (xp-xm)/(xp+xm); }
    
    return 0;
}


int tanh_c (float *Y, const float *X, const size_t N)
{
    _Complex float y, xp, xm;

    for (size_t n=0; n<N; ++n, X+=2, ++Y)
    {
        //y = ctanhf(X[n2]+1.0if*X[n2+1]);
        xp = cexpf(*X + 1.0if**(X+1));
        xm = cexpf(-*X - 1.0if**(X+1));
        y = (xp-xm) / (xp+xm);
        *Y = *(float *)&y; *++Y = *((float *)&y+1);
    }
    
    return 0;
}


int tanh_z (double *Y, const double *X, const size_t N)
{
    _Complex double y, xp, xm;

    for (size_t n=0; n<N; ++n, X+=2, ++Y)
    {
        xp = cexp(*X + 1.0i**(X+1));
        xm = cexp(-*X - 1.0i**(X+1));
        y = (xp-xm) / (xp+xm);
        *Y = *(double *)&y; *++Y = *((double *)&y+1);
    }
    
    return 0;
}


int tanh_inplace_s (float *X, const size_t N)
{
    float xp, xm;

    //for (size_t n=0; n<N; ++n, ++X) { *X = tanhf(*X); }
    //for (size_t n=0; n<N; ++n, ++X) { x = expf(2.0f**X); *X = (x-1.0f)/(x+1.0f); }   //this is faster, but less numerical accuracy
    for (size_t n=0; n<N; ++n, ++X) { xp = expf(*X); xm = expf(-*X); *X = (xp-xm)/(xp+xm); }

    return 0;
}


int tanh_inplace_d (double *X, const size_t N)
{
    double xp, xm;

    //for (size_t n=0; n<N; ++n, ++X) { *X = tanh(*X); }
    for (size_t n=0; n<N; ++n, ++X) { xp = exp(*X); xm = exp(-*X); *X = (xp-xm)/(xp+xm); }
    
    return 0;
}


int tanh_inplace_c (float *X, const size_t N)
{
    _Complex float y, xp, xm;

    for (size_t n=0; n<N; ++n, ++X)
    {
        //x = ctanhf(X[n2]+1.0if*X[n2+1]);
        xp = cexpf(*X + 1.0if**(X+1));
        xm = cexpf(-*X - 1.0if**(X+1));
        y = (xp-xm) / (xp+xm);
        *X = *(float *)&y; *++X = *((float *)&y+1);
    }
    
    return 0;
}


int tanh_inplace_z (double *X, const size_t N)
{
    _Complex double y, xp, xm;
    
    for (size_t n=0; n<N; ++n, ++X)
    {
        //x = ctanh(X[n2]+1.0i*X[n2+1]);
        xp = cexp(*X + 1.0i**(X+1));
        xm = cexp(-*X - 1.0i**(X+1));
        y = (xp-xm) / (xp+xm);
        *X = *(double *)&y; *++X = *((double *)&y+1);
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
