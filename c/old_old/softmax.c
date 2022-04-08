//This gets softmax layer-wise activation function for X.

#include <stdio.h>
#include <math.h>
#include <cblas.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int softmax_s (float *Y, const float *X, const size_t R, const size_t C, const char iscolmajor, const size_t dim);
int softmax_d (double *Y, const double *X, const size_t R, const size_t C, const char iscolmajor, const size_t dim);

int softmax_inplace_s (float *X, const size_t R, const size_t C, const char iscolmajor, const size_t dim);
int softmax_inplace_d (double *X, const size_t R, const size_t C, const char iscolmajor, const size_t dim);


int softmax_s (float *Y, const float *X, const size_t R, const size_t C, const char iscolmajor, const size_t dim)
{
    float sm;
struct timespec tic, toc; clock_gettime(CLOCK_REALTIME,&tic);
    for (size_t n=0; n<R*C; ++n, ++X, ++Y) { *Y = expf(*X); }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t c=0; c<C; ++c)
            {
                sm = cblas_sasum((int)R,&Y[c*R],1);
                cblas_sscal((int)R,1.0f/sm,&Y[c*R],1);
            }
        }
        else
        {
            for (size_t c=0; c<C; ++c)
            {
                sm = cblas_sasum((int)R,&Y[c],(int)C);
                cblas_sscal((int)R,1.0f/sm,&Y[c],(int)C);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (size_t r=0; r<R; ++r)
            {
                sm = cblas_sasum((int)C,&Y[r],(int)R);
                cblas_sscal((int)C,1.0f/sm,&Y[r],(int)R);
            }
        }
        else
        {
            for (size_t r=0; r<R; ++r)
            {
                sm = cblas_sasum((int)C,&Y[r*C],1);
                cblas_sscal((int)C,1.0f/sm,&Y[r*C],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in softmax_s: dim must be 0 or 1.\n"); return 1;
    }
clock_gettime(CLOCK_REALTIME,&toc); fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);
    return 0;
}


int softmax_d (double *Y, const double *X, const size_t R, const size_t C, const char iscolmajor, const size_t dim)
{
    double sm;

    for (size_t n=0; n<R*C; ++n, ++X, ++Y) { *Y = exp(*X); }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t c=0; c<C; ++c)
            {
                sm = cblas_dasum((int)R,&Y[c*R],1);
                cblas_dscal((int)R,1.0/sm,&Y[c*R],1);
            }
        }
        else
        {
            for (size_t c=0; c<C; ++c)
            {
                sm = cblas_dasum((int)R,&Y[c],(int)C);
                cblas_dscal((int)R,1.0/sm,&Y[c],(int)C);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (size_t r=0; r<R; ++r)
            {
                sm = cblas_dasum((int)C,&Y[r],(int)R);
                cblas_dscal((int)C,1.0/sm,&Y[r],(int)R);
            }
        }
        else
        {
            for (size_t r=0; r<R; ++r)
            {
                sm = cblas_dasum((int)C,&Y[r*C],1);
                cblas_dscal((int)C,1.0/sm,&Y[r*C],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in softmax_d: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


int softmax_inplace_s (float *X, const size_t R, const size_t C, const char iscolmajor, const size_t dim)
{
    float sm;

    for (size_t n=0; n<R*C; ++n) { *X = expf(*X); }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t c=0; c<C; ++c)
            {
                sm = cblas_sasum((int)R,&X[c*R],1);
                cblas_sscal((int)R,1.0f/sm,&X[c*R],1);
            }
        }
        else
        {
            for (size_t c=0; c<C; ++c)
            {
                sm = cblas_sasum((int)R,&X[c],(int)C);
                cblas_sscal((int)R,1.0f/sm,&X[c],(int)C);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (size_t r=0; r<R; ++r)
            {
                sm = cblas_sasum((int)C,&X[r],(int)R);
                cblas_sscal((int)C,1.0f/sm,&X[r],(int)R);
            }
        }
        else
        {
            for (size_t r=0; r<R; ++r)
            {
                sm = cblas_sasum((int)C,&X[r*C],1);
                cblas_sscal((int)C,1.0f/sm,&X[r*C],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in softmax_inplace_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int softmax_inplace_d (double *X, const size_t R, const size_t C, const char iscolmajor, const size_t dim)
{
    double sm;

    for (size_t n=0; n<R*C; ++n) { *X = exp(*X); }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t c=0; c<C; ++c)
            {
                sm = cblas_dasum((int)R,&X[c*R],1);
                cblas_dscal((int)R,1.0/sm,&X[c*R],1);
            }
        }
        else
        {
            for (size_t c=0; c<C; ++c)
            {
                sm = cblas_dasum((int)R,&X[c],(int)C);
                cblas_dscal((int)R,1.0/sm,&X[c],(int)C);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (size_t r=0; r<R; ++r)
            {
                sm = cblas_dasum((int)C,&X[r],(int)R);
                cblas_dscal((int)C,1.0/sm,&X[r],(int)R);
            }
        }
        else
        {
            for (size_t r=0; r<R; ++r)
            {
                sm = cblas_dasum((int)C,&X[r*C],1);
                cblas_dscal((int)C,1.0/sm,&X[r*C],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in softmax_inplace_d: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
