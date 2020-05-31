//This gets softmax layer-wise activation function for X.

#include <stdio.h>
#include <math.h>
#include <cblas.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int softmax_s (float *Y, const float *X, const char iscolmajor, const int R, const int C, const int dim);
int softmax_d (double *Y, const double *X, const char iscolmajor, const int R, const int C, const int dim);

int softmax_inplace_s (float *X, const char iscolmajor, const int R, const int C, const int dim);
int softmax_inplace_d (double *X, const char iscolmajor, const int R, const int C, const int dim);


int softmax_s (float *Y, const float *X, const char iscolmajor, const int R, const int C, const int dim)
{
    int r, c, n;
    float sm;

    //Checks
    if (R<1) { fprintf(stderr,"error in softmax_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in softmax_s: C (ncols X) must be positive\n"); return 1; }

    for (n=0; n<R*C; n++) { Y[n] = expf(X[n]); }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                sm = cblas_sasum(R,&Y[c*R],1);
                cblas_sscal(R,1.0f/sm,&Y[c*R],1);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                sm = cblas_sasum(R,&Y[c],C);
                cblas_sscal(R,1.0f/sm,&Y[c],C);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                sm = cblas_sasum(C,&Y[r],R);
                cblas_sscal(C,1.0f/sm,&Y[r],R);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                sm = cblas_sasum(C,&Y[r*C],1);
                cblas_sscal(C,1.0f/sm,&Y[r*C],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in softmax_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int softmax_d (double *Y, const double *X, const char iscolmajor, const int R, const int C, const int dim)
{
    int r, c, n;
    double sm;

    //Checks
    if (R<1) { fprintf(stderr,"error in softmax_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in softmax_d: C (ncols X) must be positive\n"); return 1; }

    for (n=0; n<R*C; n++) { Y[n] = exp(X[n]); }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                sm = cblas_dasum(R,&Y[c*R],1);
                cblas_dscal(R,1.0/sm,&Y[c*R],1);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                sm = cblas_dasum(R,&Y[c],C);
                cblas_dscal(R,1.0/sm,&Y[c],C);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                sm = cblas_dasum(C,&Y[r],R);
                cblas_dscal(C,1.0/sm,&Y[r],R);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                sm = cblas_dasum(C,&Y[r*C],1);
                cblas_dscal(C,1.0/sm,&Y[r*C],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in softmax_d: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


int softmax_inplace_s (float *X, const char iscolmajor, const int R, const int C, const int dim)
{
    int r, c, n;
    float sm;

    //Checks
    if (R<1) { fprintf(stderr,"error in softmax_inplace_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in softmax_inplace_s: C (ncols X) must be positive\n"); return 1; }

    for (n=0; n<R*C; n++) { X[n] = expf(X[n]); }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                sm = cblas_sasum(R,&X[c*R],1);
                cblas_sscal(R,1.0f/sm,&X[c*R],1);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                sm = cblas_sasum(R,&X[c],C);
                cblas_sscal(R,1.0f/sm,&X[c],C);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                sm = cblas_sasum(C,&X[r],R);
                cblas_sscal(C,1.0f/sm,&X[r],R);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                sm = cblas_sasum(C,&X[r*C],1);
                cblas_sscal(C,1.0f/sm,&X[r*C],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in softmax_inplace_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int softmax_inplace_d (double *X, const char iscolmajor, const int R, const int C, const int dim)
{
    int r, c, n;
    double sm;

    //Checks
    if (R<1) { fprintf(stderr,"error in softmax_inplace_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in softmax_inplace_d: C (ncols X) must be positive\n"); return 1; }

    for (n=0; n<R*C; n++) { X[n] = exp(X[n]); }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                sm = cblas_dasum(R,&X[c*R],1);
                cblas_dscal(R,1.0/sm,&X[c*R],1);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                sm = cblas_dasum(R,&X[c],C);
                cblas_dscal(R,1.0/sm,&X[c],C);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                sm = cblas_dasum(C,&X[r],R);
                cblas_dscal(C,1.0/sm,&X[r],R);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                sm = cblas_dasum(C,&X[r*C],1);
                cblas_dscal(C,1.0/sm,&X[r*C],1);
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
