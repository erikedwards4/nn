//Gets L2-norm (root-mean-square) of each row or col of X according to dim.
//For complex case, output is real.
//This works in place.

#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int norm2_s (float *Y, float *X, const int R, const int C, const int dim, const char iscolmajor);
int norm2_d (double *Y, double *X, const int R, const int C, const int dim, const char iscolmajor);
int norm2_c (float *Y, float *X, const int R, const int C, const int dim, const char iscolmajor);
int norm2_z (double *Y, double *X, const int R, const int C, const int dim, const char iscolmajor);


int norm2_s (float *Y, float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in norm2_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in norm2_s: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_snrm2(R,&X[c*R],1);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_snrm2(R,&X[c],C);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_snrm2(C,&X[r],R);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_snrm2(C,&X[r*C],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in norm2_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int norm2_d (double *Y, double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in norm2_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in norm2_d: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_dnrm2(R,&X[c*R],1);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_dnrm2(R,&X[c],C);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_dnrm2(C,&X[r],R);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_dnrm2(C,&X[r*C],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in norm2_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int norm2_c (float *Y, float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in norm2_c: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in norm2_c: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_scnrm2(R,&X[2*c*R],1);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_scnrm2(R,&X[2*c],C);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_scnrm2(C,&X[2*r],R);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_scnrm2(C,&X[2*r*C],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in norm2_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int norm2_z (double *Y, double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in norm2_z: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in norm2_z: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_dznrm2(R,&X[2*c*R],1);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_dznrm2(R,&X[2*c],C);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_dznrm2(C,&X[2*r],R);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_dznrm2(C,&X[2*r*C],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in norm2_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
