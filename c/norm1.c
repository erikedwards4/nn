//Gets L1 norm (sum of absolute-values) for each row or col of X according to dim.
//For complex case, output is real.

#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int norm1_s (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int norm1_d (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor);
int norm1_c (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int norm1_z (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor);


int norm1_s (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in norm1_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in norm1_s: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_sasum(R,&X[c*R],1);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_sasum(R,&X[c],C);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_sasum(C,&X[r],R);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_sasum(C,&X[r*C],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in norm1_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int norm1_d (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in norm1_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in norm1_d: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_dasum(R,&X[c*R],1);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_dasum(R,&X[c],C);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_dasum(C,&X[r],R);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_dasum(C,&X[r*C],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in norm1_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int norm1_c (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in norm1_c: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in norm1_c: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_scasum(R,&X[2*c*R],1);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_scasum(R,&X[2*c],C);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_scasum(C,&X[2*r],R);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_scasum(C,&X[2*r*C],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in norm1_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int norm1_z (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in norm1_z: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in norm1_z: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_dzasum(R,&X[2*c*R],1);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_dzasum(R,&X[2*c],C);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_dzasum(C,&X[2*r],R);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_dzasum(C,&X[2*r*C],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in norm1_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
