//Gets mean of each row or col of X according to dim.
//For complex case, real and imag parts calculated separately.

#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int mean_s (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int mean_d (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor);
int mean_c (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int mean_z (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor);


int mean_s (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    const float o = 1.0f;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in mean_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in mean_s: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_sdot(R,&X[c*R],1,&o,0) / R;
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_sdot(R,&X[c],C,&o,0) / R;
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_sdot(C,&X[r],R,&o,0) / C;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_sdot(C,&X[r*C],1,&o,0) / C;
            }
        }
    }
    else
    {
        fprintf(stderr,"error in mean_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int mean_d (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    const double o = 1.0;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in mean_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in mean_d: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_ddot(R,&X[c*R],1,&o,0) / R;
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_ddot(R,&X[c],C,&o,0) / R;
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_ddot(C,&X[r],R,&o,0) / C;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_ddot(C,&X[r*C],1,&o,0) / C;
            }
        }
    }
    else
    {
        fprintf(stderr,"error in mean_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int mean_c (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    const float o =  1.0f;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in mean_c: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in mean_c: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_sdot(R,&X[2*c*R],2,&o,0) / R;
                Y[c+1] = cblas_sdot(R,&X[2*c*R+1],2,&o,0) / R;
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_sdot(R,&X[2*c],2*C,&o,0) / R;
                Y[c+1] = cblas_sdot(R,&X[2*c+1],2*C,&o,0) / R;
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_sdot(C,&X[2*r],2*R,&o,0) / C;
                Y[r+1] = cblas_sdot(C,&X[2*r+1],2*R,&o,0) / C;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_sdot(C,&X[2*r*C],2,&o,0) / C;
                Y[r+1] = cblas_sdot(C,&X[2*r*C+1],2,&o,0) / C;
            }
        }
    }
    else
    {
        fprintf(stderr,"error in mean_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int mean_z (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    const double o =  1.0;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in mean_z: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in mean_z: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_ddot(R,&X[2*c*R],2,&o,0) / R;
                Y[c+1] = cblas_ddot(R,&X[2*c*R+1],2,&o,0) / R;
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                Y[c] = cblas_ddot(R,&X[2*c],2*C,&o,0) / R;
                Y[c+1] = cblas_ddot(R,&X[2*c+1],2*C,&o,0) / R;
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_ddot(C,&X[2*r],2*R,&o,0) / C;
                Y[r+1] = cblas_ddot(C,&X[2*r+1],2*R,&o,0) / C;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                Y[r] = cblas_ddot(C,&X[2*r*C],2,&o,0) / C;
                Y[r+1] = cblas_ddot(C,&X[2*r*C+1],2,&o,0) / C;
            }
        }
    }
    else
    {
        fprintf(stderr,"error in mean_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
