//Splits 1 input X into 2 outputs Y1, Y2

#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int split2_s (float *Y1, float *Y2, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int split2_d (double *Y1, double *Y2, const double *X, const int R, const int C, const int dim, const char iscolmajor);
int split2_c (float *Y1, float *Y2, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int split2_z (double *Y1, double *Y2, const double *X, const int R, const int C, const int dim, const char iscolmajor);


int split2_s (float *Y1, float *Y2, const float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    const int R2 = R/2, C2 = C/2;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in split2_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in split2_s: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                cblas_scopy(R2,&X[c*R],1,&Y1[c*R2],1);
                cblas_scopy(R2,&X[c*R+R2],1,&Y2[c*R2],1);
            }
        }
        else
        {
            cblas_scopy(R*C2,X,1,Y1,1);
            cblas_scopy(R*C2,&X[R*C2],1,Y2,1);
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            cblas_scopy(R*C2,X,1,Y1,1);
            cblas_scopy(R*C2,&X[R*C2],1,Y2,1);
        }
        else
        {
            for (r=0; r<R; r++)
            {
                cblas_scopy(C2,&X[r*C],1,&Y1[r*C2],1);
                cblas_scopy(C2,&X[r*C+C2],1,&Y2[r*C2],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in split2_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int split2_d (double *Y1, double *Y2, const double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    const int R2 = R/2, C2 = C/2;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in split2_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in split2_d: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                cblas_dcopy(R2,&X[c*R],1,&Y1[c*R2],1);
                cblas_dcopy(R2,&X[c*R+R2],1,&Y2[c*R2],1);
            }
        }
        else
        {
            cblas_dcopy(R*C2,X,1,Y1,1);
            cblas_dcopy(R*C2,&X[R*C2],1,Y2,1);
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            cblas_dcopy(R*C2,X,1,Y1,1);
            cblas_dcopy(R*C2,&X[R*C2],1,Y2,1);
        }
        else
        {
            for (r=0; r<R; r++)
            {
                cblas_dcopy(C2,&X[r*C],1,&Y1[r*C2],1);
                cblas_dcopy(C2,&X[r*C+C2],1,&Y2[r*C2],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in split2_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int split2_c (float *Y1, float *Y2, const float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    const int R2 = R/2, C2 = C/2;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in split2_c: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in split2_c: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                cblas_ccopy(R2,&X[2*c*R],1,&Y1[c*R],1);
                cblas_ccopy(R2,&X[2*c*R+R],1,&Y2[c*R],1);
            }
        }
        else
        {
            cblas_ccopy(R*C2,X,1,Y1,1);
            cblas_ccopy(R*C2,&X[R*C],1,Y2,1);
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            cblas_ccopy(R*C2,X,1,Y1,1);
            cblas_ccopy(R*C2,&X[R*C],1,Y2,1);
        }
        else
        {
            for (r=0; r<R; r++)
            {
                cblas_ccopy(C2,&X[2*r*C],1,&Y1[r*C],1);
                cblas_ccopy(C2,&X[2*r*C+C],1,&Y2[r*C],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in split2_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int split2_z (double *Y1, double *Y2, const double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    const int R2 = R/2, C2 = C/2;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in split2_z: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in split2_z: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                cblas_zcopy(R2,&X[2*c*R],1,&Y1[c*R],1);
                cblas_zcopy(R2,&X[2*c*R+R],1,&Y2[c*R],1);
            }
        }
        else
        {
            cblas_zcopy(R*C2,X,1,Y1,1);
            cblas_zcopy(R*C2,&X[R*C],1,Y2,1);
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            cblas_zcopy(R*C2,X,1,Y1,1);
            cblas_zcopy(R*C2,&X[R*C],1,Y2,1);
        }
        else
        {
            for (r=0; r<R; r++)
            {
                cblas_zcopy(C2,&X[2*r*C],1,&Y1[r*C],1);
                cblas_zcopy(C2,&X[2*r*C+C],1,&Y2[r*C],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in split2_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
