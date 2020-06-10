//Splits 1 input X into 3 outputs Y1, Y2, Y3

#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int split3_s (float *Y1, float *Y2, float *Y3, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int split3_d (double *Y1, double *Y2, double *Y3, const double *X, const int R, const int C, const int dim, const char iscolmajor);
int split3_c (float *Y1, float *Y2, float *Y3, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int split3_z (double *Y1, double *Y2, double *Y3, const double *X, const int R, const int C, const int dim, const char iscolmajor);


int split3_s (float *Y1, float *Y2, float *Y3, const float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    const int R3 = R/3, C3 = C/3;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in split3_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in split3_s: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                cblas_scopy(R3,&X[c*R],1,&Y1[c*R3],1);
                cblas_scopy(R3,&X[c*R+R3],1,&Y2[c*R3],1);
                cblas_scopy(R3,&X[c*R+2*R3],1,&Y3[c*R3],1);
            }
        }
        else
        {
            cblas_scopy(R*C3,X,1,Y1,1);
            cblas_scopy(R*C3,&X[R*C3],1,Y2,1);
            cblas_scopy(R*C3,&X[2*R*C3],1,Y3,1);
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            cblas_scopy(R*C3,X,1,Y1,1);
            cblas_scopy(R*C3,&X[R*C3],1,Y2,1);
            cblas_scopy(R*C3,&X[2*R*C3],1,Y3,1);
        }
        else
        {
            for (r=0; r<R; r++)
            {
                cblas_scopy(C3,&X[r*C],1,&Y1[r*C3],1);
                cblas_scopy(C3,&X[r*C+C3],1,&Y2[r*C3],1);
                cblas_scopy(C3,&X[r*C+2*C3],1,&Y3[r*C3],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in split3_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int split3_d (double *Y1, double *Y2, double *Y3, const double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    const int R3 = R/3, C3 = C/3;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in split3_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in split3_d: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                cblas_dcopy(R3,&X[c*R],1,&Y1[c*R3],1);
                cblas_dcopy(R3,&X[c*R+R3],1,&Y2[c*R3],1);
                cblas_dcopy(R3,&X[c*R+2*R3],1,&Y3[c*R3],1);
            }
        }
        else
        {
            cblas_dcopy(R*C3,X,1,Y1,1);
            cblas_dcopy(R*C3,&X[R*C3],1,Y2,1);
            cblas_dcopy(R*C3,&X[2*R*C3],1,Y3,1);
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            cblas_dcopy(R*C3,X,1,Y1,1);
            cblas_dcopy(R*C3,&X[R*C3],1,Y2,1);
            cblas_dcopy(R*C3,&X[2*R*C3],1,Y3,1);
        }
        else
        {
            for (r=0; r<R; r++)
            {
                cblas_dcopy(C3,&X[r*C],1,&Y1[r*C3],1);
                cblas_dcopy(C3,&X[r*C+C3],1,&Y2[r*C3],1);
                cblas_dcopy(C3,&X[r*C+2*C3],1,&Y3[r*C3],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in split3_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int split3_c (float *Y1, float *Y2, float *Y3, const float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    const int R3 = R/3, C3 = C/3;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in split3_c: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in split3_c: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                cblas_ccopy(R3,&X[2*c*R],1,&Y1[2*c*R3],1);
                cblas_ccopy(R3,&X[2*(c*R+R3)],1,&Y2[2*c*R3],1);
                cblas_ccopy(R3,&X[2*(c*R+2*R3)],1,&Y3[2*c*R3],1);
            }
        }
        else
        {
            cblas_ccopy(R*C3,X,1,Y1,1);
            cblas_ccopy(R*C3,&X[2*R*C3],1,Y2,1);
            cblas_ccopy(R*C3,&X[4*R*C3],1,Y3,1);
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            cblas_ccopy(R*C3,X,1,Y1,1);
            cblas_ccopy(R*C3,&X[2*R*C3],1,Y2,1);
            cblas_ccopy(R*C3,&X[4*R*C3],1,Y3,1);
        }
        else
        {
            for (r=0; r<R; r++)
            {
                cblas_ccopy(C3,&X[2*r*C],1,&Y1[2*r*C3],1);
                cblas_ccopy(C3,&X[2*(r*C+C3)],1,&Y2[2*r*C3],1);
                cblas_ccopy(C3,&X[2*(r*C+2*C3)],1,&Y3[2*r*C3],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in split3_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int split3_z (double *Y1, double *Y2, double *Y3, const double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    const int R3 = R/3, C3 = C/3;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in split3_z: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in split3_z: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                cblas_zcopy(R3,&X[2*c*R],1,&Y1[2*c*R3],1);
                cblas_zcopy(R3,&X[2*(c*R+R3)],1,&Y2[2*c*R3],1);
                cblas_zcopy(R3,&X[2*(c*R+2*R3)],1,&Y3[2*c*R3],1);
            }
        }
        else
        {
            cblas_zcopy(R*C3,X,1,Y1,1);
            cblas_zcopy(R*C3,&X[2*R*C3],1,Y2,1);
            cblas_zcopy(R*C3,&X[4*R*C3],1,Y3,1);
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            cblas_zcopy(R*C3,X,1,Y1,1);
            cblas_zcopy(R*C3,&X[2*R*C3],1,Y2,1);
            cblas_zcopy(R*C3,&X[4*R*C3],1,Y3,1);
        }
        else
        {
            for (r=0; r<R; r++)
            {
                cblas_zcopy(C3,&X[2*r*C],1,&Y1[2*r*C3],1);
                cblas_zcopy(C3,&X[2*(r*C+C3)],1,&Y2[2*r*C3],1);
                cblas_zcopy(C3,&X[2*(r*C+2*C3)],1,&Y3[2*r*C3],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in split3_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
