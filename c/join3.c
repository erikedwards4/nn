//Joins 2 inputs X1, X2 into 1 output Y

#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int join3_s (float *Y, const float *X1, const float *X2, const float *X3, const int R1, const int R2, const int R3, const int C1, const int C2, const int C3, const int dim, const char iscolmajor);
int join3_d (double *Y, const double *X1, const double *X2, const double *X3, const int R1, const int R2, const int R3, const int C1, const int C2, const int C3, const int dim, const char iscolmajor);
int join3_c (float *Y, const float *X1, const float *X2, const float *X3, const int R1, const int R2, const int R3, const int C1, const int C2, const int C3, const int dim, const char iscolmajor);
int join3_z (double *Y, const double *X1, const double *X2, const double *X3, const int R1, const int R2, const int R3, const int C1, const int C2, const int C3, const int dim, const char iscolmajor);


int join3_s (float *Y, const float *X1, const float *X2, const float *X3, const int R1, const int R2, const int R3, const int C1, const int C2, const int C3, const int dim, const char iscolmajor)
{
    const int R = R1+R2+R3, C = C1+C2+C3;
    int r, c;

    //Checks
    if (R1<1) { fprintf(stderr,"error in join3_s: R1 (nrows X1) must be positive\n"); return 1; }
    if (R2<1) { fprintf(stderr,"error in join3_s: R2 (nrows X2) must be positive\n"); return 1; }
    if (R3<1) { fprintf(stderr,"error in join3_s: R3 (nrows X3) must be positive\n"); return 1; }
    if (C1<1) { fprintf(stderr,"error in join3_s: C1 (ncols X1) must be positive\n"); return 1; }
    if (C2<1) { fprintf(stderr,"error in join3_s: C2 (ncols X2) must be positive\n"); return 1; }
    if (C3<1) { fprintf(stderr,"error in join3_s: C3 (ncols X3) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                cblas_scopy(R1,&X1[c*R1],1,&Y[c*R],1);
                cblas_scopy(R2,&X2[c*R2],1,&Y[c*R+R1],1);
                cblas_scopy(R3,&X3[c*R3],1,&Y[c*R+R1+R2],1);
            }
        }
        else
        {
            cblas_scopy(R1*C1,X1,1,Y,1);
            cblas_scopy(R2*C2,X2,1,&Y[R1*C1],1);
            cblas_scopy(R3*C3,X3,1,&Y[R1*C1+R2*C2],1);
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            cblas_scopy(R1*C1,X1,1,Y,1);
            cblas_scopy(R2*C2,X2,1,&Y[R1*C1],1);
            cblas_scopy(R3*C3,X3,1,&Y[R1*C1+R2*C2],1);
        }
        else
        {
            for (r=0; r<R; r++)
            {
                cblas_scopy(R1,&X1[r*C1],1,&Y[r*C],1);
                cblas_scopy(R2,&X2[r*C2],1,&Y[r*C+C1],1);
                cblas_scopy(R3,&X3[r*C3],1,&Y[r*C+C1+C2],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in join3_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int join3_d (double *Y, const double *X1, const double *X2, const double *X3, const int R1, const int R2, const int R3, const int C1, const int C2, const int C3, const int dim, const char iscolmajor)
{
    const int R = R1+R2+R3, C = C1+C2+C3;
    int r, c;

    //Checks
    if (R1<1) { fprintf(stderr,"error in join3_d: R1 (nrows X1) must be positive\n"); return 1; }
    if (R2<1) { fprintf(stderr,"error in join3_d: R2 (nrows X2) must be positive\n"); return 1; }
    if (R3<1) { fprintf(stderr,"error in join3_d: R3 (nrows X3) must be positive\n"); return 1; }
    if (C1<1) { fprintf(stderr,"error in join3_d: C1 (ncols X1) must be positive\n"); return 1; }
    if (C2<1) { fprintf(stderr,"error in join3_d: C2 (ncols X2) must be positive\n"); return 1; }
    if (C3<1) { fprintf(stderr,"error in join3_d: C3 (ncols X3) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                cblas_dcopy(R1,&X1[c*R1],1,&Y[c*R],1);
                cblas_dcopy(R2,&X2[c*R2],1,&Y[c*R+R1],1);
                cblas_dcopy(R3,&X3[c*R3],1,&Y[c*R+R1+R2],1);
            }
        }
        else
        {
            cblas_dcopy(R1*C1,X1,1,Y,1);
            cblas_dcopy(R2*C2,X2,1,&Y[R1*C1],1);
            cblas_dcopy(R3*C3,X3,1,&Y[R1*C1+R2*C2],1);
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            cblas_dcopy(R1*C1,X1,1,Y,1);
            cblas_dcopy(R2*C2,X2,1,&Y[R1*C1],1);
            cblas_dcopy(R3*C3,X3,1,&Y[R1*C1+R2*C2],1);
        }
        else
        {
            for (r=0; r<R; r++)
            {
                cblas_dcopy(R1,&X1[r*C1],1,&Y[r*C],1);
                cblas_dcopy(R2,&X2[r*C2],1,&Y[r*C+C1],1);
                cblas_dcopy(R3,&X3[r*C3],1,&Y[r*C+C1+C2],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in join3_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int join3_c (float *Y, const float *X1, const float *X2, const float *X3, const int R1, const int R2, const int R3, const int C1, const int C2, const int C3, const int dim, const char iscolmajor)
{
    const int R = R1+R2+R3, C = C1+C2+C3;
    int r, c;

    //Checks
    if (R1<1) { fprintf(stderr,"error in join3_c: R1 (nrows X1) must be positive\n"); return 1; }
    if (R2<1) { fprintf(stderr,"error in join3_c: R2 (nrows X2) must be positive\n"); return 1; }
    if (C1<1) { fprintf(stderr,"error in join3_c: C1 (ncols X1) must be positive\n"); return 1; }
    if (C2<1) { fprintf(stderr,"error in join3_c: C2 (ncols X2) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                cblas_ccopy(R1,&X1[2*c*R1],1,&Y[2*c*R],1);
                cblas_ccopy(R2,&X2[2*c*R2],1,&Y[2*(c*R+R1)],1);
                cblas_ccopy(R3,&X3[2*c*R3],1,&Y[2*(c*R+R1+R2)],1);
            }
        }
        else
        {
            cblas_ccopy(R1*C1,X1,1,Y,1);
            cblas_ccopy(R2*C2,X2,1,&Y[2*R1*C1],1);
            cblas_ccopy(R3*C3,X3,1,&Y[2*(R1*C1+R2*C2)],1);
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            cblas_ccopy(R1*C1,X1,1,Y,1);
            cblas_ccopy(R2*C2,X2,1,&Y[2*R1*C1],1);
            cblas_ccopy(R3*C3,X3,1,&Y[2*(R1*C1+R2*C2)],1);
        }
        else
        {
            for (r=0; r<R; r++)
            {
                cblas_ccopy(R1,&X1[2*r*C1],1,&Y[2*r*C],1);
                cblas_ccopy(R2,&X2[2*r*C2],1,&Y[2*(r*C+C1)],1);
                cblas_ccopy(R3,&X3[2*r*C3],1,&Y[2*(r*C+C1+C2)],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in join3_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int join3_z (double *Y, const double *X1, const double *X2, const double *X3, const int R1, const int R2, const int R3, const int C1, const int C2, const int C3, const int dim, const char iscolmajor)
{
    const int R = R1+R2+R3, C = C1+C2+C3;
    int r, c;

    //Checks
    if (R1<1) { fprintf(stderr,"error in join3_z: R1 (nrows X1) must be positive\n"); return 1; }
    if (R2<1) { fprintf(stderr,"error in join3_z: R2 (nrows X2) must be positive\n"); return 1; }
    if (C1<1) { fprintf(stderr,"error in join3_z: C1 (ncols X1) must be positive\n"); return 1; }
    if (C2<1) { fprintf(stderr,"error in join3_z: C2 (ncols X2) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                cblas_zcopy(R1,&X1[2*c*R1],1,&Y[2*c*R],1);
                cblas_zcopy(R2,&X2[2*c*R2],1,&Y[2*(c*R+R1)],1);
                cblas_zcopy(R3,&X3[2*c*R3],1,&Y[2*(c*R+R1+R2)],1);
            }
        }
        else
        {
            cblas_zcopy(R1*C1,X1,1,Y,1);
            cblas_zcopy(R2*C2,X2,1,&Y[2*R1*C1],1);
            cblas_zcopy(R3*C3,X3,1,&Y[2*(R1*C1+R2*C2)],1);
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            cblas_zcopy(R1*C1,X1,1,Y,1);
            cblas_zcopy(R2*C2,X2,1,&Y[2*R1*C1],1);
            cblas_zcopy(R3*C3,X3,1,&Y[2*(R1*C1+R2*C2)],1);
        }
        else
        {
            for (r=0; r<R; r++)
            {
                cblas_zcopy(R1,&X1[2*r*C1],1,&Y[2*r*C],1);
                cblas_zcopy(R2,&X2[2*r*C2],1,&Y[2*(r*C+C1)],1);
                cblas_zcopy(R3,&X3[2*r*C3],1,&Y[2*(r*C+C1+C2)],1);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in join3_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
