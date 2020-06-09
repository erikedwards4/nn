//Gets standard deviation of each row or col of X according to dim.
//For complex case, output is real.
//This works in place.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int std_s (float *Y, float *X, const int R, const int C, const int dim, const char iscolmajor, const char biased);
int std_d (double *Y, double *X, const int R, const int C, const int dim, const char iscolmajor, const char biased);
int std_c (float *Y, float *X, const int R, const int C, const int dim, const char iscolmajor, const char biased);
int std_z (double *Y, double *X, const int R, const int C, const int dim, const char iscolmajor, const char biased);


int std_s (float *Y, float *X, const int R, const int C, const int dim, const char iscolmajor, const char biased)
{
    const float o = 1.0f;
    float m, den;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in std_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in std_s: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (biased) { den = (float)R; } else { den = (float)(R-1); }
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                m = -cblas_sdot(R,&X[c*R],1,&o,0) / R;
                cblas_saxpy(R,m,&o,0,&X[c*R],1);
                Y[c] = sqrtf(cblas_snrm2(R,&X[c*R],1)/den);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                m = -cblas_sdot(R,&X[c],C,&o,0) / R;
                cblas_saxpy(R,m,&o,0,&X[c],C);
                Y[c] = sqrtf(cblas_snrm2(R,&X[c],C)/den);
            }
        }
    }
    else if (dim==1)
    {
        if (biased) { den = (float)C; } else { den = (float)(C-1); }
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                m = -cblas_sdot(C,&X[r],R,&o,0) / C;
                cblas_saxpy(C,m,&o,0,&X[r],R);
                Y[r] = sqrtf(cblas_snrm2(C,&X[r],R)/den);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                m = -cblas_sdot(C,&X[r*C],1,&o,0) / C;
                cblas_saxpy(C,m,&o,0,&X[r*C],1);
                Y[r] = sqrtf(cblas_snrm2(C,&X[r*C],1)/den);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in std_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int std_d (double *Y, double *X, const int R, const int C, const int dim, const char iscolmajor, const char biased)
{
    const double o = 1.0;
    double m, den;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in std_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in std_d: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (biased) { den = (double)R; } else { den = (double)(R-1); }
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                m = -cblas_ddot(R,&X[c*R],1,&o,0) / R;
                cblas_daxpy(R,m,&o,0,&X[c*R],1);
                Y[c] = sqrt(cblas_dnrm2(R,&X[c*R],1)/den);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                m = -cblas_ddot(R,&X[c],C,&o,0) / R;
                cblas_daxpy(R,m,&o,0,&X[c],C);
                Y[c] = sqrt(cblas_dnrm2(R,&X[c],C)/den);
            }
        }
    }
    else if (dim==1)
    {
        if (biased) { den = (double)C; } else { den = (double)(C-1); }
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                m = -cblas_ddot(C,&X[r],R,&o,0) / C;
                cblas_daxpy(C,m,&o,0,&X[r],R);
                Y[r] = sqrt(cblas_dnrm2(C,&X[r],R)/den);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                m = -cblas_ddot(C,&X[r*C],1,&o,0) / C;
                cblas_daxpy(C,m,&o,0,&X[r*C],1);
                Y[r] = sqrt(cblas_dnrm2(C,&X[r*C],1)/den);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in std_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int std_c (float *Y, float *X, const int R, const int C, const int dim, const char iscolmajor, const char biased)
{
    const float o[2] =  {1.0f,0.0f};
    _Complex float m;
    float den;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in std_c: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in std_c: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (biased) { den = (float)R; } else { den = (float)(R-1); }
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                m = -cblas_cdotu(R,&X[2*c*R],1,&o[0],0) / R;
                cblas_caxpy(R,(float *)&m,&o[0],0,&X[2*c*R],1);
                Y[c] = sqrtf(cblas_scnrm2(R,&X[2*c*R],1)/den);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                m = -cblas_cdotu(R,&X[2*c],C,&o[0],0) / R;
                cblas_caxpy(R,(float *)&m,&o[0],0,&X[2*c],C);
                Y[c] = sqrtf(cblas_scnrm2(R,&X[2*c],C)/den);
            }
        }
    }
    else if (dim==1)
    {
        if (biased) { den = (float)C; } else { den = (float)(C-1); }
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                m = -cblas_cdotu(C,&X[2*r],R,&o[0],0) / C;
                cblas_caxpy(C,(float *)&m,&o[0],0,&X[2*r],R);
                Y[r] = sqrtf(cblas_scnrm2(C,&X[2*r],R)/den);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                m = -cblas_cdotu(C,&X[2*r*C],1,&o[0],0) / C;
                cblas_caxpy(C,(float *)&m,&o[0],0,&X[2*r*C],1);
                Y[r] = sqrtf(cblas_scnrm2(C,&X[2*r*C],1)/den);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in std_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int std_z (double *Y, double *X, const int R, const int C, const int dim, const char iscolmajor, const char biased)
{
    const double o[2] =  {1.0,0.0};
    _Complex double m;
    double den;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in std_z: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in std_z: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (biased) { den = (double)R; } else { den = (double)(R-1); }
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                m = -cblas_zdotu(R,&X[2*c*R],1,&o[0],0) / R;
                cblas_zaxpy(R,(double *)&m,&o[0],0,&X[2*c*R],1);
                Y[c] = sqrt(cblas_dznrm2(R,&X[2*c*R],1)/den);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                m = -cblas_zdotu(R,&X[2*c],C,&o[0],0) / R;
                cblas_zaxpy(R,(double *)&m,&o[0],0,&X[2*c],C);
                Y[c] = sqrt(cblas_dznrm2(R,&X[2*c],C)/den);
            }
        }
    }
    else if (dim==1)
    {
        if (biased) { den = (double)C; } else { den = (double)(C-1); }
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                m = -cblas_zdotu(C,&X[2*r],R,&o[0],0) / C;
                cblas_zaxpy(C,(double *)&m,&o[0],0,&X[2*r],R);
                Y[r] = sqrt(cblas_dznrm2(C,&X[2*r],R)/den);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                m = -cblas_zdotu(C,&X[2*r*C],1,&o[0],0) / C;
                cblas_zaxpy(C,(double *)&m,&o[0],0,&X[2*r*C],1);
                Y[r] = sqrt(cblas_dznrm2(C,&X[2*r*C],1)/den);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in std_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
