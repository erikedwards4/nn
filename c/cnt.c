//Gets count of nonzero values for each row or col of X according to dim.
//For complex case, real and imag parts calculated separately.

#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int cnt_s (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int cnt_d (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor);
int cnt_c (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int cnt_z (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor);


int cnt_s (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in cnt_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in cnt_s: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                Y[c] = 0.0f;
                for (r=0; r<R; r++) { Y[c] += (float)(X[c*R+r]!=0.0f); }
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                Y[c] = 0.0f;
                for (r=0; r<R; r++) { Y[c] += (float)(X[r*C+c]!=0.0f); }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                Y[r] = 0.0f;
                for (c=0; c<C; c++) { Y[r] += (float)(X[c*R+r]!=0.0f); }
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                Y[r] = 0.0f;
                for (c=0; c<C; c++) { Y[r] += (float)(X[r*C+c]!=0.0f); }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in cnt_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int cnt_d (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in cnt_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in cnt_d: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                Y[c] = 0.0;
                for (r=0; r<R; r++) { Y[c] += (double)(X[c*R+r]!=0.0); }
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                Y[c] = 0.0;
                for (r=0; r<R; r++) { Y[c] += (double)(X[r*C+c]!=0.0); }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                Y[r] = 0.0;
                for (c=0; c<C; c++) { Y[r] += (double)(X[c*R+r]!=0.0); }
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                Y[r] = 0.0;
                for (c=0; c<C; c++) { Y[r] += (double)(X[r*C+c]!=0.0); }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in cnt_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int cnt_c (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in cnt_c: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in cnt_c: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                Y[c] = 0.0f;
                for (r=0; r<R; r++) { Y[c] += (float)(X[2*(c*R+r)]!=0.0f || X[2*(c*R+r)+1]!=0.0f); }
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                Y[c] = 0.0f;
                for (r=0; r<R; r++) { Y[c] += (float)(X[2*(r*C+c)]!=0.0f || X[2*(r*C+c)+1]!=0.0f); }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                Y[r] = 0.0f;
                for (c=0; c<C; c++) { Y[r] += (float)(X[2*(c*R+r)]!=0.0f || X[2*(c*R+r)+1]!=0.0f); }
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                Y[r] = 0.0f;
                for (c=0; c<C; c++) { Y[r] += (float)(X[2*(r*C+c)]!=0.0f || X[2*(r*C+c)+1]!=0.0f); }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in cnt_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int cnt_z (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in cnt_z: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in cnt_z: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                Y[c] = 0.0;
                for (r=0; r<R; r++) { Y[c] += (double)(X[2*(c*R+r)]!=0.0 || X[2*(c*R+r)+1]!=0.0); }
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                Y[c] = 0.0;
                for (r=0; r<R; r++) { Y[c] += (double)(X[2*(r*C+c)]!=0.0 || X[2*(r*C+c)+1]!=0.0); }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                Y[r] = 0.0;
                for (c=0; c<C; c++) { Y[r] += (double)(X[2*(c*R+r)]!=0.0 || X[2*(c*R+r)+1]!=0.0); }
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                Y[r] = 0.0;
                for (c=0; c<C; c++) { Y[r] += (double)(X[2*(r*C+c)]!=0.0 || X[2*(r*C+c)+1]!=0.0); }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in cnt_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
