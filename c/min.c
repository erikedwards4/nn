//Gets minimum of values for each row or col of X according to dim.
//For complex case, real and imag parts calculated separately.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int min_s (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int min_d (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor);
int min_c (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int min_z (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor);


int min_s (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;
    float mn;

    //Checks
    if (R<1) { fprintf(stderr,"error in min_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in min_s: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                mn = X[c*R];
                for (r=1; r<R; r++)
                {
                    if (X[c*R+r]<mn) { mn = X[c*R+r]; }
                }
                Y[c] = mn;
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mn = X[c];
                for (r=1; r<R; r++)
                {
                    if (X[r*C+c]<mn) { mn = X[r*C+c]; }
                }
                Y[c] = mn;
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                mn = X[r];
                for (c=1; c<C; c++)
                {
                    if (X[c*R+r]<mn) { mn = X[c*R+r]; }
                }
                Y[r] = mn;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                mn = X[r*C];
                for (c=1; c<C; c++)
                {
                    if (X[r*C+c]<mn) { mn = X[r*C+c]; }
                }
                Y[r] = mn;
            }
        }
    }
    else
    {
        fprintf(stderr,"error in min_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int min_d (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;
    double mn;

    //Checks
    if (R<1) { fprintf(stderr,"error in min_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in min_d: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                mn = X[c*R];
                for (r=1; r<R; r++)
                {
                    if (X[c*R+r]<mn) { mn = X[c*R+r]; }
                }
                Y[c] = mn;
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mn = X[c];
                for (r=1; r<R; r++)
                {
                    if (X[r*C+c]<mn) { mn = X[r*C+c]; }
                }
                Y[c] = mn;
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                mn = X[r];
                for (c=1; c<C; c++)
                {
                    if (X[c*R+r]<mn) { mn = X[c*R+r]; }
                }
                Y[r] = mn;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                mn = X[r*C];
                for (c=1; c<C; c++)
                {
                    if (X[r*C+c]<mn) { mn = X[r*C+c]; }
                }
                Y[r] = mn;
            }
        }
    }
    else
    {
        fprintf(stderr,"error in min_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int min_c (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;
    float mn, a2;

    //Checks
    if (R<1) { fprintf(stderr,"error in min_c: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in min_c: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                mn = X[2*c*R]*X[2*c*R] + X[2*c*R+1]*X[2*c*R+1];
                for (r=1; r<R; r++)
                {
                    a2 = X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1];
                    if (a2<mn) { mn = a2; }
                }
                Y[c] = sqrtf(mn);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mn = X[2*c]*X[2*c] + X[2*c+1]*X[2*c+1];
                for (r=1; r<R; r++)
                {
                    a2 = X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1];
                    if (a2<mn) { mn = a2; }
                }
                Y[c] = sqrtf(mn);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                mn = X[2*r]*X[2*r] + X[2*r+1]*X[2*r+1];
                for (c=1; c<C; c++)
                {
                    a2 = X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1];
                    if (a2<mn) { mn = a2; }
                }
                Y[r] = sqrtf(mn);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                mn = X[2*r*C]*X[2*r*C] + X[2*r*C+1]*X[2*r*C+1];
                for (c=1; c<C; c++)
                {
                    a2 = X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1];
                    if (a2<mn) { mn = a2; }
                }
                Y[r] = sqrtf(mn);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in min_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int min_z (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;
    double mn, a2;

    //Checks
    if (R<1) { fprintf(stderr,"error in min_z: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in min_z: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                mn = X[2*c*R]*X[2*c*R] + X[2*c*R+1]*X[2*c*R+1];
                for (r=1; r<R; r++)
                {
                    a2 = X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1];
                    if (a2<mn) { mn = a2; }
                }
                Y[c] = sqrt(mn);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mn = X[2*c]*X[2*c] + X[2*c+1]*X[2*c+1];
                for (r=1; r<R; r++)
                {
                    a2 = X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1];
                    if (a2<mn) { mn = a2; }
                }
                Y[c] = sqrt(mn);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                mn = X[2*r]*X[2*r] + X[2*r+1]*X[2*r+1];
                for (c=1; c<C; c++)
                {
                    a2 = X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1];
                    if (a2<mn) { mn = a2; }
                }
                Y[r] = sqrt(mn);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                mn = X[2*r*C]*X[2*r*C] + X[2*r*C+1]*X[2*r*C+1];
                for (c=1; c<C; c++)
                {
                    a2 = X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1];
                    if (a2<mn) { mn = a2; }
                }
                Y[r] = sqrt(mn);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in min_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
