//Gets range (max-min) of values for each row or col of X according to dim.
//For complex case, real and imag parts calculated separately.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int range_s (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int range_d (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor);
int range_c (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int range_z (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor);


int range_s (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;
    float mn, mx;

    //Checks
    if (R<1) { fprintf(stderr,"error in range_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in range_s: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                mn = mx = X[c*R];
                for (r=1; r<R; r++)
                {
                    if (X[c*R+r]<mn) { mn = X[c*R+r]; }
                    else if (X[c*R+r]>mx) { mx = X[c*R+r]; }
                }
                Y[c] = mx - mn;
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mn = mx = X[c];
                for (r=1; r<R; r++)
                {
                    if (X[r*C+c]<mn) { mn = X[r*C+c]; }
                    else if (X[r*C+c]>mx) { mx = X[r*C+c]; }
                }
                Y[c] = mx - mn;
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                mn = mx = X[r];
                for (c=1; c<C; c++)
                {
                    if (X[c*R+r]<mn) { mn = X[c*R+r]; }
                    else if (X[c*R+r]>mx) { mx = X[c*R+r]; }
                }
                Y[r] = mx - mn;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                mn = mx = X[r*C];
                for (c=1; c<C; c++)
                {
                    if (X[r*C+c]<mn) { mn = X[r*C+c]; }
                    else if (X[r*C+c]>mx) { mx = X[r*C+c]; }
                }
                Y[r] = mx - mn;
            }
        }
    }
    else
    {
        fprintf(stderr,"error in range_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int range_d (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;
    double mn, mx;

    //Checks
    if (R<1) { fprintf(stderr,"error in range_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in range_d: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                mn = mx = X[c*R];
                for (r=1; r<R; r++)
                {
                    if (X[c*R+r]<mn) { mn = X[c*R+r]; }
                    else if (X[c*R+r]>mx) { mx = X[c*R+r]; }
                }
                Y[c] = mx - mn;
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mn = mx = X[c];
                for (r=1; r<R; r++)
                {
                    if (X[r*C+c]<mn) { mn = X[r*C+c]; }
                    else if (X[r*C+c]>mx) { mx = X[r*C+c]; }
                }
                Y[c] = mx - mn;
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                mn = mx = X[r];
                for (c=1; c<C; c++)
                {
                    if (X[c*R+r]<mn) { mn = X[c*R+r]; }
                    else if (X[c*R+r]>mx) { mx = X[c*R+r]; }
                }
                Y[r] = mx - mn;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                mn = mx = X[r*C];
                for (c=1; c<C; c++)
                {
                    if (X[r*C+c]<mn) { mn = X[r*C+c]; }
                    else if (X[r*C+c]>mx) { mx = X[r*C+c]; }
                }
                Y[r] = mx - mn;
            }
        }
    }
    else
    {
        fprintf(stderr,"error in range_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int range_c (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;
    float mn, mx, a2;

    //Checks
    if (R<1) { fprintf(stderr,"error in range_c: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in range_c: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                mn = mx = X[2*c*R]*X[2*c*R] + X[2*c*R+1]*X[2*c*R+1];
                for (r=1; r<R; r++)
                {
                    a2 = X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1];
                    if (a2<mn) { mn = a2; } else if (a2>mx) { mx = a2; }
                }
                Y[c] = sqrtf(mx) - sqrtf(mn);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mn = mx = X[2*c]*X[2*c] + X[2*c+1]*X[2*c+1];
                for (r=1; r<R; r++)
                {
                    a2 = X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1];
                    if (a2<mn) { mn = a2; } else if (a2>mx) { mx = a2; }
                }
                Y[c] = sqrtf(mx) - sqrtf(mn);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                mn = mx = X[2*r]*X[2*r] + X[2*r+1]*X[2*r+1];
                for (c=1; c<C; c++)
                {
                    a2 = X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1];
                    if (a2<mn) { mn = a2; } else if (a2>mx) { mx = a2; }
                }
                Y[r] = sqrtf(mx) - sqrtf(mn);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                mn = mx = X[2*r*C]*X[2*r*C] + X[2*r*C+1]*X[2*r*C+1];
                for (c=1; c<C; c++)
                {
                    a2 = X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1];
                    if (a2<mn) { mn = a2; } else if (a2>mx) { mx = a2; }
                }
                Y[r] = sqrtf(mx) - sqrtf(mn);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in range_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int range_z (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;
    double mn, mx, a2;

    //Checks
    if (R<1) { fprintf(stderr,"error in range_z: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in range_z: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                mn = mx = X[2*c*R]*X[2*c*R] + X[2*c*R+1]*X[2*c*R+1];
                for (r=1; r<R; r++)
                {
                    a2 = X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1];
                    if (a2<mn) { mn = a2; } else if (a2>mx) { mx = a2; }
                }
                Y[c] = sqrt(mx) - sqrt(mn);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mn = mx = X[2*c]*X[2*c] + X[2*c+1]*X[2*c+1];
                for (r=1; r<R; r++)
                {
                    a2 = X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1];
                    if (a2<mn) { mn = a2; } else if (a2>mx) { mx = a2; }
                }
                Y[c] = sqrt(mx) - sqrt(mn);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                mn = mx = X[2*r]*X[2*r] + X[2*r+1]*X[2*r+1];
                for (c=1; c<C; c++)
                {
                    a2 = X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1];
                    if (a2<mn) { mn = a2; } else if (a2>mx) { mx = a2; }
                }
                Y[r] = sqrt(mx) - sqrt(mn);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                mn = mx = X[2*r*C]*X[2*r*C] + X[2*r*C+1]*X[2*r*C+1];
                for (c=1; c<C; c++)
                {
                    a2 = X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1];
                    if (a2<mn) { mn = a2; } else if (a2>mx) { mx = a2; }
                }
                Y[r] = sqrt(mx) - sqrt(mn);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in range_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
