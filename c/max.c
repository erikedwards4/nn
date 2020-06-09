//Gets maximum of values for each row or col of X according to dim.
//For complex case, real and imag parts calculated separately.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int max_s (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int max_d (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor);
int max_c (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int max_z (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor);


int max_s (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;
    float mx;

    //Checks
    if (R<1) { fprintf(stderr,"error in max_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in max_s: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                mx = X[c*R];
                for (r=1; r<R; r++)
                {
                    if (X[c*R+r]>mx) { mx = X[c*R+r]; }
                }
                Y[c] = mx;
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mx = X[c];
                for (r=1; r<R; r++)
                {
                    if (X[r*C+c]>mx) { mx = X[r*C+c]; }
                }
                Y[c] = mx;
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                mx = X[r];
                for (c=1; c<C; c++)
                {
                    if (X[c*R+r]>mx) { mx = X[c*R+r]; }
                }
                Y[r] = mx;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                mx = X[r*C];
                for (c=1; c<C; c++)
                {
                    if (X[r*C+c]>mx) { mx = X[r*C+c]; }
                }
                Y[r] = mx;
            }
        }
    }
    else
    {
        fprintf(stderr,"error in max_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int max_d (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;
    double mx;

    //Checks
    if (R<1) { fprintf(stderr,"error in max_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in max_d: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                mx = X[c*R];
                for (r=1; r<R; r++)
                {
                    if (X[c*R+r]>mx) { mx = X[c*R+r]; }
                }
                Y[c] = mx;
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mx = X[c];
                for (r=1; r<R; r++)
                {
                    if (X[r*C+c]>mx) { mx = X[r*C+c]; }
                }
                Y[c] = mx;
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                mx = X[r];
                for (c=1; c<C; c++)
                {
                    if (X[c*R+r]>mx) { mx = X[c*R+r]; }
                }
                Y[r] = mx;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                mx = X[r*C];
                for (c=1; c<C; c++)
                {
                    if (X[r*C+c]>mx) { mx = X[r*C+c]; }
                }
                Y[r] = mx;
            }
        }
    }
    else
    {
        fprintf(stderr,"error in max_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int max_c (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;
    float mx, a2;

    //Checks
    if (R<1) { fprintf(stderr,"error in max_c: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in max_c: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                mx = X[2*c*R]*X[2*c*R] + X[2*c*R+1]*X[2*c*R+1];
                for (r=1; r<R; r++)
                {
                    a2 = X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1];
                    if (a2>mx) { mx = a2; }
                }
                Y[c] = sqrtf(mx);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mx = X[2*c]*X[2*c] + X[2*c+1]*X[2*c+1];
                for (r=1; r<R; r++)
                {
                    a2 = X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1];
                    if (a2>mx) { mx = a2; }
                }
                Y[c] = sqrtf(mx);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                mx = X[2*r]*X[2*r] + X[2*r+1]*X[2*r+1];
                for (c=1; c<C; c++)
                {
                    a2 = X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1];
                    if (a2>mx) { mx = a2; }
                }
                Y[r] = sqrtf(mx);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                mx = X[2*r*C]*X[2*r*C] + X[2*r*C+1]*X[2*r*C+1];
                for (c=1; c<C; c++)
                {
                    a2 = X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1];
                    if (a2>mx) { mx = a2; }
                }
                Y[r] = sqrtf(mx);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in max_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int max_z (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;
    double mx, a2;

    //Checks
    if (R<1) { fprintf(stderr,"error in max_z: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in max_z: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                mx = X[2*c*R]*X[2*c*R] + X[2*c*R+1]*X[2*c*R+1];
                for (r=1; r<R; r++)
                {
                    a2 = X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1];
                    if (a2>mx) { mx = a2; }
                }
                Y[c] = sqrt(mx);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mx = X[2*c]*X[2*c] + X[2*c+1]*X[2*c+1];
                for (r=1; r<R; r++)
                {
                    a2 = X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1];
                    if (a2>mx) { mx = a2; }
                }
                Y[c] = sqrt(mx);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                mx = X[2*r]*X[2*r] + X[2*r+1]*X[2*r+1];
                for (c=1; c<C; c++)
                {
                    a2 = X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1];
                    if (a2>mx) { mx = a2; }
                }
                Y[r] = sqrt(mx);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                mx = X[2*r*C]*X[2*r*C] + X[2*r*C+1]*X[2*r*C+1];
                for (c=1; c<C; c++)
                {
                    a2 = X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1];
                    if (a2>mx) { mx = a2; }
                }
                Y[r] = sqrt(mx);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in max_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
