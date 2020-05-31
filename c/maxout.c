//This gets maxout layer-wise activation function for X.
//This is just the max along rows or cols of X according to dim.

#include <stdio.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int maxout_s (float *Y, const float *X, const char iscolmajor, const int R, const int C, const int dim);
int maxout_d (double *Y, const double *X, const char iscolmajor, const int R, const int C, const int dim);


int maxout_s (float *Y, const float *X, const char iscolmajor, const int R, const int C, const int dim)
{
    int r, c, n;
    float mx;

    //Checks
    if (R<1) { fprintf(stderr,"error in maxout_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in maxout_s: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                n = c*R; mx = X[n];
                while (++n<(c+1)*R) { if (X[n]>mx) { mx = X[n]; } }
                Y[c] = mx;
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                n = c; mx = X[n]; n += C;
                while (n<R*C) { if (X[n]>mx) { mx = X[n]; } n+=C; }
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
                n = r; mx = X[n]; n += R;
                while (n<R*C) { if (X[n]>mx) { mx = X[n]; } n+=R; }
                Y[r] = mx;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                n = r*C; mx = X[n];
                while (++n<(r+1)*C) { if (X[n]>mx) { mx = X[n]; } }
                Y[r] = mx;
            }
        }
    }
    else
    {
        fprintf(stderr,"error in maxout_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int maxout_d (double *Y, const double *X, const char iscolmajor, const int R, const int C, const int dim)
{
    int r, c, n;
    double mx;

    //Checks
    if (R<1) { fprintf(stderr,"error in maxout_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in maxout_d: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                n = c*R; mx = X[n];
                while (++n<(c+1)*R) { if (X[n]>mx) { mx = X[n]; } }
                Y[c] = mx;
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                n = c; mx = X[n]; n += C;
                while (n<R*C) { if (X[n]>mx) { mx = X[n]; } n+=C; }
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
                n = r; mx = X[n]; n += R;
                while (n<R*C) { if (X[n]>mx) { mx = X[n]; } n+=R; }
                Y[r] = mx;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                n = r*C; mx = X[n];
                while (++n<(r+1)*C) { if (X[n]>mx) { mx = X[n]; } }
                Y[r] = mx;
            }
        }
    }
    else
    {
        fprintf(stderr,"error in maxout_d: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
