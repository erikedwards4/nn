//Gets maximum of values for each row or col of X according to dim.
//For complex case, gets max absolute value (output is real).

//The inplace versions replace the first R or C elements of X,
//without regard to iscolmajor since output is a vector.


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
//#include <time.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int max_s (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int max_d (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor);
int max_c (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int max_z (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor);

int max_inplace_s (float *X, const int R, const int C, const int dim, const char iscolmajor);
int max_inplace_d (double *X, const int R, const int C, const int dim, const char iscolmajor);
int max_inplace_c (float *X, const int R, const int C, const int dim, const char iscolmajor);
int max_inplace_z (double *X, const int R, const int C, const int dim, const char iscolmajor);


int max_s (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;
    float mx;
    //struct timespec tic, toc;

    //Checks
    if (R<1) { fprintf(stderr,"error in max_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in max_s: C (ncols X) must be positive\n"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                mx = X[c*R];
                for (r=1; r<R; r++) { if (X[c*R+r]>mx) { mx = X[c*R+r]; } }
                Y[c] = mx;
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mx = X[c];
                for (r=1; r<R; r++) { if (X[r*C+c]>mx) { mx = X[r*C+c]; } }
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
                for (c=1; c<C; c++) { if (X[c*R+r]>mx) { mx = X[c*R+r]; } }
                Y[r] = mx;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                mx = X[r*C];
                for (c=1; c<C; c++) { if (X[r*C+c]>mx) { mx = X[r*C+c]; } }
                Y[r] = mx;
            }
        }
    }
    else
    {
        fprintf(stderr,"error in max_s: dim must be 0 or 1.\n"); return 1;
    }
    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);

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
                for (r=1; r<R; r++) { if (X[c*R+r]>mx) { mx = X[c*R+r]; } }
                Y[c] = mx;
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mx = X[c];
                for (r=1; r<R; r++) { if (X[r*C+c]>mx) { mx = X[r*C+c]; } }
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
                for (c=1; c<C; c++) { if (X[c*R+r]>mx) { mx = X[c*R+r]; } }
                Y[r] = mx;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                mx = X[r*C];
                for (c=1; c<C; c++) { if (X[r*C+c]>mx) { mx = X[r*C+c]; } }
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

    //Checks
    if (R<1) { fprintf(stderr,"error in max_c: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in max_c: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                r = (int)cblas_icamax(R,&X[2*c*R],1);
                Y[c] = sqrtf(X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1]);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                r = (int)cblas_icamax(R,&X[2*c],C);
                Y[c] = sqrtf(X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1]);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                c = (int)cblas_icamax(C,&X[2*r],R);
                Y[r] = sqrtf(X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1]);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                c = (int)cblas_icamax(C,&X[2*r*C],1);
                Y[r] = sqrtf(X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1]);
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

    //Checks
    if (R<1) { fprintf(stderr,"error in max_z: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in max_z: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                r = (int)cblas_izamax(R,&X[2*c*R],1);
                Y[c] = sqrt(X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1]);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                r = (int)cblas_izamax(R,&X[2*c],C);
                Y[c] = sqrt(X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1]);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                c = (int)cblas_izamax(C,&X[2*r],R);
                Y[r] = sqrt(X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1]);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                c = (int)cblas_izamax(C,&X[2*r*C],1);
                Y[r] = sqrt(X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1]);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in max_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int max_inplace_s (float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    float mx;
    int r, c;
    //struct timespec tic, toc;

    //Checks
    if (R<1) { fprintf(stderr,"error in max_inplace_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in max_inplace_s: C (ncols X) must be positive\n"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                r = (int)cblas_isamax(R,&X[c*R],1);
                if (X[c*R+r]<0.0f)
                {
                    mx = X[c*R+r];
                    for (r=0; r<R; r++) { X[c*R+r] -= mx; }
                    r = (int)cblas_isamax(R,&X[c*R],1);
                    X[c] = X[c*R+r] + mx;
                }
                else { X[c] = X[c*R+r]; }
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mx = X[c];
                for (r=1; r<R; r++) { if (X[r*C+c]>mx) { mx = X[r*C+c]; } }
                X[c] = mx;
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
                for (c=1; c<C; c++) { if (X[c*R+r]>mx) { mx = X[c*R+r]; } }
                X[r] = mx;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                c = (int)cblas_isamax(C,&X[r*C],1);
                if (X[r*C+c]<0.0f)
                {
                    mx = X[r*C+c];
                    for (c=0; c<C; c++) { X[r*C+c] -= mx; }
                    c = (int)cblas_isamax(C,&X[r*C],1);
                    X[r] = X[r*C+c] + mx;
                }
                else { X[r] = X[r*C+c]; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in max_inplace_s: dim must be 0 or 1.\n"); return 1;
    }
    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);

    return 0;
}


int max_inplace_d (double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    double mx;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in max_inplace_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in max_inplace_d: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                r = (int)cblas_idamax(R,&X[c*R],1);
                if (X[c*R+r]<0.0)
                {
                    mx = X[c*R+r];
                    for (r=0; r<R; r++) { X[c*R+r] -= mx; }
                    r = (int)cblas_idamax(R,&X[c*R],1);
                    X[c] = X[c*R+r] + mx;
                }
                else { X[c] = X[c*R+r]; }
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mx = X[c];
                for (r=1; r<R; r++) { if (X[r*C+c]>mx) { mx = X[r*C+c]; } }
                X[c] = mx;
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
                for (c=1; c<C; c++) { if (X[c*R+r]>mx) { mx = X[c*R+r]; } }
                X[r] = mx;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                c = (int)cblas_idamax(C,&X[r*C],1);
                if (X[r*C+c]<0.0)
                {
                    mx = X[r*C+c];
                    for (c=0; c<C; c++) { X[r*C+c] -= mx; }
                    c = (int)cblas_idamax(C,&X[r*C],1);
                    X[r] = X[r*C+c] + mx;
                }
                else { X[r] = X[r*C+c]; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in max_inplace_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int max_inplace_c (float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in max_inplace_c: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in max_inplace_c: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                r = (int)cblas_icamax(R,&X[2*c*R],1);
                X[c] = sqrtf(X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1]);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                r = (int)cblas_icamax(R,&X[2*c],C);
                X[c] = sqrtf(X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1]);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                c = (int)cblas_icamax(C,&X[2*r],R);
                X[r] = sqrtf(X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1]);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                c = (int)cblas_icamax(C,&X[2*r*C],1);
                X[r] = sqrtf(X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1]);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in max_inplace_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int max_inplace_z (double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in max_inplace_z: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in max_inplace_z: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                r = (int)cblas_izamax(R,&X[2*c*R],1);
                X[c] = sqrt(X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1]);
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                r = (int)cblas_izamax(R,&X[2*c],C);
                X[c] = sqrt(X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1]);
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (r=0; r<R; r++)
            {
                c = (int)cblas_izamax(C,&X[2*r],R);
                X[r] = sqrt(X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1]);
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                c = (int)cblas_izamax(C,&X[2*r*C],1);
                X[r] = sqrt(X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1]);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in max_inplace_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
