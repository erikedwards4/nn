//Gets minimum of values for each row or col of X according to dim.
//For complex case, gets min absolute value (output is real).

//The inplace versions replace the first R or C elements of X,
//without regard to iscolmajor since output is a vector.

//Unfortunately, compiler doesn't find cblas_i?amin (but finds cblas_i?amax),
//so I do an inefficient solution for complex-valued case for now.


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
//#include <time.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int min_s (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int min_d (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor);
int min_c (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor);
int min_z (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor);

int min_inplace_s (float *X, const int R, const int C, const int dim, const char iscolmajor);
int min_inplace_d (double *X, const int R, const int C, const int dim, const char iscolmajor);
int min_inplace_c (float *X, const int R, const int C, const int dim, const char iscolmajor);
int min_inplace_z (double *X, const int R, const int C, const int dim, const char iscolmajor);


int min_s (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    int r, c;
    float mn;
    //struct timespec tic, toc;

    //Checks
    if (R<1) { fprintf(stderr,"error in min_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in min_s: C (ncols X) must be positive\n"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                mn = X[c*R];
                for (r=1; r<R; r++) { if (X[c*R+r]<mn) { mn = X[c*R+r]; } }
                Y[c] = mn;
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mn = X[c];
                for (r=1; r<R; r++) { if (X[r*C+c]<mn) { mn = X[r*C+c]; } }
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
                for (c=1; c<C; c++) { if (X[c*R+r]<mn) { mn = X[c*R+r]; } }
                Y[r] = mn;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                mn = X[r*C];
                for (c=1; c<C; c++) { if (X[r*C+c]<mn) { mn = X[r*C+c]; } }
                Y[r] = mn;
            }
        }
    }
    else
    {
        fprintf(stderr,"error in min_s: dim must be 0 or 1.\n"); return 1;
    }
    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);

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
                for (r=1; r<R; r++) { if (X[c*R+r]<mn) { mn = X[c*R+r]; } }
                Y[c] = mn;
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mn = X[c];
                for (r=1; r<R; r++) { if (X[r*C+c]<mn) { mn = X[r*C+c]; } }
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
                for (c=1; c<C; c++) { if (X[c*R+r]<mn) { mn = X[c*R+r]; } }
                Y[r] = mn;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                mn = X[r*C];
                for (c=1; c<C; c++) { if (X[r*C+c]<mn) { mn = X[r*C+c]; } }
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


// int min_c (float *Y, const float *X, const int R, const int C, const int dim, const char iscolmajor)
// {
//     int r, c;

//     //Checks
//     if (R<1) { fprintf(stderr,"error in min_c: R (nrows X) must be positive\n"); return 1; }
//     if (C<1) { fprintf(stderr,"error in min_c: C (ncols X) must be positive\n"); return 1; }

//     if (dim==0)
//     {
//         if (iscolmajor)
//         {
//             for (c=0; c<C; c++)
//             {
//                 r = (int)cblas_icamin(R,&X[2*c*R],1);
//                 Y[c] = sqrtf(X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1]);
//             }
//         }
//         else
//         {
//             for (c=0; c<C; c++)
//             {
//                 r = (int)cblas_icamin(R,&X[2*c],C);
//                 Y[c] = sqrtf(X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1]);
//             }
//         }
//     }
//     else if (dim==1)
//     {
//         if (iscolmajor)
//         {
//             for (r=0; r<R; r++)
//             {
//                 c = (int)cblas_icamin(C,&X[2*r],R);
//                 Y[r] = sqrtf(X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1]);
//             }
//         }
//         else
//         {
//             for (r=0; r<R; r++)
//             {
//                 c = (int)cblas_icamin(C,&X[2*r*C],1);
//                 Y[r] = sqrtf(X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1]);
//             }
//         }
//     }
//     else
//     {
//         fprintf(stderr,"error in min_c: dim must be 0 or 1.\n"); return 1;
//     }

//     return 0;
// }


// int min_z (double *Y, const double *X, const int R, const int C, const int dim, const char iscolmajor)
// {
//     int r, c;

//     //Checks
//     if (R<1) { fprintf(stderr,"error in min_z: R (nrows X) must be positive\n"); return 1; }
//     if (C<1) { fprintf(stderr,"error in min_z: C (ncols X) must be positive\n"); return 1; }

//     if (dim==0)
//     {
//         if (iscolmajor)
//         {
//             for (c=0; c<C; c++)
//             {
//                 r = (int)cblas_izamin(R,&X[2*c*R],1);
//                 Y[c] = sqrt(X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1]);
//             }
//         }
//         else
//         {
//             for (c=0; c<C; c++)
//             {
//                 r = (int)cblas_izamin(R,&X[2*c],C);
//                 Y[c] = sqrt(X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1]);
//             }
//         }
//     }
//     else if (dim==1)
//     {
//         if (iscolmajor)
//         {
//             for (r=0; r<R; r++)
//             {
//                 c = (int)cblas_izamin(C,&X[2*r],R);
//                 Y[r] = sqrt(X[2*(c*R+r)]*X[2*(c*R+r)] + X[2*(c*R+r)+1]*X[2*(c*R+r)+1]);
//             }
//         }
//         else
//         {
//             for (r=0; r<R; r++)
//             {
//                 c = (int)cblas_izamin(C,&X[2*r*C],1);
//                 Y[r] = sqrt(X[2*(r*C+c)]*X[2*(r*C+c)] + X[2*(r*C+c)+1]*X[2*(r*C+c)+1]);
//             }
//         }
//     }
//     else
//     {
//         fprintf(stderr,"error in min_z: dim must be 0 or 1.\n"); return 1;
//     }

//     return 0;
// }


int min_inplace_s (float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    float mn;
    int r, c;
    //struct timespec tic, toc;

    //Checks
    if (R<1) { fprintf(stderr,"error in min_inplace_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in min_inplace_s: C (ncols X) must be positive\n"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                r = (int)cblas_isamax(R,&X[c*R],1);
                if (X[c*R+r]>0.0f)
                {
                    mn = X[c*R+r];
                    for (r=0; r<R; r++) { X[c*R+r] -= mn; }
                    r = (int)cblas_isamax(R,&X[c*R],1);
                    X[c] = X[c*R+r] + mn;
                }
                else { X[c] = X[c*R+r]; }
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mn = X[c];
                for (r=1; r<R; r++) { if (X[r*C+c]<mn) { mn = X[r*C+c]; } }
                X[c] = mn;
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
                for (c=1; c<C; c++) { if (X[c*R+r]<mn) { mn = X[c*R+r]; } }
                X[r] = mn;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                c = (int)cblas_isamax(C,&X[r*C],1);
                if (X[r*C+c]>0.0f)
                {
                    mn = X[r*C+c];
                    for (c=0; c<C; c++) { X[r*C+c] -= mn; }
                    c = (int)cblas_isamax(C,&X[r*C],1);
                    X[r] = X[r*C+c] + mn;
                }
                else { X[r] = X[r*C+c]; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in min_inplace_s: dim must be 0 or 1.\n"); return 1;
    }
    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);

    return 0;
}


int min_inplace_d (double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    double mn;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in min_inplace_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in min_inplace_d: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (c=0; c<C; c++)
            {
                r = (int)cblas_idamax(R,&X[c*R],1);
                if (X[c*R+r]>0.0)
                {
                    mn = X[c*R+r];
                    for (r=0; r<R; r++) { X[c*R+r] -= mn; }
                    r = (int)cblas_idamax(R,&X[c*R],1);
                    X[c] = X[c*R+r] + mn;
                }
                else { X[c] = X[c*R+r]; }
            }
        }
        else
        {
            for (c=0; c<C; c++)
            {
                mn = X[c];
                for (r=1; r<R; r++) { if (X[r*C+c]<mn) { mn = X[r*C+c]; } }
                X[c] = mn;
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
                for (c=1; c<C; c++) { if (X[c*R+r]<mn) { mn = X[c*R+r]; } }
                X[r] = mn;
            }
        }
        else
        {
            for (r=0; r<R; r++)
            {
                c = (int)cblas_idamax(C,&X[r*C],1);
                if (X[r*C+c]>0.0)
                {
                    mn = X[r*C+c];
                    for (c=0; c<C; c++) { X[r*C+c] -= mn; }
                    c = (int)cblas_idamax(C,&X[r*C],1);
                    X[r] = X[r*C+c] + mn;
                }
                else { X[r] = X[r*C+c]; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in min_inplace_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int min_inplace_c (float *X, const int R, const int C, const int dim, const char iscolmajor)
{
    float mn, a2;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in min_inplace_c: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in min_inplace_c: C (ncols X) must be positive\n"); return 1; }

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
                X[c] = sqrtf(mn);
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
                X[c] = sqrtf(mn);
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
                X[r] = sqrtf(mn);
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
                X[r] = sqrtf(mn);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in min_inplace_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int min_inplace_z (double *X, const int R, const int C, const int dim, const char iscolmajor)
{
    double mn, a2;
    int r, c;

    //Checks
    if (R<1) { fprintf(stderr,"error in min_inplace_z: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in min_inplace_z: C (ncols X) must be positive\n"); return 1; }

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
                X[c] = sqrt(mn);
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
                X[c] = sqrt(mn);
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
                X[r] = sqrt(mn);
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
                X[r] = sqrt(mn);
            }
        }
    }
    else
    {
        fprintf(stderr,"error in min_inplace_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
