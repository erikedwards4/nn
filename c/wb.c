//Dot product with weights (w) plus biases (b) for each row or col of X according to dim.
//This is standard input method for most computational and NN neurons.
//cblas_sgemv was found to be 4-10x faster than for loop of cblas_sdot for >1k rows/cols.
//But cblas_sdot for loop slightly faster (a few ns) when <10 rows/cols to loop through.

#include <stdio.h>
#include <cblas.h>
#include <time.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int wb_s (float *Y, const float *X, const char iscolmajor, const int R, const int C, const float *W, const float *b, const int dim);
int wb_d (double *Y, const double *X, const char iscolmajor, const int R, const int C, const double *W, const double *b, const int dim);
int wb_c (float *Y, const float *X, const char iscolmajor, const int R, const int C, const float *W, const float *b, const int dim);
int wb_z (double *Y, const double *X, const char iscolmajor, const int R, const int C, const double *W, const double *b, const int dim);


int wb_s (float *Y, const float *X, const char iscolmajor, const int R, const int C, const float *W, const float *b, const int dim)
{
    const float o = 1.0f;
    //int r, c;
    struct timespec tic, toc;

    //Checks
    if (R<1) { fprintf(stderr,"error in wb_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in wb_s: C (ncols X) must be positive\n"); return 1; }

    clock_gettime(CLOCK_REALTIME,&tic);
    if (dim==0)
    {
        cblas_scopy(C,b,0,Y,1);
        //cblas_scopy(C,b,0,Y,1);
        if (iscolmajor)
        {
            cblas_sgemv(CblasColMajor,CblasTrans,R,C,o,X,R,W,1,o,Y,1);
            //for (c=0; c<C; c++) { Y[c] += cblas_sdot(R,&X[c*R],1,W,1); }
        }
        else
        {
            cblas_sgemv(CblasRowMajor,CblasTrans,R,C,o,X,C,W,1,o,Y,1);
            //for (c=0; c<C; c++) { Y[c] += cblas_sdot(R,&X[c],C,W,1); }
        }
    }
    else if (dim==1)
    {
        cblas_scopy(R,b,0,Y,1);
        //cblas_scopy(R,b,0,Y,1);
        if (iscolmajor)
        {
            cblas_sgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,W,1,o,Y,1);
            //for (r=0; r<R; r++) { Y[r] += cblas_sdot(C,&X[r],R,W,1); }
        }
        else
        {
            cblas_sgemv(CblasRowMajor,CblasNoTrans,R,C,o,X,C,W,1,o,Y,1);
            //for (r=0; r<R; r++) { Y[r] += cblas_sdot(C,&X[r*C],1,W,1); }
        }
    }
    else
    {
        fprintf(stderr,"error in wb_s: dim must be 0 or 1.\n"); return 1;
    }
    clock_gettime(CLOCK_REALTIME,&toc);
    fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);

    return 0;
}


int wb_d (double *Y, const double *X, const char iscolmajor, const int R, const int C, const double *W, const double *b, const int dim)
{
    const double o = 1.0;

    //Checks
    if (R<1) { fprintf(stderr,"error in wb_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in wb_d: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        cblas_dcopy(C,b,0,Y,1);
        if (iscolmajor)
        {
            cblas_dgemv(CblasColMajor,CblasTrans,R,C,o,X,R,W,1,o,Y,1);
        }
        else
        {
            cblas_dgemv(CblasRowMajor,CblasTrans,R,C,o,X,C,W,1,o,Y,1);
        }
    }
    else if (dim==1)
    {
        cblas_dcopy(R,b,0,Y,1);
        if (iscolmajor)
        {
            cblas_dgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,W,1,o,Y,1);
        }
        else
        {
            cblas_dgemv(CblasRowMajor,CblasNoTrans,R,C,o,X,C,W,1,o,Y,1);
        }
    }
    else
    {
        fprintf(stderr,"error in wb_d: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


int wb_c (float *Y, const float *X, const char iscolmajor, const int R, const int C, const float *W, const float *b, const int dim)
{
    const float o[2] =  {1.0f,0.0f};

    //Checks
    if (R<1) { fprintf(stderr,"error in wb_c: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in wb_c: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        cblas_ccopy(C,b,0,Y,1);
        if (iscolmajor)
        {
            cblas_cgemv(CblasColMajor,CblasTrans,R,C,o,X,R,W,1,o,Y,1);
        }
        else
        {
            cblas_cgemv(CblasRowMajor,CblasTrans,R,C,o,X,C,W,1,o,Y,1);
        }
    }
    else if (dim==1)
    {
        cblas_ccopy(R,b,0,Y,1);
        if (iscolmajor)
        {
            cblas_cgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,W,1,o,Y,1);
        }
        else
        {
            cblas_cgemv(CblasRowMajor,CblasNoTrans,R,C,o,X,C,W,1,o,Y,1);
        }
    }
    else
    {
        fprintf(stderr,"error in wb_c: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


int wb_z (double *Y, const double *X, const char iscolmajor, const int R, const int C, const double *W, const double *b, const int dim)
{
    const double o[2] =  {1.0,0.0};

    //Checks
    if (R<1) { fprintf(stderr,"error in wb_z: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in wb_z: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        cblas_zcopy(C,b,0,Y,1);
        if (iscolmajor)
        {
            cblas_zgemv(CblasColMajor,CblasTrans,R,C,o,X,R,W,1,o,Y,1);
        }
        else
        {
            cblas_zgemv(CblasRowMajor,CblasTrans,R,C,o,X,C,W,1,o,Y,1);
        }
    }
    else if (dim==1)
    {
        cblas_zcopy(R,b,0,Y,1);
        if (iscolmajor)
        {
            cblas_zgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,W,1,o,Y,1);
        }
        else
        {
            cblas_zgemv(CblasRowMajor,CblasNoTrans,R,C,o,X,C,W,1,o,Y,1);
        }
    }
    else
    {
        fprintf(stderr,"error in wb_z: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
