//Matrix multiply with weights (W) plus biases (B) for each row or col of X according to dim.
//This is standard input method for most computational and NN neurons.

#include <stdio.h>
#include <cblas.h>
#include <time.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int WB_s (float *Y, const float *X, const char iscolmajor, const int R, const int C, const float *W, const float *B, const int N, const int dim);
int WB_d (double *Y, const double *X, const char iscolmajor, const int R, const int C, const double *W, const double *B, const int N, const int dim);
int WB_c (float *Y, const float *X, const char iscolmajor, const int R, const int C, const float *W, const float *B, const int N, const int dim);
int WB_z (double *Y, const double *X, const char iscolmajor, const int R, const int C, const double *W, const double *B, const int N, const int dim);


int WB_s (float *Y, const float *X, const char iscolmajor, const int R, const int C, const float *W, const float *B, const int N, const int dim)
{
    const float o = 1.0f;
    int n;
    struct timespec tic, toc;

    //Checks
    if (R<1) { fprintf(stderr,"error in WB_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in WB_s: C (ncols X) must be positive\n"); return 1; }
    if (N<1) { fprintf(stderr,"error in WB_s: length B must be positive\n"); return 1; }

    clock_gettime(CLOCK_REALTIME,&tic);
    if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++) { cblas_scopy(C,&B[n],0,&Y[n],N); }
            cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,N,X,R,o,Y,N);
        }
        else
        {
            for (n=0; n<N; n++) { cblas_scopy(C,&B[n],0,&Y[n*R],1); }
            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,R,X,C,o,Y,N);
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++) { cblas_scopy(R,&B[n],0,&Y[n*C],1); }
            cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,R,W,C,o,Y,R);
        }
        else
        {
            for (n=0; n<N; n++) { cblas_scopy(R,&B[n],0,&Y[n],N); }
            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,C,W,N,o,Y,N);
        }
    }
    else
    {
        fprintf(stderr,"error in WB_s: dim must be 0 or 1.\n"); return 1;
    }
    clock_gettime(CLOCK_REALTIME,&toc);
    fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);

    return 0;
}

//dim==0            dim==1
//W is m x k        X is m x k
//X is k x n        W is k x n
//B is m x 1        B is 1 x n
//Y is m x n        Y is m x n
//m=N, n=C, k=R     m=R, n=N, k=C
int WB_d (double *Y, const double *X, const char iscolmajor, const int R, const int C, const double *W, const double *B, const int N, const int dim)
{
    const double o = 1.0;
    int n;

    //Checks
    if (R<1) { fprintf(stderr,"error in WB_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in WB_d: C (ncols X) must be positive\n"); return 1; }
    if (N<1) { fprintf(stderr,"error in WB_d: length B must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++) { cblas_dcopy(C,&B[n],0,&Y[n],N); }
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,N,X,R,o,Y,N);
        }
        else
        {
            for (n=0; n<N; n++) { cblas_dcopy(C,&B[n],0,&Y[n*R],1); }
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,R,X,C,o,Y,N);
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++) { cblas_dcopy(R,&B[n],0,&Y[n*C],1); }
            cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,R,W,C,o,Y,R);
        }
        else
        {
            for (n=0; n<N; n++) { cblas_dcopy(R,&B[n],0,&Y[n],N); }
            cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,C,W,N,o,Y,N);
        }
    }
    else
    {
        fprintf(stderr,"error in WB_d: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


int WB_c (float *Y, const float *X, const char iscolmajor, const int R, const int C, const float *W, const float *B, const int N, const int dim)
{
    const float o[2] =  {1.0f,0.0f};
    int n;

    //Checks
    if (R<1) { fprintf(stderr,"error in WB_c: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in WB_c: C (ncols X) must be positive\n"); return 1; }
    if (N<1) { fprintf(stderr,"error in WB_c: length B must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++) { cblas_ccopy(C,&B[n],0,&Y[n],N); }
            cblas_cgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,N,X,R,o,Y,N);
        }
        else
        {
            for (n=0; n<N; n++) { cblas_ccopy(C,&B[n],0,&Y[n*R],1); }
            cblas_cgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,R,X,C,o,Y,N);
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++) { cblas_ccopy(R,&B[n],0,&Y[n*C],1); }
            cblas_cgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,R,W,C,o,Y,R);
        }
        else
        {
            for (n=0; n<N; n++) { cblas_ccopy(R,&B[n],0,&Y[n],N); }
            cblas_cgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,C,W,N,o,Y,N);
        }
    }
    else
    {
        fprintf(stderr,"error in WB_c: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


int WB_z (double *Y, const double *X, const char iscolmajor, const int R, const int C, const double *W, const double *B, const int N, const int dim)
{
    const double o[2] =  {1.0,0.0};
    int n;

    //Checks
    if (R<1) { fprintf(stderr,"error in WB_z: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in WB_z: C (ncols X) must be positive\n"); return 1; }
    if (N<1) { fprintf(stderr,"error in WB_z: length B must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++) { cblas_zcopy(C,&B[n],0,&Y[n],N); }
            cblas_zgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,N,X,R,o,Y,N);
        }
        else
        {
            for (n=0; n<N; n++) { cblas_zcopy(C,&B[n],0,&Y[n*R],1); }
            cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,R,X,C,o,Y,N);
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++) { cblas_zcopy(R,&B[n],0,&Y[n*C],1); }
            cblas_zgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,R,W,C,o,Y,R);
        }
        else
        {
            for (n=0; n<N; n++) { cblas_zcopy(R,&B[n],0,&Y[n],N); }
            cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,C,W,N,o,Y,N);
        }
    }
    else
    {
        fprintf(stderr,"error in WB_z: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
