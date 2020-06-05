//Matrix multiply with weights (W) for each row or col of X according to dim.
//This is standard input method for most computational and NN neurons, but without biases.

#include <stdio.h>
#include <cblas.h>
//#include <time.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int W_s (float *Y, const float *X, const char iscolmajor, const int R, const int C, const float *W, const int N, const int dim);
int W_d (double *Y, const double *X, const char iscolmajor, const int R, const int C, const double *W, const int N, const int dim);
int W_c (float *Y, const float *X, const char iscolmajor, const int R, const int C, const float *W, const int N, const int dim);
int W_z (double *Y, const double *X, const char iscolmajor, const int R, const int C, const double *W, const int N, const int dim);


int W_s (float *Y, const float *X, const char iscolmajor, const int R, const int C, const float *W, const int N, const int dim)
{
    const float z = 0.0f, o = 1.0f;
    int n;
    //struct timespec tic, toc;

    //Checks
    if (R<1) { fprintf(stderr,"error in W_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in W_s: C (ncols X) must be positive\n"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    cblas_scopy(C,&B[n],0,&Y[n],N);
    if (dim==0)
    {
        cblas_scopy(N*C,z,0,Y,1);
        if (iscolmajor)
        {
            if (N==1) { cblas_sgemv(CblasColMajor,CblasTrans,R,C,o,X,R,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_sgemv(CblasColMajor,CblasTrans,R,C,o,X,R,&W[n],N,o,&Y[n],N); }
            }
            else { cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,N,X,R,o,Y,N); }
        }
        else
        {
            if (N==1) { cblas_sgemv(CblasRowMajor,CblasTrans,R,C,o,X,C,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_sgemv(CblasRowMajor,CblasTrans,R,C,o,X,C,&W[n*R],1,o,&Y[n*C],1); }
            }
            else { cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,R,X,C,o,Y,C); }
        }
    }
    else if (dim==1)
    {
        cblas_scopy(N*R,z,0,Y,1);
        if (iscolmajor)
        {
            if (N==1) { cblas_sgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_sgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,&W[n*C],1,o,&Y[n*R],1); }
            }
            else { cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,R,W,C,o,Y,R); }
        }
        else
        {
            if (N==1) { cblas_sgemv(CblasRowMajor,CblasNoTrans,R,C,o,X,C,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_sgemv(CblasRowMajor,CblasNoTrans,R,C,o,X,C,&W[n],N,o,&Y[n],N); }
            }
            else { cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,C,W,N,o,Y,N); }
        }
    }
    else
    {
        fprintf(stderr,"error in W_s: dim must be 0 or 1.\n"); return 1;
    }
    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);

    return 0;
}


int W_d (double *Y, const double *X, const char iscolmajor, const int R, const int C, const double *W, const int N, const int dim)
{
    const double z = 0.0, o = 1.0;
    int n;

    //Checks
    if (R<1) { fprintf(stderr,"error in W_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in W_d: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        cblas_dcopy(N*C,z,0,Y,1);
        if (iscolmajor)
        {
            if (N==1) { cblas_dgemv(CblasColMajor,CblasTrans,R,C,o,X,R,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_dgemv(CblasColMajor,CblasTrans,R,C,o,X,R,&W[n],N,o,&Y[n],N); }
            }
            else { cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,N,X,R,o,Y,N); }
        }
        else
        {
            if (N==1) { cblas_dgemv(CblasRowMajor,CblasTrans,R,C,o,X,C,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_dgemv(CblasRowMajor,CblasTrans,R,C,o,X,C,&W[n*R],1,o,&Y[n*C],1); }
            }
            else { cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,R,X,C,o,Y,C); }
        }
    }
    else if (dim==1)
    {
        cblas_dcopy(N*R,z,0,Y,1);
        if (iscolmajor)
        {
            if (N==1) { cblas_dgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_dgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,&W[n*C],1,o,&Y[n*R],1); }
            }
            else { cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,R,W,C,o,Y,R); }
        }
        else
        {
            if (N==1) { cblas_dgemv(CblasRowMajor,CblasNoTrans,R,C,o,X,C,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_dgemv(CblasRowMajor,CblasNoTrans,R,C,o,X,C,&W[n],N,o,&Y[n],N); }
            }
            else { cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,C,W,N,o,Y,N); }
        }
    }
    else
    {
        fprintf(stderr,"error in W_d: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


int W_c (float *Y, const float *X, const char iscolmajor, const int R, const int C, const float *W, const int N, const int dim)
{
    const float z[2] = {0.0f,0.0f}, o[2] =  {1.0f,0.0f};
    int n;

    //Checks
    if (R<1) { fprintf(stderr,"error in W_c: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in W_c: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        cblas_ccopy(N*C,z,0,Y,1);
        if (iscolmajor)
        {
            if (N==1) { cblas_cgemv(CblasColMajor,CblasTrans,R,C,o,X,R,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_cgemv(CblasColMajor,CblasTrans,R,C,o,X,R,&W[2*n],N,o,&Y[2*n],N); }
            }
            else { cblas_cgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,N,X,R,o,Y,N); }
        }
        else
        {
            if (N==1) { cblas_cgemv(CblasRowMajor,CblasTrans,R,C,o,X,C,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_cgemv(CblasRowMajor,CblasTrans,R,C,o,X,C,&W[2*n*R],1,o,&Y[2*n*C],1); }
            }
            else { cblas_cgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,R,X,C,o,Y,C); }
        }
    }
    else if (dim==1)
    {
        cblas_ccopy(N*R,z,0,Y,1);
        if (iscolmajor)
        {
            if (N==1) { cblas_cgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_cgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,&W[2*n*C],1,o,&Y[2*n*R],1); }
            }
            else { cblas_cgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,R,W,C,o,Y,R); }
        }
        else
        {
            if (N==1) { cblas_cgemv(CblasRowMajor,CblasNoTrans,R,C,o,X,C,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_cgemv(CblasRowMajor,CblasNoTrans,R,C,o,X,C,&W[2*n],N,o,&Y[2*n],N); }
            }
            else { cblas_cgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,C,W,N,o,Y,N); }
        }
    }
    else
    {
        fprintf(stderr,"error in W_c: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


int W_z (double *Y, const double *X, const char iscolmajor, const int R, const int C, const double *W, const int N, const int dim)
{
    const double z[2] = {0.0,0.0}, o[2] =  {1.0,0.0};
    int n;

    //Checks
    if (R<1) { fprintf(stderr,"error in W_z: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in W_z: C (ncols X) must be positive\n"); return 1; }

    if (dim==0)
    {
        cblas_zcopy(N*C,z,0,Y,1);
        if (iscolmajor)
        {
            if (N==1) { cblas_zgemv(CblasColMajor,CblasTrans,R,C,o,X,R,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_zgemv(CblasColMajor,CblasTrans,R,C,o,X,R,&W[2*n],N,o,&Y[2*n],N); }
            }
            else { cblas_zgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,N,X,R,o,Y,N); }
        }
        else
        {
            if (N==1) { cblas_zgemv(CblasRowMajor,CblasTrans,R,C,o,X,C,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_zgemv(CblasRowMajor,CblasTrans,R,C,o,X,C,&W[2*n*R],1,o,&Y[2*n*C],1); }
            }
            else { cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,R,X,C,o,Y,C); }
        }
    }
    else if (dim==1)
    {
        cblas_zcopy(N*R,z,0,Y,1);
        if (iscolmajor)
        {
            if (N==1) { cblas_zgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_zgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,&W[2*n*C],1,o,&Y[2*n*R],1); }
            }
            else { cblas_zgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,R,W,C,o,Y,R); }
        }
        else
        {
            if (N==1) { cblas_zgemv(CblasRowMajor,CblasNoTrans,R,C,o,X,C,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_zgemv(CblasRowMajor,CblasNoTrans,R,C,o,X,C,&W[2*n],N,o,&Y[2*n],N); }
            }
            else { cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,C,W,N,o,Y,N); }
        }
    }
    else
    {
        fprintf(stderr,"error in W_z: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
