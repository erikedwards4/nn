//Matrix multiply with weights (W) plus biases (B) for each row or col of X according to dim.
//This is standard input method for most computational and NN neurons.
//Timing experiments found: for dim==0, rowmajor faster; for dim=1, colmajor faster.
//                          matrix multiply (cblas_?gemm) faster only for large N (N>~32)
//                          for the single-neuron case (N=1), this is just as fast as separate function (wb_?), so just combine here

#include <stdio.h>
#include <cblas.h>
//#include <time.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int wx_b_s (float *Y, const float *X, const char iscolmajor, const int R, const int C, const float *W, const float *B, const int N, const int dim);
int wx_b_d (double *Y, const double *X, const char iscolmajor, const int R, const int C, const double *W, const double *B, const int N, const int dim);
int wx_b_c (float *Y, const float *X, const char iscolmajor, const int R, const int C, const float *W, const float *B, const int N, const int dim);
int wx_b_z (double *Y, const double *X, const char iscolmajor, const int R, const int C, const double *W, const double *B, const int N, const int dim);


int wx_b_s (float *Y, const float *X, const char iscolmajor, const int R, const int C, const float *W, const float *B, const int N, const int dim)
{
    const float o = 1.0f;
    int n;
    //struct timespec tic, toc;

    //Checks
    if (R<1) { fprintf(stderr,"error in wx_b_s: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in wx_b_s: C (ncols X) must be positive\n"); return 1; }
    if (N<1) { fprintf(stderr,"error in wx_b_s: N (length B) must be positive\n"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++) { cblas_scopy(C,&B[n],0,&Y[n],N); }
            if (N==1) { cblas_sgemv(CblasColMajor,CblasTrans,R,C,o,X,R,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_sgemv(CblasColMajor,CblasTrans,R,C,o,X,R,&W[n],N,o,&Y[n],N); }
            }
            else { cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,N,X,R,o,Y,N); }
        }
        else
        {
            for (n=0; n<N; n++) { cblas_scopy(C,&B[n],0,&Y[n*C],1); }
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
        if (iscolmajor)
        {
            for (n=0; n<N; n++) { cblas_scopy(R,&B[n],0,&Y[n*R],1); }
            if (N==1) { cblas_sgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_sgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,&W[n*C],1,o,&Y[n*R],1); }
            }
            else { cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,R,W,C,o,Y,R); }
        }
        else
        {
            for (n=0; n<N; n++) { cblas_scopy(R,&B[n],0,&Y[n],N); }
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
        fprintf(stderr,"error in wx_b_s: dim must be 0 or 1.\n"); return 1;
    }
    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);

    return 0;
}


int wx_b_d (double *Y, const double *X, const char iscolmajor, const int R, const int C, const double *W, const double *B, const int N, const int dim)
{
    const double o = 1.0;
    int n;

    //Checks
    if (R<1) { fprintf(stderr,"error in wx_b_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in wx_b_d: C (ncols X) must be positive\n"); return 1; }
    if (N<1) { fprintf(stderr,"error in wx_b_d: N (length B) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++) { cblas_dcopy(C,&B[n],0,&Y[n],N); }
            if (N==1) { cblas_dgemv(CblasColMajor,CblasTrans,R,C,o,X,R,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_dgemv(CblasColMajor,CblasTrans,R,C,o,X,R,&W[n],N,o,&Y[n],N); }
            }
            else { cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,N,X,R,o,Y,N); }
        }
        else
        {
            for (n=0; n<N; n++) { cblas_dcopy(C,&B[n],0,&Y[n*C],1); }
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
        if (iscolmajor)
        {
            for (n=0; n<N; n++) { cblas_dcopy(R,&B[n],0,&Y[n*R],1); }
            if (N==1) { cblas_dgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_dgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,&W[n*C],1,o,&Y[n*R],1); }
            }
            else { cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,R,W,C,o,Y,R); }
        }
        else
        {
            for (n=0; n<N; n++) { cblas_dcopy(R,&B[n],0,&Y[n],N); }
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
        fprintf(stderr,"error in wx_b_d: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


int wx_b_c (float *Y, const float *X, const char iscolmajor, const int R, const int C, const float *W, const float *B, const int N, const int dim)
{
    const float o[2] =  {1.0f,0.0f};
    int n;

    //Checks
    if (R<1) { fprintf(stderr,"error in wx_b_c: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in wx_b_c: C (ncols X) must be positive\n"); return 1; }
    if (N<1) { fprintf(stderr,"error in wx_b_c: N (length B) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++) { cblas_ccopy(C,&B[2*n],0,&Y[2*n],N); }
            if (N==1) { cblas_cgemv(CblasColMajor,CblasTrans,R,C,o,X,R,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_cgemv(CblasColMajor,CblasTrans,R,C,o,X,R,&W[2*n],N,o,&Y[2*n],N); }
            }
            else { cblas_cgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,N,X,R,o,Y,N); }
        }
        else
        {
            for (n=0; n<N; n++) { cblas_ccopy(C,&B[2*n],0,&Y[2*n*C],1); }
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
        if (iscolmajor)
        {
            for (n=0; n<N; n++) { cblas_ccopy(R,&B[2*n],0,&Y[2*n*R],1); }
            if (N==1) { cblas_cgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_cgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,&W[2*n*C],1,o,&Y[2*n*R],1); }
            }
            else { cblas_cgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,R,W,C,o,Y,R); }
        }
        else
        {
            for (n=0; n<N; n++) { cblas_ccopy(R,&B[2*n],0,&Y[2*n],N); }
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
        fprintf(stderr,"error in wx_b_c: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


int wx_b_z (double *Y, const double *X, const char iscolmajor, const int R, const int C, const double *W, const double *B, const int N, const int dim)
{
    const double o[2] =  {1.0,0.0};
    int n;

    //Checks
    if (R<1) { fprintf(stderr,"error in wx_b_z: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in wx_b_z: C (ncols X) must be positive\n"); return 1; }
    if (N<1) { fprintf(stderr,"error in wx_b_z: N (length B) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++) { cblas_zcopy(C,&B[2*n],0,&Y[2*n],N); }
            if (N==1) { cblas_zgemv(CblasColMajor,CblasTrans,R,C,o,X,R,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_zgemv(CblasColMajor,CblasTrans,R,C,o,X,R,&W[2*n],N,o,&Y[2*n],N); }
            }
            else { cblas_zgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,C,R,o,W,N,X,R,o,Y,N); }
        }
        else
        {
            for (n=0; n<N; n++) { cblas_zcopy(C,&B[2*n],0,&Y[2*n*C],1); }
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
        if (iscolmajor)
        {
            for (n=0; n<N; n++) { cblas_zcopy(R,&B[2*n],0,&Y[2*n*R],1); }
            if (N==1) { cblas_zgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,W,1,o,Y,1); }
            else if (N<32)
            {
                for (n=0; n<N; n++) { cblas_zgemv(CblasColMajor,CblasNoTrans,R,C,o,X,R,&W[2*n*C],1,o,&Y[2*n*R],1); }
            }
            else { cblas_zgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,R,N,C,o,X,R,W,C,o,Y,R); }
        }
        else
        {
            for (n=0; n<N; n++) { cblas_zcopy(R,&B[2*n],0,&Y[2*n],N); }
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
        fprintf(stderr,"error in wx_b_z: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
