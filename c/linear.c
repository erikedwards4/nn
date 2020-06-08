//Matrix multiply with weights (W) plus biases (B) for each row or col of X according to dim.
//This is standard input method for most computational and NN neurons.
//Timing experiments found: for dim==0, rowmajor faster; for dim=1, colmajor faster.
//                          matrix multiply (cblas_?gemm) faster only for large N (N>~32)
//                          for the single-neuron case (N=1), this is just as fast as separate function, so just combined here
//The output Y is a matrix with: \n";
//Y = W*X + B,  for dim=0
//Y = X*W + B,  for dim=1
//

#include <stdio.h>
#include <cblas.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int linear_s (float *Y, const float *X, const float *W, const float *B, const int Ni, const int No, const int T, const int dim, const char iscolmajor);
int linear_d (double *Y, const double *X, const double *W, const double *B, const int Ni, const int No, const int T, const int dim, const char iscolmajor);
int linear_c (float *Y, const float *X, const float *W, const float *B, const int Ni, const int No, const int T, const int dim, const char iscolmajor);
int linear_z (double *Y, const double *X, const double *W, const double *B, const int Ni, const int No, const int T, const int dim, const char iscolmajor);


int linear_s (float *Y, const float *X, const float *W, const float *B, const int Ni, const int No, const int T, const int dim, const char iscolmajor)
{
    const float o = 1.0f;
    int n;

    //Checks
    if (Ni<1) { fprintf(stderr,"error in linear_s: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1) { fprintf(stderr,"error in linear_s: No (num output neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in linear_s: T (num time points) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<No; n++) { cblas_scopy(T,&B[n],0,&Y[n],No); }
            if (No==1) { cblas_sgemv(CblasColMajor,CblasTrans,Ni,T,o,X,Ni,W,1,o,Y,1); }
            else if (No<32)
            {
                for (n=0; n<No; n++) { cblas_sgemv(CblasColMajor,CblasTrans,Ni,T,o,X,Ni,&W[n],No,o,&Y[n],No); }
            }
            else { cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,No,T,Ni,o,W,No,X,Ni,o,Y,No); }
        }
        else
        {
            for (n=0; n<No; n++) { cblas_scopy(T,&B[n],0,&Y[n*T],1); }
            if (No==1) { cblas_sgemv(CblasRowMajor,CblasTrans,Ni,T,o,X,T,W,1,o,Y,1); }
            else if (No<32)
            {
                for (n=0; n<No; n++) { cblas_sgemv(CblasRowMajor,CblasTrans,Ni,T,o,X,T,&W[n*Ni],1,o,&Y[n*T],1); }
            }
            else { cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,No,T,Ni,o,W,Ni,X,T,o,Y,T); }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<No; n++) { cblas_scopy(T,&B[n],0,&Y[n*T],1); }
            if (No==1) { cblas_sgemv(CblasColMajor,CblasNoTrans,T,Ni,o,X,T,W,1,o,Y,1); }
            else if (No<32)
            {
                for (n=0; n<No; n++) { cblas_sgemv(CblasColMajor,CblasNoTrans,T,Ni,o,X,T,&W[n*Ni],1,o,&Y[n*T],1); }
            }
            else { cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,T,No,Ni,o,X,T,W,Ni,o,Y,T); }
        }
        else
        {
            for (n=0; n<No; n++) { cblas_scopy(T,&B[n],0,&Y[n],No); }
            if (No==1) { cblas_sgemv(CblasRowMajor,CblasNoTrans,T,Ni,o,X,Ni,W,1,o,Y,1); }
            else if (No<32)
            {
                for (n=0; n<No; n++) { cblas_sgemv(CblasRowMajor,CblasNoTrans,T,Ni,o,X,Ni,&W[n],No,o,&Y[n],No); }
            }
            else { cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,T,No,Ni,o,X,Ni,W,No,o,Y,No); }
        }
    }
    else
    {
        fprintf(stderr,"error in linear_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int linear_d (double *Y, const double *X, const double *W, const double *B, const int Ni, const int No, const int T, const int dim, const char iscolmajor)
{
    const double o = 1.0;
    int n;

    //Checks
    if (Ni<1) { fprintf(stderr,"error in linear_d: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1) { fprintf(stderr,"error in linear_d: No (num output neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in linear_d: T (num time points) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<No; n++) { cblas_dcopy(T,&B[n],0,&Y[n],No); }
            if (No==1) { cblas_dgemv(CblasColMajor,CblasTrans,Ni,T,o,X,Ni,W,1,o,Y,1); }
            else if (No<32)
            {
                for (n=0; n<No; n++) { cblas_dgemv(CblasColMajor,CblasTrans,Ni,T,o,X,Ni,&W[n],No,o,&Y[n],No); }
            }
            else { cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,No,T,Ni,o,W,No,X,Ni,o,Y,No); }
        }
        else
        {
            for (n=0; n<No; n++) { cblas_dcopy(T,&B[n],0,&Y[n*T],1); }
            if (No==1) { cblas_dgemv(CblasRowMajor,CblasTrans,Ni,T,o,X,T,W,1,o,Y,1); }
            else if (No<32)
            {
                for (n=0; n<No; n++) { cblas_dgemv(CblasRowMajor,CblasTrans,Ni,T,o,X,T,&W[n*Ni],1,o,&Y[n*T],1); }
            }
            else { cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,No,T,Ni,o,W,Ni,X,T,o,Y,T); }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<No; n++) { cblas_dcopy(T,&B[n],0,&Y[n*T],1); }
            if (No==1) { cblas_dgemv(CblasColMajor,CblasNoTrans,T,Ni,o,X,T,W,1,o,Y,1); }
            else if (No<32)
            {
                for (n=0; n<No; n++) { cblas_dgemv(CblasColMajor,CblasNoTrans,T,Ni,o,X,T,&W[n*Ni],1,o,&Y[n*T],1); }
            }
            else { cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,T,No,Ni,o,X,T,W,Ni,o,Y,T); }
        }
        else
        {
            for (n=0; n<No; n++) { cblas_dcopy(T,&B[n],0,&Y[n],No); }
            if (No==1) { cblas_dgemv(CblasRowMajor,CblasNoTrans,T,Ni,o,X,Ni,W,1,o,Y,1); }
            else if (No<32)
            {
                for (n=0; n<No; n++) { cblas_dgemv(CblasRowMajor,CblasNoTrans,T,Ni,o,X,Ni,&W[n],No,o,&Y[n],No); }
            }
            else { cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,T,No,Ni,o,X,Ni,W,No,o,Y,No); }
        }
    }
    else
    {
        fprintf(stderr,"error in linear_d: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


int linear_c (float *Y, const float *X, const float *W, const float *B, const int Ni, const int No, const int T, const int dim, const char iscolmajor)
{
    const float o[2] =  {1.0f,0.0f};
    int n;

    //Checks
    if (Ni<1) { fprintf(stderr,"error in linear_c: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1) { fprintf(stderr,"error in linear_c: No (num output neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in linear_c: T (num time points) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<No; n++) { cblas_ccopy(T,&B[2*n],0,&Y[2*n],No); }
            if (No==1) { cblas_cgemv(CblasColMajor,CblasTrans,Ni,T,o,X,Ni,W,1,o,Y,1); }
            else if (No<32)
            {
                for (n=0; n<No; n++) { cblas_cgemv(CblasColMajor,CblasTrans,Ni,T,o,X,Ni,&W[2*n],No,o,&Y[2*n],No); }
            }
            else { cblas_cgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,No,T,Ni,o,W,No,X,Ni,o,Y,No); }
        }
        else
        {
            for (n=0; n<No; n++) { cblas_ccopy(T,&B[2*n],0,&Y[2*n*T],1); }
            if (No==1) { cblas_cgemv(CblasRowMajor,CblasTrans,Ni,T,o,X,T,W,1,o,Y,1); }
            else if (No<32)
            {
                for (n=0; n<No; n++) { cblas_cgemv(CblasRowMajor,CblasTrans,Ni,T,o,X,T,&W[2*n*Ni],1,o,&Y[2*n*T],1); }
            }
            else { cblas_cgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,No,T,Ni,o,W,Ni,X,T,o,Y,T); }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<No; n++) { cblas_ccopy(T,&B[2*n],0,&Y[2*n*T],1); }
            if (No==1) { cblas_cgemv(CblasColMajor,CblasNoTrans,T,Ni,o,X,T,W,1,o,Y,1); }
            else if (No<32)
            {
                for (n=0; n<No; n++) { cblas_cgemv(CblasColMajor,CblasNoTrans,T,Ni,o,X,T,&W[2*n*Ni],1,o,&Y[2*n*T],1); }
            }
            else { cblas_cgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,T,No,Ni,o,X,T,W,Ni,o,Y,T); }
        }
        else
        {
            for (n=0; n<No; n++) { cblas_ccopy(T,&B[2*n],0,&Y[2*n],No); }
            if (No==1) { cblas_cgemv(CblasRowMajor,CblasNoTrans,T,Ni,o,X,Ni,W,1,o,Y,1); }
            else if (No<32)
            {
                for (n=0; n<No; n++) { cblas_cgemv(CblasRowMajor,CblasNoTrans,T,Ni,o,X,Ni,&W[2*n],No,o,&Y[2*n],No); }
            }
            else { cblas_cgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,T,No,Ni,o,X,Ni,W,No,o,Y,No); }
        }
    }
    else
    {
        fprintf(stderr,"error in linear_c: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


int linear_z (double *Y, const double *X, const double *W, const double *B, const int Ni, const int No, const int T, const int dim, const char iscolmajor)
{
    const double o[2] =  {1.0,0.0};
    int n;

    //Checks
    if (Ni<1) { fprintf(stderr,"error in linear_z: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1) { fprintf(stderr,"error in linear_z: No (num output neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in linear_z: T (num time points) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<No; n++) { cblas_zcopy(T,&B[2*n],0,&Y[2*n],No); }
            if (No==1) { cblas_zgemv(CblasColMajor,CblasTrans,Ni,T,o,X,Ni,W,1,o,Y,1); }
            else if (No<32)
            {
                for (n=0; n<No; n++) { cblas_zgemv(CblasColMajor,CblasTrans,Ni,T,o,X,Ni,&W[2*n],No,o,&Y[2*n],No); }
            }
            else { cblas_zgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,No,T,Ni,o,W,No,X,Ni,o,Y,No); }
        }
        else
        {
            for (n=0; n<No; n++) { cblas_zcopy(T,&B[2*n],0,&Y[2*n*T],1); }
            if (No==1) { cblas_zgemv(CblasRowMajor,CblasTrans,Ni,T,o,X,T,W,1,o,Y,1); }
            else if (No<32)
            {
                for (n=0; n<No; n++) { cblas_zgemv(CblasRowMajor,CblasTrans,Ni,T,o,X,T,&W[2*n*Ni],1,o,&Y[2*n*T],1); }
            }
            else { cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,No,T,Ni,o,W,Ni,X,T,o,Y,T); }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<No; n++) { cblas_zcopy(T,&B[2*n],0,&Y[2*n*T],1); }
            if (No==1) { cblas_zgemv(CblasColMajor,CblasNoTrans,T,Ni,o,X,T,W,1,o,Y,1); }
            else if (No<32)
            {
                for (n=0; n<No; n++) { cblas_zgemv(CblasColMajor,CblasNoTrans,T,Ni,o,X,T,&W[2*n*Ni],1,o,&Y[2*n*T],1); }
            }
            else { cblas_zgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,T,No,Ni,o,X,T,W,Ni,o,Y,T); }
        }
        else
        {
            for (n=0; n<No; n++) { cblas_zcopy(T,&B[2*n],0,&Y[2*n],No); }
            if (No==1) { cblas_zgemv(CblasRowMajor,CblasNoTrans,T,Ni,o,X,Ni,W,1,o,Y,1); }
            else if (No<32)
            {
                for (n=0; n<No; n++) { cblas_zgemv(CblasRowMajor,CblasNoTrans,T,Ni,o,X,Ni,&W[2*n],No,o,&Y[2*n],No); }
            }
            else { cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,T,No,Ni,o,X,Ni,W,No,o,Y,No); }
        }
    }
    else
    {
        fprintf(stderr,"error in linear_z: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
