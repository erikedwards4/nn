//CELL operation: does output side of Jordan RNN layer.
//The inputs here are from the IN stage, so reduced to N driving input time-series in X.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int jordan_s (float *Y, const float *X, const float *U, float *Y1, const float *W, const float *B, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const float o = 1.0f;

    float *H;
    if (!(H=(float *)malloc(N*sizeof(float)))) { fprintf(stderr,"error in jordan_s: problem with malloc. "); perror("malloc"); return 1; }

    if (N==1u)
    {
        H[0] = 1.0f / (1.0f+expf(-X[0]-U[0]*Y1[0]));
        Y[0] = 1.0f / (1.0f+expf(-B[0]-W[0]*H[0]));
        for (size_t t=1; t<T; ++t)
        {
            H[0] = 1.0f / (1.0f+expf(-X[t]-U[0]*Y[t-1]));
            Y[t] = 1.0f / (1.0f+expf(-B[0]-W[0]*H[0]));
        }
    }
    else
    {
        if (dim==0u)
        {
            if (iscolmajor)
            {
                for (size_t n=0u; n<N; ++n) { cblas_scopy((int)T,&B[n],0,&Y[n],(int)N); }
                cblas_scopy((int)N,&X[0],1,H,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,Y1,1,o,H,1);
                for (size_t n=0u; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-H[n])); }
                cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[0],1);
                for (size_t n=0u; n<N; ++n) { Y[n] = 1.0f/(1.0f+expf(-Y[n])); }
                for (size_t t=1; t<T; ++t)
                {
                    cblas_scopy((int)N,&X[t*N],1,H,1);
                    cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,&Y[(t-1)*N],1,o,H,1);
                    for (size_t n=0u; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-H[n])); }
                    cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[t*N],1);
                    for (size_t n=0u; n<N; ++n) { Y[t*N+n] = 1.0f/(1.0f+expf(-Y[t*N+n])); }
                }
            }
            else
            {
                for (size_t n=0u; n<N; ++n) { cblas_scopy((int)T,&B[n],0,&Y[n*T],1); }
                cblas_scopy((int)N,&X[0],(int)T,H,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,Y1,1,o,H,1);
                for (size_t n=0u; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-H[n])); }
                cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[0],(int)T);
                for (size_t n=0u; n<N; ++n) { Y[n*T] = 1.0f/(1.0f+expf(-Y[n*T])); }
                for (size_t t=1; t<T; ++t)
                {
                    cblas_scopy((int)N,&X[t],(int)T,H,1);
                    cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,&Y[t-1],(int)T,o,H,1);
                    for (size_t n=0u; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-H[n])); }
                    cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[t],(int)T);
                    for (size_t n=0u; n<N; ++n) { Y[t+n*T] = 1.0f/(1.0f+expf(-Y[t+n*T])); }
                }
            }
        }
        else if (dim==1u)
        {
            if (iscolmajor)
            {
                for (size_t n=0u; n<N; ++n) { cblas_scopy((int)T,&B[n],0,&Y[n*T],1); }
                cblas_scopy((int)N,&X[0],(int)T,H,1);
                cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,Y1,1,o,H,1);
                for (size_t n=0u; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-H[n])); }
                cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[0],(int)T);
                for (size_t n=0u; n<N; ++n) { Y[n*T] = 1.0f/(1.0f+expf(-Y[n*T])); }
                for (size_t t=1; t<T; ++t)
                {
                    cblas_scopy((int)N,&X[t],(int)T,H,1);
                    cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,&Y[t-1],(int)T,o,H,1);
                    for (size_t n=0u; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-H[n])); }
                    cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[t],(int)T);
                    for (size_t n=0u; n<N; ++n) { Y[t+n*T] = 1.0f/(1.0f+expf(-Y[t+n*T])); }
                }
            }
            else
            {
                for (size_t n=0u; n<N; ++n) { cblas_scopy((int)T,&B[n],0,&Y[n],(int)N); }
                cblas_scopy((int)N,&X[0],1,H,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,Y1,1,o,H,1);
                for (size_t n=0u; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-H[n])); }
                cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[0],1);
                for (size_t n=0u; n<N; ++n) { Y[n] = 1.0f/(1.0f+expf(-Y[n])); }
                for (size_t t=1; t<T; ++t)
                {
                    cblas_scopy((int)N,&X[t*N],1,H,1);
                    cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,&Y[(t-1)*N],1,o,H,1);
                    for (size_t n=0u; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-H[n])); }
                    cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[t*N],1);
                    for (size_t n=0u; n<N; ++n) { Y[t*N+n] = 1.0f/(1.0f+expf(-Y[t*N+n])); }
                }
            }
        }
        else
        {
            fprintf(stderr,"error in jordan_s: dim must be 0 or 1.\n"); return 1;
        }
    }

    return 0;
}


int jordan_d (double *Y, const double *X, const double *U, double *Y1, const double *W, const double *B, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const double o = 1.0;
    
    double *H;
    if (!(H=(double *)malloc(N*sizeof(double)))) { fprintf(stderr,"error in jordan_d: problem with malloc. "); perror("malloc"); return 1; }

    if (N==1u)
    {
        H[0] = 1.0 / (1.0+exp(-X[0]-U[0]*Y1[0]));
        Y[0] = 1.0 / (1.0+exp(-B[0]-W[0]*H[0]));
        for (size_t t=1; t<T; ++t)
        {
            H[0] = 1.0 / (1.0+exp(-X[t]-U[0]*Y[t-1]));
            Y[t] = 1.0 / (1.0+exp(-B[0]-W[0]*H[0]));
        }
    }
    else
    {
        if (dim==0u)
        {
            if (iscolmajor)
            {
                for (size_t n=0u; n<N; ++n) { cblas_dcopy((int)T,&B[n],0,&Y[n],(int)N); }
                cblas_dcopy((int)N,&X[0],1,H,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,Y1,1,o,H,1);
                for (size_t n=0u; n<N; ++n) { H[n] = 1.0/(1.0+exp(-H[n])); }
                cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[0],1);
                for (size_t n=0u; n<N; ++n) { Y[n] = 1.0/(1.0+exp(-Y[n])); }
                for (size_t t=1; t<T; ++t)
                {
                    cblas_dcopy((int)N,&X[t*N],1,H,1);
                    cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,&Y[(t-1)*N],1,o,H,1);
                    for (size_t n=0u; n<N; ++n) { H[n] = 1.0/(1.0+exp(-H[n])); }
                    cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[t*N],1);
                    for (size_t n=0u; n<N; ++n) { Y[t*N+n] = 1.0/(1.0+exp(-Y[t*N+n])); }
                }
            }
            else
            {
                for (size_t n=0u; n<N; ++n) { cblas_dcopy((int)T,&B[n],0,&Y[n*T],1); }
                cblas_dcopy((int)N,&X[0],(int)T,H,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,Y1,1,o,H,1);
                for (size_t n=0u; n<N; ++n) { H[n] = 1.0/(1.0+exp(-H[n])); }
                cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[0],(int)T);
                for (size_t n=0u; n<N; ++n) { Y[n*T] = 1.0/(1.0+exp(-Y[n*T])); }
                for (size_t t=1; t<T; ++t)
                {
                    cblas_dcopy((int)N,&X[t],(int)T,H,1);
                    cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,&Y[t-1],(int)T,o,H,1);
                    for (size_t n=0u; n<N; ++n) { H[n] = 1.0/(1.0+exp(-H[n])); }
                    cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[t],(int)T);
                    for (size_t n=0u; n<N; ++n) { Y[t+n*T] = 1.0/(1.0+exp(-Y[t+n*T])); }
                }
            }
        }
        else if (dim==1u)
        {
            if (iscolmajor)
            {
                for (size_t n=0u; n<N; ++n) { cblas_dcopy((int)T,&B[n],0,&Y[n*T],1); }
                cblas_dcopy((int)N,&X[0],(int)T,H,1);
                cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,Y1,1,o,H,1);
                for (size_t n=0u; n<N; ++n) { H[n] = 1.0/(1.0+exp(-H[n])); }
                cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[0],(int)T);
                for (size_t n=0u; n<N; ++n) { Y[n*T] = 1.0/(1.0+exp(-Y[n*T])); }
                for (size_t t=1; t<T; ++t)
                {
                    cblas_dcopy((int)N,&X[t],(int)T,H,1);
                    cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,&Y[t-1],(int)T,o,H,1);
                    for (size_t n=0u; n<N; ++n) { H[n] = 1.0/(1.0+exp(-H[n])); }
                    cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[t],(int)T);
                    for (size_t n=0u; n<N; ++n) { Y[t+n*T] = 1.0/(1.0+exp(-Y[t+n*T])); }
                }
            }
            else
            {
                for (size_t n=0u; n<N; ++n) { cblas_dcopy((int)T,&B[n],0,&Y[n],(int)N); }
                cblas_dcopy((int)N,&X[0],1,H,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,Y1,1,o,H,1);
                for (size_t n=0u; n<N; ++n) { H[n] = 1.0/(1.0+exp(-H[n])); }
                cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[0],1);
                for (size_t n=0u; n<N; ++n) { Y[n] = 1.0/(1.0+exp(-Y[n])); }
                for (size_t t=1; t<T; ++t)
                {
                    cblas_dcopy((int)N,&X[t*N],1,H,1);
                    cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,&Y[(t-1)*N],1,o,H,1);
                    for (size_t n=0u; n<N; ++n) { H[n] = 1.0/(1.0+exp(-H[n])); }
                    cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[t*N],1);
                    for (size_t n=0u; n<N; ++n) { Y[t*N+n] = 1.0/(1.0+exp(-Y[t*N+n])); }
                }
            }
        }
        else
        {
            fprintf(stderr,"error in jordan_d: dim must be 0 or 1.\n"); return 1;
        }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
