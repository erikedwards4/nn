//CELL operation: does output side of Elman RNN layer.
//There is no strict definition of Elman RNN in the literature.
//For example, the Wikipedia article doesn't specify the dims of the hidden or other vectors.
//Here I implement an interpretation where there are N neurons and thus N inputs/outputs.
//The inputs here are from the IN stage, so reduced to N driving input time-series in X.

//Again, this is not an "Elman network", rather a layer of N neurons
//that I have named "Elman" neurons due to their great similarity to an Elman RNN.

//To do: should I allow other output activations other than logistic?

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int elman_s (float *Y, const float *X, const float *U, float *H, const float *W, const float *B, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const float o = 1.0f;

    if (N==1u)
    {
        for (size_t t=0u; t<T; ++t)
        {
            H[0] = 1.0f / (1.0f+expf(-X[t]-U[0]*H[0]));
            Y[t] = 1.0f / (1.0f+expf(-B[0]-W[0]*H[0]));
        }
    }
    else
    {
        float *tmp;
        if (!(tmp=(float *)malloc(N*sizeof(float)))) { fprintf(stderr,"error in elman_s: problem with malloc. "); perror("malloc"); return 1; }

        if (dim==0u)
        {
            if (iscolmajor)
            {
                for (size_t t=0u; t<T; ++t)
                {
                    cblas_scopy((int)N,&X[t*N],1,tmp,1);
                    cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,tmp,1);
                    for (size_t n=0u; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-tmp[n])); }
                    //cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[t*N],1);
                    //for (size_t n=0u; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-X[t*N+n])); }
                    //for (size_t n=0u; n<N; ++n) { Y[t*N+n] = 1.0f/(1.0f+expf(-cblas_sdot((int)N,&W[n],(int)N,H,1)-B[n])); }
                    cblas_scopy((int)N,B,1,tmp,1);
                    cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,tmp,1);
                    for (size_t n=0u; n<N; ++n) { Y[t*N+n] = 1.0f/(1.0f+expf(-tmp[n])); }
                }
            }
            else
            {
                for (size_t t=0u; t<T; ++t)
                {
                    cblas_scopy((int)N,&X[t],(int)T,tmp,1);
                    cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,tmp,1);
                    for (size_t n=0u; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-tmp[n])); }
                    cblas_scopy((int)N,B,1,tmp,1);
                    cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,tmp,1);
                    for (size_t n=0u; n<N; ++n) { Y[t+n*T] = 1.0f/(1.0f+expf(-tmp[n])); }
                }
            }
        }
        else if (dim==1u)
        {
            if (iscolmajor)
            {
                for (size_t t=0u; t<T; ++t)
                {
                    cblas_scopy((int)N,&X[t],(int)T,tmp,1);
                    cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,tmp,1);
                    for (size_t n=0u; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-tmp[n])); }
                    cblas_scopy((int)N,B,1,tmp,1);
                    cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,tmp,1);
                    for (size_t n=0u; n<N; ++n) { Y[t+n*T] = 1.0f/(1.0f+expf(-tmp[n])); }
                }
            }
            else
            {
                for (size_t t=0u; t<T; ++t)
                {
                    cblas_scopy((int)N,&X[t*N],1,tmp,1);
                    cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,tmp,1);
                    for (size_t n=0u; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-tmp[n])); }
                    cblas_scopy((int)N,B,1,tmp,1);
                    cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,tmp,1);
                    for (size_t n=0u; n<N; ++n) { Y[t*N+n] = 1.0f/(1.0f+expf(-tmp[n])); }
                }
            }
        }
        else
        {
            fprintf(stderr,"error in elman_s: dim must be 0 or 1.\n"); return 1;
        }
    }

    return 0;
}


int elman_d (double *Y, const double *X, const double *U, double *H, const double *W, const double *B, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const double o = 1.0;

    if (N==1u)
    {
        for (size_t t=0u; t<T; ++t)
        {
            H[0] = 1.0 / (1.0+exp(-X[t]-U[0]*H[0]));
            Y[t] = 1.0 / (1.0+exp(-B[0]-W[0]*H[0]));
        }
    }
    else
    {
        double *tmp;
        if (!(tmp=(double *)malloc(N*sizeof(double)))) { fprintf(stderr,"error in elman_d: problem with malloc. "); perror("malloc"); return 1; }

        if (dim==0u)
        {
            if (iscolmajor)
            {
                for (size_t t=0u; t<T; ++t)
                {
                    cblas_dcopy((int)N,&X[t*N],1,tmp,1);
                    cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,tmp,1);
                    for (size_t n=0u; n<N; ++n) { H[n] = 1.0/(1.0+exp(-tmp[n])); }
                    cblas_dcopy((int)N,B,1,tmp,1);
                    cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,tmp,1);
                    for (size_t n=0u; n<N; ++n) { Y[t*N+n] = 1.0/(1.0+exp(-tmp[n])); }
                }
            }
            else
            {
                for (size_t t=0u; t<T; ++t)
                {
                    cblas_dcopy((int)N,&X[t],(int)T,tmp,1);
                    cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,tmp,1);
                    for (size_t n=0u; n<N; ++n) { H[n] = 1.0/(1.0+exp(-tmp[n])); }
                    cblas_dcopy((int)N,B,1,tmp,1);
                    cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,tmp,1);
                    for (size_t n=0u; n<N; ++n) { Y[t+n*T] = 1.0/(1.0+exp(-tmp[n])); }
                }
            }
        }
        else if (dim==1u)
        {
            if (iscolmajor)
            {
                for (size_t t=0u; t<T; ++t)
                {
                    cblas_dcopy((int)N,&X[t],(int)T,tmp,1);
                    cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,tmp,1);
                    for (size_t n=0u; n<N; ++n) { H[n] = 1.0/(1.0+exp(-tmp[n])); }
                    cblas_dcopy((int)N,B,1,tmp,1);
                    cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,tmp,1);
                    for (size_t n=0u; n<N; ++n) { Y[t+n*T] = 1.0/(1.0+exp(-tmp[n])); }
                }
            }
            else
            {
                for (size_t t=0u; t<T; ++t)
                {
                    cblas_dcopy((int)N,&X[t*N],1,tmp,1);
                    cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,tmp,1);
                    for (size_t n=0u; n<N; ++n) { H[n] = 1.0/(1.0+exp(-tmp[n])); }
                    cblas_dcopy((int)N,B,1,tmp,1);
                    cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,tmp,1);
                    for (size_t n=0u; n<N; ++n) { Y[t*N+n] = 1.0/(1.0+exp(-tmp[n])); }
                }
            }
        }
        else
        {
            fprintf(stderr,"error in elman_d: dim must be 0 or 1.\n"); return 1;
        }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
