//This does CELL (~soma) stage of minimal GRU (gated recurrent unit) model.
//This requires each neuron to have 2 input time series, X and Xf,
//where X is the usual input and Xf the input for the forget-gate.
//Both X and Xf are the output of a linear IN stage (weights and baises).

//For dim=0: F[:,t] = sig(Xf[:,t] + Uf*Y[:,t-1])
//           H[:,t] = F[:,t].*Y[:,t-1]
//           Y[:,t] = H[:,t] + (1-F[:,t]).*tanh(X[:,t] + U*H[:,t])
//
//For dim=1: F[t,:] = sig(Xf[t,:] + Y[t-1,:]*Uf)
//           H[t,:] = F[t,:].*Y[t-1,:]
//           Y[t,:] = H[t,:] + (1-F[t,:]).*tanh(X[t,:] + H[t,:]*U)
//
//where sig is the logistic (sigmoid) nonlinearity = 1/(1+exp(-x)),
//F is the forget gate signal, H is an intermediate vector,
//U and Uf are NxN update matrices, and Y is the output.

//Note that, the neurons of a layer are independent only if U and Uf are diagonal matrices.
//This is only really a CELL (~soma) stage in that case.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int gru_min2_s (float *Y, const float *X, const float *Xf, const float *U, const float *Uf, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int gru_min2_d (double *Y, const double *X, const double *Xf, const double *U, const double *Uf, const size_t N, const size_t T, const char iscolmajor, const size_t dim);

int gru_min2_inplace_s (float *X, const float *Xf, const float *U, const float *Uf, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int gru_min2_inplace_d (double *X, const double *Xf, const double *U, const double *Uf, const size_t N, const size_t T, const char iscolmajor, const size_t dim);


int gru_min2_s (float *Y, const float *X, const float *Xf, const float *U, const float *Uf, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const float o = 1.0f;
    size_t nT, tN;

    float *F, *H;
    if (!(F=(float *)malloc(N*sizeof(float)))) { fprintf(stderr,"error in gru_min2_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(float *)malloc(N*sizeof(float)))) { fprintf(stderr,"error in gru_min2_s: problem with malloc. "); perror("malloc"); return 1; }

    if (N==1)
    {
        F[0] = 1.0f / (1.0f+expf(-Xf[0]));
        Y[0] = (1.0f-F[0]) * tanhf(X[0]);
        for (size_t t=1; t<T; ++t)
        {
            F[0] = 1.0f / (1.0f+expf(-Xf[t]-Uf[0]*Y[t-1]));
            H[0] = F[0] * Y[t-1];
            Y[t] = H[0] + (1.0f-F[0])*tanhf(X[t]+U[0]*H[0]);
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                F[n] = 1.0f / (1.0f+expf(-Xf[n]));
                Y[n] = (1.0f-F[n]) * tanhf(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N;
                cblas_scopy((int)N,&Xf[tN],1,F,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Uf,(int)N,&Y[tN-N],1,o,F,1);
                for (size_t n=0; n<N; ++n)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n] * Y[tN-N+n];
                }
                cblas_scopy((int)N,&X[tN],1,&Y[tN],1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&Y[tN],1);
                for (size_t n=0; n<N; ++n)
                {
                    Y[tN+n] = H[n] + (1.0f-F[n])*tanhf(Y[tN+n]);
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                F[n] = 1.0f / (1.0f+expf(-Xf[nT]));
                Y[nT] = (1.0f-F[n]) * tanhf(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_scopy((int)N,&Xf[t],(int)T,F,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,Uf,(int)N,&Y[t-1],(int)T,o,F,1);
                for (size_t n=0; n<N; ++n)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n] * Y[t-1+n*T];
                }
                cblas_scopy((int)N,&X[t],(int)T,&Y[t],(int)T);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&Y[t],(int)T);
                for (size_t n=0; n<N; ++n)
                {
                    nT = n*T;
                    Y[t+nT] = H[n] + (1.0f-F[n])*tanhf(Y[t+nT]);
                }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                F[n] = 1.0f / (1.0f+expf(-Xf[nT]));
                Y[nT] = (1.0f-F[n]) * tanhf(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_scopy((int)N,&Xf[t],(int)T,F,1);
                cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,Uf,(int)N,&Y[t-1],(int)T,o,F,1);
                for (size_t n=0; n<N; ++n)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n] * Y[t-1+n*T];
                }
                cblas_scopy((int)N,&X[t],(int)T,&Y[t],(int)T);
                cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&Y[t],(int)T);
                for (size_t n=0; n<N; ++n)
                {
                    nT = n*T;
                    Y[t+nT] = H[n] + (1.0f-F[n])*tanhf(Y[t+nT]);
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                F[n] = 1.0f / (1.0f+expf(-Xf[n]));
                Y[n] = (1.0f-F[n]) * tanhf(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N;
                cblas_scopy((int)N,&Xf[tN],1,F,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Uf,(int)N,&Y[tN-N],1,o,F,1);
                for (size_t n=0; n<N; ++n)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n] * Y[tN-N+n];
                }
                cblas_scopy((int)N,&X[tN],1,&Y[tN],1);
                cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&Y[tN],1);
                for (size_t n=0; n<N; ++n)
                {
                    Y[tN+n] = H[n] + (1.0f-F[n])*tanhf(Y[tN+n]);
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in gru_min2_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int gru_min2_d (double *Y, const double *X, const double *Xf, const double *U, const double *Uf, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const double o = 1.0;
    size_t nT, tN;

    double *F, *H;
    if (!(F=(double *)malloc(N*sizeof(double)))) { fprintf(stderr,"error in gru_min2_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(double *)malloc(N*sizeof(double)))) { fprintf(stderr,"error in gru_min2_d: problem with malloc. "); perror("malloc"); return 1; }

    if (N==1)
    {
        F[0] = 1.0 / (1.0+exp(-X[0]));
        Y[0] = (1.0-F[0]) * tanh(X[0]);
        for (size_t t=1; t<T; ++t)
        {
            F[0] = 1.0 / (1.0+exp(-X[t]-Uf[0]*Y[t-1]));
            H[0] = F[0] * Y[t-1];
            Y[t] = H[0] + (1.0-F[0])*tanh(X[t]+U[0]*H[0]);
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                F[n] = 1.0 / (1.0+exp(-Xf[n]));
                Y[n] = (1.0-F[n]) * tanh(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N;
                cblas_dcopy((int)N,&Xf[tN],1,F,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Uf,(int)N,&Y[tN-N],1,o,F,1);
                for (size_t n=0; n<N; ++n)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n] * Y[tN-N+n];
                }
                cblas_dcopy((int)N,&X[tN],1,&Y[tN],1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&Y[tN],1);
                for (size_t n=0; n<N; ++n)
                {
                    Y[tN+n] = H[n] + (1.0-F[n])*tanh(Y[tN+n]);
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                F[n] = 1.0 / (1.0+exp(-Xf[nT]));
                Y[nT] = (1.0-F[n]) * tanh(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_dcopy((int)N,&Xf[t],(int)T,F,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,Uf,(int)N,&Y[t-1],(int)T,o,F,1);
                for (size_t n=0; n<N; ++n)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n] * Y[t-1+n*T];
                }
                cblas_dcopy((int)N,&X[t],(int)T,&Y[t],(int)T);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&Y[t],(int)T);
                for (size_t n=0; n<N; ++n)
                {
                    nT = n*T;
                    Y[t+nT] = H[n] + (1.0-F[n])*tanh(Y[t+nT]);
                }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                F[n] = 1.0 / (1.0+exp(-Xf[nT]));
                Y[nT] = (1.0-F[n]) * tanh(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_dcopy((int)N,&Xf[t],(int)T,F,1);
                cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,Uf,(int)N,&Y[t-1],(int)T,o,F,1);
                for (size_t n=0; n<N; ++n)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n] * Y[t-1+n*T];
                }
                cblas_dcopy((int)N,&X[t],(int)T,&Y[t],(int)T);
                cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&Y[t],(int)T);
                for (size_t n=0; n<N; ++n)
                {
                    nT = n*T;
                    Y[t+nT] = H[n] + (1.0-F[n])*tanh(Y[t+nT]);
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                F[n] = 1.0 / (1.0+exp(-Xf[n]));
                Y[n] = (1.0-F[n]) * tanh(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N;
                cblas_dcopy((int)N,&Xf[tN],1,F,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Uf,(int)N,&Y[tN-N],1,o,F,1);
                for (size_t n=0; n<N; ++n)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n] * Y[tN-N+n];
                }
                cblas_dcopy((int)N,&X[tN],1,&Y[tN],1);
                cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&Y[tN],1);
                for (size_t n=0; n<N; ++n)
                {
                    Y[tN+n] = H[n] + (1.0-F[n])*tanh(Y[tN+n]);
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in gru_min2_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}



int gru_min2_inplace_s (float *X, const float *Xf, const float *U, const float *Uf, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const float o = 1.0f;
    size_t nT, tN;

    float *F, *H;
    if (!(F=(float *)malloc(N*sizeof(float)))) { fprintf(stderr,"error in gru_min2_inplace_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(float *)malloc(N*sizeof(float)))) { fprintf(stderr,"error in gru_min2_inplace_s: problem with malloc. "); perror("malloc"); return 1; }

    if (N==1)
    {
        F[0] = 1.0f / (1.0f+expf(-X[0]));
        X[0] = (1.0f-F[0]) * tanhf(X[0]);
        for (size_t t=1; t<T; ++t)
        {
            F[0] = 1.0f / (1.0f+expf(-X[t]-Uf[0]*X[t-1]));
            H[0] = F[0]*X[t-1];
            X[t] = H[0] + (1.0f-F[0])*tanhf(X[t]+U[0]*H[0]);
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                F[n] = 1.0f / (1.0f+expf(-Xf[n]));
                X[n] = (1.0f-F[n]) * tanhf(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N;
                cblas_scopy((int)N,&Xf[tN],1,F,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Uf,(int)N,&X[tN-N],1,o,F,1);
                for (size_t n=0; n<N; ++n)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n] * X[tN-N+n];
                }
                cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[tN],1);
                for (size_t n=0; n<N; ++n)
                {
                    X[tN+n] = H[n] + (1.0f-F[n])*tanhf(X[tN+n]);
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                F[n] = 1.0f / (1.0f+expf(-Xf[nT]));
                X[nT] = (1.0f-F[n]) * tanhf(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_scopy((int)N,&Xf[t],(int)T,F,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,Uf,(int)N,&X[t-1],(int)T,o,F,1);
                for (size_t n=0; n<N; ++n)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n] * X[t-1+n*T];
                }
                cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[t],(int)T);
                for (size_t n=0; n<N; ++n)
                {
                    nT = n*T;
                    X[t+nT] = H[n] + (1.0f-F[n])*tanhf(X[t+nT]);
                }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                F[n] = 1.0f / (1.0f+expf(-Xf[nT]));
                X[nT] = (1.0f-F[n]) * tanhf(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_scopy((int)N,&Xf[t],(int)T,F,1);
                cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,Uf,(int)N,&X[t-1],(int)T,o,F,1);
                for (size_t n=0; n<N; ++n)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n] * X[t-1+n*T];
                }
                cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[t],(int)T);
                for (size_t n=0; n<N; ++n)
                {
                    nT = n*T;
                    X[t+nT] = H[n] + (1.0f-F[n])*tanhf(X[t+nT]);
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                F[n] = 1.0f / (1.0f+expf(-Xf[n]));
                X[n] = (1.0f-F[n]) * tanhf(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N;
                cblas_scopy((int)N,&Xf[tN],1,F,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Uf,(int)N,&X[tN-N],1,o,F,1);
                for (size_t n=0; n<N; ++n)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n] * X[tN-N+n];
                }
                cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[tN],1);
                for (size_t n=0; n<N; ++n)
                {
                    X[tN+n] = H[n] + (1.0f-F[n])*tanhf(X[tN+n]);
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in gru_min2_inplace_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int gru_min2_inplace_d (double *X, const double *Xf, const double *U, const double *Uf, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const double o = 1.0;
    size_t nT, tN;
    
    double *F, *H;
    if (!(F=(double *)malloc(N*sizeof(double)))) { fprintf(stderr,"error in gru_min2_inplace_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(double *)malloc(N*sizeof(double)))) { fprintf(stderr,"error in gru_min2_inplace_d: problem with malloc. "); perror("malloc"); return 1; }

    if (N==1)
    {
        F[0] = 1.0 / (1.0+exp(-X[0]));
        X[0] = (1.0-F[0]) * tanh(X[0]);
        for (size_t t=1; t<T; ++t)
        {
            F[0] = 1.0 / (1.0+exp(-X[t]-Uf[0]*X[t-1]));
            H[0] = F[0]*X[t-1];
            X[t] = H[0] + (1.0-F[0])*tanh(X[t]+U[0]*H[0]);
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                F[n] = 1.0 / (1.0+exp(-Xf[n]));
                X[n] = (1.0-F[n]) * tanh(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N;
                cblas_dcopy((int)N,&Xf[tN],1,F,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Uf,(int)N,&X[tN-N],1,o,F,1);
                for (size_t n=0; n<N; ++n)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n] * X[tN-N+n];
                }
                cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[tN],1);
                for (size_t n=0; n<N; ++n)
                {
                    X[tN+n] = H[n] + (1.0-F[n])*tanh(X[tN+n]);
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                F[n] = 1.0 / (1.0+exp(-Xf[nT]));
                X[nT] = (1.0-F[n]) * tanh(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_dcopy((int)N,&Xf[t],(int)T,F,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,Uf,(int)N,&X[t-1],(int)T,o,F,1);
                for (size_t n=0; n<N; ++n)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n] * X[t-1+n*T];
                }
                cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[t],(int)T);
                for (size_t n=0; n<N; ++n)
                {
                    nT = n*T;
                    X[t+nT] = H[n] + (1.0-F[n])*tanh(X[t+nT]);
                }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                F[n] = 1.0 / (1.0+exp(-Xf[nT]));
                X[nT] = (1.0-F[n]) * tanh(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_dcopy((int)N,&Xf[t],(int)T,F,1);
                cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,Uf,(int)N,&X[t-1],(int)T,o,F,1);
                for (size_t n=0; n<N; ++n)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n] * X[t-1+n*T];
                }
                cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[t],(int)T);
                for (size_t n=0; n<N; ++n)
                {
                    nT = n*T;
                    X[t+nT] = H[n] + (1.0-F[n])*tanh(X[t+nT]);
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                F[n] = 1.0 / (1.0+exp(-Xf[n]));
                X[n] = (1.0-F[n]) * tanh(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N;
                cblas_dcopy((int)N,&Xf[tN],1,F,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Uf,(int)N,&X[tN-N],1,o,F,1);
                for (size_t n=0; n<N; ++n)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n] * X[tN-N+n];
                }
                cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[tN],1);
                for (size_t n=0; n<N; ++n)
                {
                    X[tN+n] = H[n] + (1.0-F[n])*tanh(X[tN+n]);
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in gru_min2_inplace_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
