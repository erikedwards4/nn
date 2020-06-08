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
#include <math.h>
#include <cblas.h>
//#include <time.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int gru_min2_s (float *Y, const float *X, const float *Xf, const float *U, const float *Uf, const int N, const int T, const int dim, const char iscolmajor);
int gru_min2_d (double *Y, const double *X, const double *Xf, const double *U, const double *Uf, const int N, const int T, const int dim, const char iscolmajor);

int gru_min2_inplace_s (float *X, const float *Xf, const float *U, const float *Uf, const int N, const int T, const int dim, const char iscolmajor);
int gru_min2_inplace_d (double *X, const double *Xf, const double *U, const double *Uf, const int N, const int T, const int dim, const char iscolmajor);


int gru_min2_s (float *Y, const float *X, const float *Xf, const float *U, const float *Uf, const int N, const int T, const int dim, const char iscolmajor)
{
    const float o = 1.0f;
    int n, t, nT, tN;
    float *F, *H;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in gru_min2_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in gru_min2_s: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(F=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in gru_min2_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in gru_min2_s: problem with malloc. "); perror("malloc"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (N==1)
    {
        F[0] = 1.0f / (1.0f+expf(-Xf[0]));
        Y[0] = (1.0f-F[0]) * tanhf(X[0]);
        for (t=1; t<T; t++)
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
            for (n=0; n<N; n++)
            {
                F[n] = 1.0f / (1.0f+expf(-Xf[n]));
                Y[n] = (1.0f-F[n]) * tanhf(X[n]);
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_scopy(N,&Xf[tN],1,F,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uf,N,&Y[tN-N],1,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n] * Y[tN-N+n];
                }
                cblas_scopy(N,&X[tN],1,&Y[tN],1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,U,N,H,1,o,&Y[tN],1);
                for (n=0; n<N; n++)
                {
                    Y[tN+n] = H[n] + (1.0f-F[n])*tanhf(Y[tN+n]);
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                F[n] = 1.0f / (1.0f+expf(-Xf[nT]));
                Y[nT] = (1.0f-F[n]) * tanhf(X[nT]);
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&Xf[t],T,F,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uf,N,&Y[t-1],T,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n] * Y[t-1+n*T];
                }
                cblas_scopy(N,&X[t],T,&Y[t],T);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,U,N,H,1,o,&Y[t],T);
                for (n=0; n<N; n++)
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
            for (n=0; n<N; n++)
            {
                nT = n*T;
                F[n] = 1.0f / (1.0f+expf(-Xf[nT]));
                Y[nT] = (1.0f-F[n]) * tanhf(X[nT]);
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&Xf[t],T,F,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Uf,N,&Y[t-1],T,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n] * Y[t-1+n*T];
                }
                cblas_scopy(N,&X[t],T,&Y[t],T);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,U,N,H,1,o,&Y[t],T);
                for (n=0; n<N; n++)
                {
                    nT = n*T;
                    Y[t+nT] = H[n] + (1.0f-F[n])*tanhf(Y[t+nT]);
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                F[n] = 1.0f / (1.0f+expf(-Xf[n]));
                Y[n] = (1.0f-F[n]) * tanhf(X[n]);
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_scopy(N,&Xf[tN],1,F,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Uf,N,&Y[tN-N],1,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n] * Y[tN-N+n];
                }
                cblas_scopy(N,&X[tN],1,&Y[tN],1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,U,N,H,1,o,&Y[tN],1);
                for (n=0; n<N; n++)
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

    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);
    return 0;
}


int gru_min2_d (double *Y, const double *X, const double *Xf, const double *U, const double *Uf, const int N, const int T, const int dim, const char iscolmajor)
{
    const double o = 1.0;
    int n, t, nT, tN;
    double *F, *H;

    //Checks
    if (N<1) { fprintf(stderr,"error in gru_min2_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in gru_min2_d: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(F=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in gru_min2_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in gru_min2_d: problem with malloc. "); perror("malloc"); return 1; }

    if (N==1)
    {
        F[0] = 1.0 / (1.0+exp(-X[0]));
        Y[0] = (1.0-F[0]) * tanh(X[0]);
        for (t=1; t<T; t++)
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
            for (n=0; n<N; n++)
            {
                F[n] = 1.0 / (1.0+exp(-Xf[n]));
                Y[n] = (1.0-F[n]) * tanh(X[n]);
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_dcopy(N,&Xf[tN],1,F,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Uf,N,&Y[tN-N],1,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n] * Y[tN-N+n];
                }
                cblas_dcopy(N,&X[tN],1,&Y[tN],1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,U,N,H,1,o,&Y[tN],1);
                for (n=0; n<N; n++)
                {
                    Y[tN+n] = H[n] + (1.0-F[n])*tanh(Y[tN+n]);
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                F[n] = 1.0 / (1.0+exp(-Xf[nT]));
                Y[nT] = (1.0-F[n]) * tanh(X[nT]);
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&Xf[t],T,F,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uf,N,&Y[t-1],T,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n] * Y[t-1+n*T];
                }
                cblas_dcopy(N,&X[t],T,&Y[t],T);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,U,N,H,1,o,&Y[t],T);
                for (n=0; n<N; n++)
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
            for (n=0; n<N; n++)
            {
                nT = n*T;
                F[n] = 1.0 / (1.0+exp(-Xf[nT]));
                Y[nT] = (1.0-F[n]) * tanh(X[nT]);
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&Xf[t],T,F,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Uf,N,&Y[t-1],T,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n] * Y[t-1+n*T];
                }
                cblas_dcopy(N,&X[t],T,&Y[t],T);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,U,N,H,1,o,&Y[t],T);
                for (n=0; n<N; n++)
                {
                    nT = n*T;
                    Y[t+nT] = H[n] + (1.0-F[n])*tanh(Y[t+nT]);
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                F[n] = 1.0 / (1.0+exp(-Xf[n]));
                Y[n] = (1.0-F[n]) * tanh(X[n]);
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_dcopy(N,&Xf[tN],1,F,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Uf,N,&Y[tN-N],1,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n] * Y[tN-N+n];
                }
                cblas_dcopy(N,&X[tN],1,&Y[tN],1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,U,N,H,1,o,&Y[tN],1);
                for (n=0; n<N; n++)
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



int gru_min2_inplace_s (float *X, const float *Xf, const float *U, const float *Uf, const int N, const int T, const int dim, const char iscolmajor)
{
    const float o = 1.0f;
    int n, t, nT, tN;
    float *F, *H;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in gru_min2_inplace_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in gru_min2_inplace_s: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(F=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in gru_min2_inplace_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in gru_min2_inplace_s: problem with malloc. "); perror("malloc"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (N==1)
    {
        F[0] = 1.0f / (1.0f+expf(-X[0]));
        X[0] = (1.0f-F[0]) * tanhf(X[0]);
        for (t=1; t<T; t++)
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
            for (n=0; n<N; n++)
            {
                F[n] = 1.0f / (1.0f+expf(-Xf[n]));
                X[n] = (1.0f-F[n]) * tanhf(X[n]);
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_scopy(N,&Xf[tN],1,F,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uf,N,&X[tN-N],1,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n] * X[tN-N+n];
                }
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,U,N,H,1,o,&X[tN],1);
                for (n=0; n<N; n++)
                {
                    X[tN+n] = H[n] + (1.0f-F[n])*tanhf(X[tN+n]);
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                F[n] = 1.0f / (1.0f+expf(-Xf[nT]));
                X[nT] = (1.0f-F[n]) * tanhf(X[nT]);
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&Xf[t],T,F,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uf,N,&X[t-1],T,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n] * X[t-1+n*T];
                }
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,U,N,H,1,o,&X[t],T);
                for (n=0; n<N; n++)
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
            for (n=0; n<N; n++)
            {
                nT = n*T;
                F[n] = 1.0f / (1.0f+expf(-Xf[nT]));
                X[nT] = (1.0f-F[n]) * tanhf(X[nT]);
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&Xf[t],T,F,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Uf,N,&X[t-1],T,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n] * X[t-1+n*T];
                }
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,U,N,H,1,o,&X[t],T);
                for (n=0; n<N; n++)
                {
                    nT = n*T;
                    X[t+nT] = H[n] + (1.0f-F[n])*tanhf(X[t+nT]);
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                F[n] = 1.0f / (1.0f+expf(-Xf[n]));
                X[n] = (1.0f-F[n]) * tanhf(X[n]);
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_scopy(N,&Xf[tN],1,F,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Uf,N,&X[tN-N],1,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n] * X[tN-N+n];
                }
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,U,N,H,1,o,&X[tN],1);
                for (n=0; n<N; n++)
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

    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);
    return 0;
}


int gru_min2_inplace_d (double *X, const double *Xf, const double *U, const double *Uf, const int N, const int T, const int dim, const char iscolmajor)
{
    const double o = 1.0;
    int n, t, nT, tN;
    double *F, *H;

    //Checks
    if (N<1) { fprintf(stderr,"error in gru_min2_inplace_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in gru_min2_inplace_d: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(F=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in gru_min2_inplace_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in gru_min2_inplace_d: problem with malloc. "); perror("malloc"); return 1; }

    if (N==1)
    {
        F[0] = 1.0 / (1.0+exp(-X[0]));
        X[0] = (1.0-F[0]) * tanh(X[0]);
        for (t=1; t<T; t++)
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
            for (n=0; n<N; n++)
            {
                F[n] = 1.0 / (1.0+exp(-Xf[n]));
                X[n] = (1.0-F[n]) * tanh(X[n]);
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_dcopy(N,&Xf[tN],1,F,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Uf,N,&X[tN-N],1,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n] * X[tN-N+n];
                }
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,U,N,H,1,o,&X[tN],1);
                for (n=0; n<N; n++)
                {
                    X[tN+n] = H[n] + (1.0-F[n])*tanh(X[tN+n]);
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                F[n] = 1.0 / (1.0+exp(-Xf[nT]));
                X[nT] = (1.0-F[n]) * tanh(X[nT]);
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&Xf[t],T,F,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uf,N,&X[t-1],T,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n] * X[t-1+n*T];
                }
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,U,N,H,1,o,&X[t],T);
                for (n=0; n<N; n++)
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
            for (n=0; n<N; n++)
            {
                nT = n*T;
                F[n] = 1.0 / (1.0+exp(-Xf[nT]));
                X[nT] = (1.0-F[n]) * tanh(X[nT]);
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&Xf[t],T,F,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Uf,N,&X[t-1],T,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n] * X[t-1+n*T];
                }
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,U,N,H,1,o,&X[t],T);
                for (n=0; n<N; n++)
                {
                    nT = n*T;
                    X[t+nT] = H[n] + (1.0-F[n])*tanh(X[t+nT]);
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                F[n] = 1.0 / (1.0+exp(-Xf[n]));
                X[n] = (1.0-F[n]) * tanh(X[n]);
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_dcopy(N,&Xf[tN],1,F,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Uf,N,&X[tN-N],1,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n] * X[tN-N+n];
                }
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,U,N,H,1,o,&X[tN],1);
                for (n=0; n<N; n++)
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
