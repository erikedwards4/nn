//This does CELL (~soma) stage of minimal GRU (gated recurrent unit) model.
//This requires each neuron to have 2 input time series, X and Xf,
//where X is the usual input and Xf the input for the forget-gate.
//Both X and Xf are the output of a linear IN stage (weights and baises).

//For dim=0: F[:,t] = sig(Xf[:,t] + Uf*Y[:,t-1])
//           H[:,t] = F[:,t].*Y[:,t-1]
//           Y[:,t] = H[:,t] + (1-f[:,t]).*tanh(X[:,t] + U*H[:,t])
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

//Unlike v2, this version takes only one X input: X = [X; Xf] for dim=0
//                                                X = [X Xf]  for dim=1
//This is essentially to allow treatment as a layer with Linux piping conventions.


#include <stdio.h>
#include <math.h>
#include <cblas.h>
//#include <time.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int gru_min_s (float *Y, const float *X, const float *U, const float *Uf, const int N, const int T, const int dim, const char iscolmajor);
int gru_min_d (double *Y, const double *X, const double *U, const double *Uf, const int N, const int T, const int dim, const char iscolmajor);

int gru_min_inplace_s (float *X, const float *U, const float *Uf, const int N, const int T, const int dim, const char iscolmajor);
int gru_min_inplace_d (double *X, const double *U, const double *Uf, const int N, const int T, const int dim, const char iscolmajor);


int gru_min_s (float *Y, const float *X, const float *U, const float *Uf, const int N, const int T, const int dim, const char iscolmajor)
{
    const float o = 1.0f;
    const int NT = N*T;
    int n, t, nT, tN, tN2;
    float *F, *H;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in gru_min_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in gru_min_s: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(F=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in gru_min_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in gru_min_s: problem with malloc. "); perror("malloc"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                F[n] = 1.0f / (1.0f+expf(-X[N+n]));
                Y[n] = (1.0f-F[n]) * tanhf(X[n]);
            }
            for (t=1; t<T; t++)
            {
                tN = t*N; tN2 = 2*tN;
                cblas_scopy(N,&X[tN2+N],1,F,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uf,N,&Y[tN-N],1,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n] * Y[tN-N+n];
                }
                cblas_scopy(N,&X[tN2],1,&Y[tN],1);
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
                F[n] = 1.0f / (1.0f+expf(-X[NT+nT]));
                Y[nT] = (1.0f-F[n]) * tanhf(X[nT]);
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&X[NT+t],T,F,1);
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
                F[n] = 1.0f / (1.0f+expf(-X[NT+nT]));
                Y[nT] = (1.0f-F[n]) * tanhf(X[nT]);
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&X[NT+t],T,F,1);
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
                F[n] = 1.0f / (1.0f+expf(-X[N+n]));
                Y[n] = (1.0f-F[n]) * tanhf(X[n]);
            }
            for (t=1; t<T; t++)
            {
                tN = t*N; tN2 = 2*tN;
                cblas_scopy(N,&X[tN2+N],1,F,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Uf,N,&Y[tN-N],1,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n] * Y[tN-N+n];
                }
                cblas_scopy(N,&X[tN2],1,&Y[tN],1);
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
        fprintf(stderr,"error in gru_min_s: dim must be 0 or 1.\n"); return 1;
    }

    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);
    return 0;
}


int gru_min_d (double *Y, const double *X, const double *U, const double *Uf, const int N, const int T, const int dim, const char iscolmajor)
{
    const double o = 1.0;
    const int NT = N*T;
    int n, t, nT, tN, tN2;
    double *F, *H;

    //Checks
    if (N<1) { fprintf(stderr,"error in gru_min_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in gru_min_d: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(F=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in gru_min_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in gru_min_d: problem with malloc. "); perror("malloc"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                F[n] = 1.0 / (1.0+exp(-X[N+n]));
                Y[n] = (1.0-F[n]) * tanh(X[n]);
            }
            for (t=1; t<T; t++)
            {
                tN = t*N; tN2 = 2*tN;
                cblas_dcopy(N,&X[tN2+N],1,F,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Uf,N,&Y[tN-N],1,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n] * Y[tN-N+n];
                }
                cblas_dcopy(N,&X[tN2],1,&Y[tN],1);
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
                F[n] = 1.0 / (1.0+exp(-X[NT+nT]));
                Y[nT] = (1.0-F[n]) * tanh(X[nT]);
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&X[NT+t],T,F,1);
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
                F[n] = 1.0 / (1.0+exp(-X[NT+nT]));
                Y[nT] = (1.0-F[n]) * tanh(X[nT]);
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&X[NT+t],T,F,1);
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
                F[n] = 1.0 / (1.0+exp(-X[N+n]));
                Y[n] = (1.0-F[n]) * tanh(X[n]);
            }
            for (t=1; t<T; t++)
            {
                tN = t*N; tN2 = 2*tN;
                cblas_dcopy(N,&X[tN2+N],1,F,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Uf,N,&Y[tN-N],1,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n] * Y[tN-N+n];
                }
                cblas_dcopy(N,&X[tN2],1,&Y[tN],1);
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
        fprintf(stderr,"error in gru_min_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}



int gru_min_inplace_s (float *X, const float *U, const float *Uf, const int N, const int T, const int dim, const char iscolmajor)
{
    const float o = 1.0f;
    const int NT = N*T, N2 = 2*N;
    int n, t, nT, tN, tN2;
    float *F, *H;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in gru_inplace_min2_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in gru_inplace_min2_s: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(F=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in gru_inplace_min2_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in gru_inplace_min2_s: problem with malloc. "); perror("malloc"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (dim==0)
    {
        if (iscolmajor)
        {
            fprintf(stderr,"d0, colmajor \n");
            for (n=0; n<N; n++)
            {
                F[n] = 1.0f / (1.0f+expf(-X[N+n]));
                X[n] = (1.0f-F[n]) * tanhf(X[n]);
            }
            for (t=1; t<T; t++)
            {
                tN = t*N; tN2 = 2*tN;
                cblas_scopy(N,&X[tN2+N],1,F,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uf,N,&X[tN2-N2],1,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n]*X[tN2-N2+n];
                }
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,U,N,H,1,o,&X[tN2],1);
                for (n=0; n<N; n++)
                {
                    X[tN2+n] = F[n]*X[tN2-N2+n] + (1.0f-F[n])*tanhf(X[tN2+n]);
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                F[n] = 1.0f / (1.0f+expf(-X[NT+nT]));
                X[nT] = (1.0f-F[n]) * tanhf(X[nT]);
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&X[NT+t],T,F,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uf,N,&X[t-1],T,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n]*X[t-1+n*T];
                }
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,U,N,H,1,o,&X[t],T);
                for (n=0; n<N; n++)
                {
                    nT = n*T;
                    X[t+nT] = F[n]*X[t-1+nT] + (1.0f-F[n])*tanhf(X[t+nT]);
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
                F[n] = 1.0f / (1.0f+expf(-X[NT+nT]));
                X[nT] = (1.0f-F[n]) * tanhf(X[nT]);
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&X[NT+t],T,F,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Uf,N,&X[t-1],T,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n]*X[t-1+n*T];
                }
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,U,N,H,1,o,&X[t],T);
                for (n=0; n<N; n++)
                {
                    nT = n*T;
                    X[t+nT] = F[n]*X[t-1+nT] + (1.0f-F[n])*tanhf(X[t+nT]);
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                F[n] = 1.0f / (1.0f+expf(-X[N+n]));
                X[n] = (1.0f-F[n]) * tanhf(X[n]);
            }
            for (t=1; t<T; t++)
            {
                tN = t*N; tN2 = 2*tN;
                cblas_scopy(N,&X[tN2+N],1,F,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Uf,N,&X[tN2-N2],1,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0f / (1.0f+expf(-F[n]));
                    H[n] = F[n]*X[tN2-N2+n];
                }
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,U,N,H,1,o,&X[tN2],1);
                for (n=0; n<N; n++)
                {
                    X[tN2+n] = F[n]*X[tN2-N2+n] + (1.0f-F[n])*tanhf(X[tN2+n]);
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in gru_inplace_min2_s: dim must be 0 or 1.\n"); return 1;
    }

    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);
    return 0;
}


int gru_min_inplace_d (double *X, const double *U, const double *Uf, const int N, const int T, const int dim, const char iscolmajor)
{
    const double o = 1.0;
    const int NT = N*T, N2 = 2*N;
    int n, t, nT, tN, tN2;
    double *F, *H;

    //Checks
    if (N<1) { fprintf(stderr,"error in gru_inplace_min2_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in gru_inplace_min2_d: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(F=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in gru_inplace_min2_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in gru_inplace_min2_d: problem with malloc. "); perror("malloc"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                F[n] = 1.0 / (1.0+exp(-X[N+n]));
                X[n] = (1.0-F[n]) * tanh(X[n]);
            }
            for (t=1; t<T; t++)
            {
                tN = t*N; tN2 = 2*tN;
                cblas_dcopy(N,&X[tN2+N],1,F,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Uf,N,&X[tN2-N2],1,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n]*X[tN2-N2+n];
                }
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,U,N,H,1,o,&X[tN2],1);
                for (n=0; n<N; n++)
                {
                    X[tN2+n] = F[n]*X[tN2-N2+n] + (1.0-F[n])*tanh(X[tN2+n]);
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                F[n] = 1.0 / (1.0+exp(-X[NT+nT]));
                X[nT] = (1.0-F[n]) * tanh(X[nT]);
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&X[NT+t],T,F,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uf,N,&X[t-1],T,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n]*X[t-1+n*T];
                }
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,U,N,H,1,o,&X[t],T);
                for (n=0; n<N; n++)
                {
                    nT = n*T;
                    X[t+nT] = F[n]*X[t-1+nT] + (1.0-F[n])*tanh(X[t+nT]);
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
                F[n] = 1.0 / (1.0+exp(-X[NT+nT]));
                X[nT] = (1.0-F[n]) * tanh(X[nT]);
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&X[NT+t],T,F,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Uf,N,&X[t-1],T,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n]*X[t-1+n*T];
                }
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,U,N,H,1,o,&X[t],T);
                for (n=0; n<N; n++)
                {
                    nT = n*T;
                    X[t+nT] = F[n]*X[t-1+nT] + (1.0-F[n])*tanh(X[t+nT]);
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                F[n] = 1.0 / (1.0+exp(-X[N+n]));
                X[n] = (1.0-F[n]) * tanh(X[n]);
            }
            for (t=1; t<T; t++)
            {
                tN = t*N; tN2 = 2*tN;
                cblas_dcopy(N,&X[tN2+N],1,F,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Uf,N,&X[tN2-N2],1,o,F,1);
                for (n=0; n<N; n++)
                {
                    F[n] = 1.0 / (1.0+exp(-F[n]));
                    H[n] = F[n]*X[tN2-N2+n];
                }
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,U,N,H,1,o,&X[tN2],1);
                for (n=0; n<N; n++)
                {
                    X[tN2+n] = F[n]*X[tN2-N2+n] + (1.0-F[n])*tanh(X[tN2+n]);
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in gru_inplace_min2_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
