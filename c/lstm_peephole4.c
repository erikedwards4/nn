//This does CELL (~soma) stage of peephole LSTM (long short-term memory) model.
//This requires each neuron to have 4 input time series, Xc, Xi, Xf, Xo,
//where Xc is the usual (or "cellular") input and Xi, Xf, Xo the inputs for the input, forget, output gates.
//Xc, Xi, Xf, Xo are the output of separate linear IN stages (weights and baises).

//For dim=0, I[:,t] = sig{Xi[:,t] + Ui*C[:,t-1]}
//           F[:,t] = sig{Xf[:,t] + Uf*C[:,t-1]}
//           O[:,t] = sig{Xo[:,t] + Uo*C[:,t-1]}
//           C[:,t] = F[:,t].*C[:,t-1] + I[:,t].*sig{Xc[t,:]}
//           Y[:,t] = tanh{O[:,t].*C[:,t]}
//with sizes Xc, Xi, Xf, Xo: N x T
//               Ui, Uf, Uo: N x N
//                        Y: N x T
//
//For dim=1, I[t,:] = sig{Xi[t,:] + C[t-1,:]*Ui} 
//           F[t,:] = sig{Xf[t,:] + C[t-1,:]*Uf} 
//           O[t,:] = sig{Xo[t,:] + C[t-1,:]*Uo}
//           C[t,:] = F[t,:].*C[t-1,:] + I[t,:].*sig{Xc[t,:]}
//           Y[t,:] = tanh{O[t,:].*C[t,:]}
//with sizes Xc, Xi, Xf, Xo: T x N
//               Ui, Uf, Uo: N x N
//                        Y: T x N
//
//where sig is the logistic (sigmoid) nonlinearity = 1/(1+exp(-x)),
//I is the input gate, F is the forget gate, O is the output gate,
//C is the "cell input activation vector",
//H is an intermediate (hidden) vector (sometimes called the "cell state vector"),
//Uc, Ui, Uf, Uo are NxN matrices, and Y is the final output (sometimes called the "hidden state vector").

//The tanh is omitted here!! (To allow trying other nonlinearities.)

//Note that, the neurons of a layer are independent only if Uc, Ui, Uf, Uo are diagonal matrices.
//This is only really a CELL (~soma) stage in that case.

#include <stdio.h>
#include <math.h>
#include <cblas.h>
//#include <time.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int lstm_peephole4_s (float *Y, const float *Xc, const float *Xi, const float *Xf, const float *Xo, const float *Ui, const float *Uf, const float *Uo, const int N, const int T, const int dim, const char iscolmajor);
int lstm_peephole4_d (double *Y, const double *Xc, const double *Xi, const double *Xf, const double *Xo, const double *Ui, const double *Uf, const double *Uo, const int N, const int T, const int dim, const char iscolmajor);

int lstm_peephole4_inplace_s (float *Xc, const float *Xi, const float *Xf, const float *Xo, const float *Ui, const float *Uf, const float *Uo, const int N, const int T, const int dim, const char iscolmajor);
int lstm_peephole4_inplace_d (double *Xc, const double *Xi, const double *Xf, const double *Xo, const double *Ui, const double *Uf, const double *Uo, const int N, const int T, const int dim, const char iscolmajor);


int lstm_peephole4_s (float *Y, const float *Xc, const float *Xi, const float *Xf, const float *Xo, const float *Ui, const float *Uf, const float *Uo, const int N, const int T, const int dim, const char iscolmajor)
{
    const float o = 1.0f;
    int n, t, nT, tN;
    float *C, *I, *F, *O;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in lstm_peephole4_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in lstm_peephole4_s: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(C=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm_peephole4_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(I=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm_peephole4_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(F=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm_peephole4_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(O=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm_peephole4_s: problem with malloc. "); perror("malloc"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (N==1)
    {
        I[0] = 1.0f / (1.0f+expf(-Xi[0]));
        C[0] = I[0] / (1.0f+expf(-Xc[0]));
        Y[0] = C[0] / (1.0f+expf(-Xo[0]));
        for (t=1; t<T; t++)
        {
            I[0] = 1.0f / (1.0f+expf(-Xi[t]-Ui[0]*C[0]));
            O[0] = 1.0f / (1.0f+expf(-Xo[t]-Uo[0]*C[0]));
            C[0] = C[0]/(1.0f+expf(-Xf[t]-Uf[0]*C[0])) + I[0]/(1.0f+expf(-Xc[t]));
            Y[t] = O[0] * C[0];
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                I[n] = 1.0f / (1.0f+expf(-Xi[n]));
                C[n] = I[n] / (1.0f+expf(-Xc[n]));
                Y[n] = C[n] / (1.0f+expf(-Xo[n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_scopy(N,&Xi[tN],1,I,1); cblas_scopy(N,&Xf[tN],1,F,1); cblas_scopy(N,&Xo[tN],1,O,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Ui,N,C,1,o,I,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uf,N,C,1,o,F,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uo,N,C,1,o,O,1);
                for (n=0; n<N; n++)
                {
                    I[n] = 1.0f / (1.0f+expf(-I[n]));
                    C[n] = C[n]/(1.0f+expf(-F[n])) + I[n]/(1.0f+expf(-Xc[tN+n]));
                    Y[tN+n] = C[n] / (1.0f+expf(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                I[n] = 1.0f / (1.0f+expf(-Xi[nT]));
                C[n] = I[n] / (1.0f+expf(-Xc[nT]));
                Y[nT] = C[n] / (1.0f+expf(-Xo[nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&Xi[t],T,I,1); cblas_scopy(N,&Xf[t],T,F,1); cblas_scopy(N,&Xo[t],T,O,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Ui,N,C,1,o,I,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uf,N,C,1,o,F,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uo,N,C,1,o,O,1);
                for (n=0; n<N; n++)
                {
                    I[n] = 1.0f / (1.0f+expf(-I[n]));
                    C[n] = C[n]/(1.0f+expf(-F[n])) + I[n]/(1.0f+expf(-Xc[t+n*T]));
                    Y[t+n*T] = C[n] / (1.0f+expf(-O[n]));
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
                I[n] = 1.0f / (1.0f+expf(-Xi[nT]));
                C[n] = I[n] / (1.0f+expf(-Xc[nT]));
                Y[nT] = C[n] / (1.0f+expf(-Xo[nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&Xi[t],T,I,1); cblas_scopy(N,&Xf[t],T,F,1); cblas_scopy(N,&Xo[t],T,O,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Ui,N,C,1,o,I,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Uf,N,C,1,o,F,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Uo,N,C,1,o,O,1);
                for (n=0; n<N; n++)
                {
                    I[n] = 1.0f / (1.0f+expf(-I[n]));
                    C[n] = C[n]/(1.0f+expf(-F[n])) + I[n]/(1.0f+expf(-Xc[t+n*T]));
                    Y[t+n*T] = C[n] / (1.0f+expf(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                I[n] = 1.0f / (1.0f+expf(-Xi[n]));
                C[n] = I[n] / (1.0f+expf(-Xc[n]));
                Y[n] = C[n] / (1.0f+expf(-Xo[n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_scopy(N,&Xi[tN],1,I,1); cblas_scopy(N,&Xf[tN],1,F,1); cblas_scopy(N,&Xo[tN],1,O,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Ui,N,C,1,o,I,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Uf,N,C,1,o,F,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Uo,N,C,1,o,O,1);
                for (n=0; n<N; n++)
                {
                    I[n] = 1.0f / (1.0f+expf(-I[n]));
                    C[n] = C[n]/(1.0f+expf(-F[n])) + I[n]/(1.0f+expf(-Xc[tN+n]));
                    Y[tN+n] = C[n] / (1.0f+expf(-O[n]));
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in lstm_peephole4_s: dim must be 0 or 1.\n"); return 1;
    }

    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);
    return 0;
}


int lstm_peephole4_d (double *Y, const double *Xc, const double *Xi, const double *Xf, const double *Xo, const double *Ui, const double *Uf, const double *Uo, const int N, const int T, const int dim, const char iscolmajor)
{
    const double o = 1.0;
    int n, t, nT, tN;
    double *C, *I, *F, *O;

    //Checks
    if (N<1) { fprintf(stderr,"error in lstm_peephole4_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in lstm_peephole4_d: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(C=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm_peephole4_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(I=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm_peephole4_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(F=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm_peephole4_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(O=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm_peephole4_d: problem with malloc. "); perror("malloc"); return 1; }

    if (N==1)
    {
        I[0] = 1.0 / (1.0+exp(-Xi[0]));
        C[0] = I[0] / (1.0+exp(-Xc[0]));
        Y[0] = C[0] / (1.0+exp(-Xo[0]));
        for (t=1; t<T; t++)
        {
            I[0] = 1.0 / (1.0+exp(-Xi[t]-Ui[0]*C[0]));
            O[0] = 1.0 / (1.0+exp(-Xo[t]-Uo[0]*C[0]));
            C[0] = C[0]/(1.0+exp(-Xf[t]-Uf[0]*C[0])) + I[0]/(1.0+exp(-Xc[t]));
            Y[t] = O[0] * C[0];
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                I[n] = 1.0 / (1.0+exp(-Xi[n]));
                C[n] = I[n] / (1.0+exp(-Xc[n]));
                Y[n] = C[n] / (1.0+exp(-Xo[n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_dcopy(N,&Xi[tN],1,I,1); cblas_dcopy(N,&Xf[tN],1,F,1); cblas_dcopy(N,&Xo[tN],1,O,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Ui,N,C,1,o,I,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Uf,N,C,1,o,F,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Uo,N,C,1,o,O,1);
                for (n=0; n<N; n++)
                {
                    I[n] = 1.0 / (1.0+exp(-I[n]));
                    C[n] = C[n]/(1.0+exp(-F[n])) + I[n]/(1.0+exp(-Xc[tN+n]));
                    Y[tN+n] = C[n] / (1.0+exp(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                I[n] = 1.0 / (1.0+exp(-Xi[nT]));
                C[n] = I[n] / (1.0+exp(-Xc[nT]));
                Y[nT] = C[n] / (1.0+exp(-Xo[nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&Xi[t],T,I,1); cblas_dcopy(N,&Xf[t],T,F,1); cblas_dcopy(N,&Xo[t],T,O,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Ui,N,C,1,o,I,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uf,N,C,1,o,F,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uo,N,C,1,o,O,1);
                for (n=0; n<N; n++)
                {
                    I[n] = 1.0 / (1.0+exp(-I[n]));
                    C[n] = C[n]/(1.0+exp(-F[n])) + I[n]/(1.0+exp(-Xc[t+n*T]));
                    Y[t+n*T] = C[n] / (1.0+exp(-O[n]));
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
                I[n] = 1.0 / (1.0+exp(-Xi[nT]));
                C[n] = I[n] / (1.0+exp(-Xc[nT]));
                Y[nT] = C[n] / (1.0+exp(-Xo[nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&Xi[t],T,I,1); cblas_dcopy(N,&Xf[t],T,F,1); cblas_dcopy(N,&Xo[t],T,O,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Ui,N,C,1,o,I,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Uf,N,C,1,o,F,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Uo,N,C,1,o,O,1);
                for (n=0; n<N; n++)
                {
                    I[n] = 1.0 / (1.0+exp(-I[n]));
                    C[n] = C[n]/(1.0+exp(-F[n])) + I[n]/(1.0+exp(-Xc[t+n*T]));
                    Y[t+n*T] = C[n] / (1.0+exp(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                I[n] = 1.0 / (1.0+exp(-Xi[n]));
                C[n] = I[n] / (1.0+exp(-Xc[n]));
                Y[n] = C[n] / (1.0+exp(-Xo[n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_dcopy(N,&Xi[tN],1,I,1); cblas_dcopy(N,&Xf[tN],1,F,1); cblas_dcopy(N,&Xo[tN],1,O,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Ui,N,C,1,o,I,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Uf,N,C,1,o,F,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Uo,N,C,1,o,O,1);
                for (n=0; n<N; n++)
                {
                    I[n] = 1.0 / (1.0+exp(-I[n]));
                    C[n] = C[n]/(1.0+exp(-F[n])) + I[n]/(1.0+exp(-Xc[tN+n]));
                    Y[tN+n] = C[n] / (1.0+exp(-O[n]));
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in lstm_peephole4_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}



int lstm_peephole4_inplace_s (float *Xc, const float *Xi, const float *Xf, const float *Xo, const float *Ui, const float *Uf, const float *Uo, const int N, const int T, const int dim, const char iscolmajor)
{
    const float o = 1.0f;
    int n, t, nT, tN;
    float *C, *I, *F, *O;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in lstm_peephole4_inplace_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in lstm_peephole4_inplace_s: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(C=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm_peephole4_inplace_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(I=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm_peephole4_inplace_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(F=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm_peephole4_inplace_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(O=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm_peephole4_inplace_s: problem with malloc. "); perror("malloc"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (N==1)
    {
        I[0] = 1.0f / (1.0f+expf(-Xi[0]));
        C[0] = I[0] / (1.0f+expf(-Xc[0]));
        Xc[0] = C[0] / (1.0f+expf(-Xo[0]));
        for (t=1; t<T; t++)
        {
            I[0] = 1.0f / (1.0f+expf(-Xi[t]-Ui[0]*C[0]));
            O[0] = 1.0f / (1.0f+expf(-Xo[t]-Uo[0]*C[0]));
            C[0] = C[0]/(1.0f+expf(-Xf[t]-Uf[0]*C[0])) + I[0]/(1.0f+expf(-Xc[t]));
            Xc[t] = O[0] * C[0];
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                I[n] = 1.0f / (1.0f+expf(-Xi[n]));
                C[n] = I[n] / (1.0f+expf(-Xc[n]));
                Xc[n] = C[n] / (1.0f+expf(-Xo[n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_scopy(N,&Xi[tN],1,I,1); cblas_scopy(N,&Xf[tN],1,F,1); cblas_scopy(N,&Xo[tN],1,O,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Ui,N,C,1,o,I,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uf,N,C,1,o,F,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uo,N,C,1,o,O,1);
                for (n=0; n<N; n++)
                {
                    I[n] = 1.0f / (1.0f+expf(-I[n]));
                    C[n] = C[n]/(1.0f+expf(-F[n])) + I[n]/(1.0f+expf(-Xc[tN+n]));
                    Xc[tN+n] = C[n] / (1.0f+expf(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                I[n] = 1.0f / (1.0f+expf(-Xi[nT]));
                C[n] = I[n] / (1.0f+expf(-Xc[nT]));
                Xc[nT] = C[n] / (1.0f+expf(-Xo[nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&Xi[t],T,I,1); cblas_scopy(N,&Xf[t],T,F,1); cblas_scopy(N,&Xo[t],T,O,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Ui,N,C,1,o,I,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uf,N,C,1,o,F,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uo,N,C,1,o,O,1);
                for (n=0; n<N; n++)
                {
                    I[n] = 1.0f / (1.0f+expf(-I[n]));
                    C[n] = C[n]/(1.0f+expf(-F[n])) + I[n]/(1.0f+expf(-Xc[t+n*T]));
                    Xc[t+n*T] = C[n] / (1.0f+expf(-O[n]));
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
                I[n] = 1.0f / (1.0f+expf(-Xi[nT]));
                C[n] = I[n] / (1.0f+expf(-Xc[nT]));
                Xc[nT] = C[n] / (1.0f+expf(-Xo[nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&Xi[t],T,I,1); cblas_scopy(N,&Xf[t],T,F,1); cblas_scopy(N,&Xo[t],T,O,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Ui,N,C,1,o,I,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Uf,N,C,1,o,F,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Uo,N,C,1,o,O,1);
                for (n=0; n<N; n++)
                {
                    I[n] = 1.0f / (1.0f+expf(-I[n]));
                    C[n] = C[n]/(1.0f+expf(-F[n])) + I[n]/(1.0f+expf(-Xc[t+n*T]));
                    Xc[t+n*T] = C[n] / (1.0f+expf(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                I[n] = 1.0f / (1.0f+expf(-Xi[n]));
                C[n] = I[n] / (1.0f+expf(-Xc[n]));
                Xc[n] = C[n] / (1.0f+expf(-Xo[n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_scopy(N,&Xi[tN],1,I,1); cblas_scopy(N,&Xf[tN],1,F,1); cblas_scopy(N,&Xo[tN],1,O,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Ui,N,C,1,o,I,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Uf,N,C,1,o,F,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Uo,N,C,1,o,O,1);
                for (n=0; n<N; n++)
                {
                    I[n] = 1.0f / (1.0f+expf(-I[n]));
                    C[n] = C[n]/(1.0f+expf(-F[n])) + I[n]/(1.0f+expf(-Xc[tN+n]));
                    Xc[tN+n] = C[n] / (1.0f+expf(-O[n]));
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in lstm_peephole4_inplace_s: dim must be 0 or 1.\n"); return 1;
    }

    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);
    return 0;
}


int lstm_peephole4_inplace_d (double *Xc, const double *Xi, const double *Xf, const double *Xo, const double *Ui, const double *Uf, const double *Uo, const int N, const int T, const int dim, const char iscolmajor)
{
    const double o = 1.0;
    int n, t, nT, tN;
    double *C, *I, *F, *O;

    //Checks
    if (N<1) { fprintf(stderr,"error in lstm_peephole4_inplace_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in lstm_peephole4_inplace_d: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(C=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm_peephole4_inplace_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(I=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm_peephole4_inplace_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(F=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm_peephole4_inplace_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(O=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm_peephole4_inplace_d: problem with malloc. "); perror("malloc"); return 1; }

    if (N==1)
    {
        I[0] = 1.0 / (1.0+exp(-Xi[0]));
        C[0] = I[0] / (1.0+exp(-Xc[0]));
        Xc[0] = C[0] / (1.0+exp(-Xo[0]));
        for (t=1; t<T; t++)
        {
            I[0] = 1.0 / (1.0+exp(-Xi[t]-Ui[0]*C[0]));
            O[0] = 1.0 / (1.0+exp(-Xo[t]-Uo[0]*C[0]));
            C[0] = C[0]/(1.0+exp(-Xf[t]-Uf[0]*C[0])) + I[0]/(1.0+exp(-Xc[t]));
            Xc[t] = O[0] * C[0];
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                I[n] = 1.0 / (1.0+exp(-Xi[n]));
                C[n] = I[n] / (1.0+exp(-Xc[n]));
                Xc[n] = C[n] / (1.0+exp(-Xo[n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_dcopy(N,&Xi[tN],1,I,1); cblas_dcopy(N,&Xf[tN],1,F,1); cblas_dcopy(N,&Xo[tN],1,O,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Ui,N,C,1,o,I,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Uf,N,C,1,o,F,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Uo,N,C,1,o,O,1);
                for (n=0; n<N; n++)
                {
                    I[n] = 1.0 / (1.0+exp(-I[n]));
                    C[n] = C[n]/(1.0+exp(-F[n])) + I[n]/(1.0+exp(-Xc[tN+n]));
                    Xc[tN+n] = C[n] / (1.0+exp(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                I[n] = 1.0 / (1.0+exp(-Xi[nT]));
                C[n] = I[n] / (1.0+exp(-Xc[nT]));
                Xc[nT] = C[n] / (1.0+exp(-Xo[nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&Xi[t],T,I,1); cblas_dcopy(N,&Xf[t],T,F,1); cblas_dcopy(N,&Xo[t],T,O,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Ui,N,C,1,o,I,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uf,N,C,1,o,F,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uo,N,C,1,o,O,1);
                for (n=0; n<N; n++)
                {
                    I[n] = 1.0 / (1.0+exp(-I[n]));
                    C[n] = C[n]/(1.0+exp(-F[n])) + I[n]/(1.0+exp(-Xc[t+n*T]));
                    Xc[t+n*T] = C[n] / (1.0+exp(-O[n]));
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
                I[n] = 1.0 / (1.0+exp(-Xi[nT]));
                C[n] = I[n] / (1.0+exp(-Xc[nT]));
                Xc[nT] = C[n] / (1.0+exp(-Xo[nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&Xi[t],T,I,1); cblas_dcopy(N,&Xf[t],T,F,1); cblas_dcopy(N,&Xo[t],T,O,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Ui,N,C,1,o,I,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Uf,N,C,1,o,F,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Uo,N,C,1,o,O,1);
                for (n=0; n<N; n++)
                {
                    I[n] = 1.0 / (1.0+exp(-I[n]));
                    C[n] = C[n]/(1.0+exp(-F[n])) + I[n]/(1.0+exp(-Xc[t+n*T]));
                    Xc[t+n*T] = C[n] / (1.0+exp(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                I[n] = 1.0 / (1.0+exp(-Xi[n]));
                C[n] = I[n] / (1.0+exp(-Xc[n]));
                Xc[n] = C[n] / (1.0+exp(-Xo[n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_dcopy(N,&Xi[tN],1,I,1); cblas_dcopy(N,&Xf[tN],1,F,1); cblas_dcopy(N,&Xo[tN],1,O,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Ui,N,C,1,o,I,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Uf,N,C,1,o,F,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Uo,N,C,1,o,O,1);
                for (n=0; n<N; n++)
                {
                    I[n] = 1.0 / (1.0+exp(-I[n]));
                    C[n] = C[n]/(1.0+exp(-F[n])) + I[n]/(1.0+exp(-Xc[tN+n]));
                    Xc[tN+n] = C[n] / (1.0+exp(-O[n]));
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in lstm_peephole4_inplace_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
