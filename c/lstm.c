//This does CELL (~soma) stage of LSTM (long short-term memory) model.
//This requires each neuron to have 4 input time series, Xc, Xi, Xf, Xo,
//where Xc is the usual (or "cellular") input and Xi, Xf, Xo the inputs for the input, forget, output gates.
//Xc, Xi, Xf, Xo are the output of separate linear IN stages (weights and baises).
//In this version, these are stacked into one matrix
//For dim=0: X = [Xc; Xi; Xf; Xo]; for dim=1: X = [Xc Xi Xf Xo].

//For dim=0, C[:,t] = tanh{Xc[:,t] + Uc*Y[:,t-1]} \n";
//           I[:,t] = sig{Xi[:,t] + Ui*Y[:,t-1]} \n";
//           F[:,t] = sig{Xf[:,t] + Uf*Y[:,t-1]} \n";
//           O[:,t] = sig{Xo[:,t] + Uo*Y[:,t-1]} \n";
//           H[:,t] = I[:,t].*C[:,t] + F[:,t].*H[:,t-1] \n";
//           Y[:,t] = O[:,t].*tanh{H[:,t]} \n";
//with sizes Xc, Xi, Xf, Xo: N x T \n";
//           Uc, Ui, Uf, Uo: N x N \n";
//           Y             : N x T \n";
//
//For dim=1, C[t,:] = tanh{Xc[t,:] + Y[t-1,:]*Uc} \n";
//           I[t,:] = sig{Xi[t,:] + Y[t-1,:]*Ui} \n";
//           F[t,:] = sig{Xf[t,:] + Y[t-1,:]*Uf} \n";
//           O[t,:] = sig{Xo[t,:] + Y[t-1,:]*Uo} \n";
//           H[t,:] = I[t,:].*C[t,:] + F[t,:].*H[t-1,:] \n";
//           Y[t,:] = O[t,:].*tanh{H[t,:]} \n";
//with sizes Xc, Xi, Xf, Xo: T x N \n";
//           Uc, Ui, Uf, Uo: N x N \n";
//           Y             : T x N \n";
//
//where sig is the logistic (sigmoid) nonlinearity = 1/(1+exp(-x)),
//I is the input gate, F is the forget gate, O is the output gate,
//C is the "cell input activation vector",
//H is an intermediate (hidden) vector (sometimes called the "cell state vector"),
//Uc, Ui, Uf, Uo are NxN matrices, and Y is the final output (sometimes called the "hidden state vector").

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

int lstm_s (float *Y, const float *X, const float *Uc, const float *Ui, const float *Uf, const float *Uo, const int N, const int T, const int dim, const char iscolmajor);
int lstm_d (double *Y, const double *X, const double *Uc, const double *Ui, const double *Uf, const double *Uo, const int N, const int T, const int dim, const char iscolmajor);

int lstm_inplace_s (float *X, const float *Uc, const float *Ui, const float *Uf, const float *Uo, const int N, const int T, const int dim, const char iscolmajor);
int lstm_inplace_d (double *X, const double *Uc, const double *Ui, const double *Uf, const double *Uo, const int N, const int T, const int dim, const char iscolmajor);


int lstm_s (float *Y, const float *X, const float *Uc, const float *Ui, const float *Uf, const float *Uo, const int N, const int T, const int dim, const char iscolmajor)
{
    const float o = 1.0f;
    const int N2 = 2*N, N3 = 3*N, NT = N*T, NT2 = 2*N*T, NT3 = 3*N*T;
    int n, t, nT, tN, tN4;
    float *C, *I, *F, *O, *H;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in lstm_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in lstm_s: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(C=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(I=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(F=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(O=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm_s: problem with malloc. "); perror("malloc"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                H[n] = tanhf(X[n]) / (1.0f+expf(-X[N+n]));
                Y[n] = tanhf(H[n]) / (1.0f+expf(-X[N3+n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N; tN4 = 4*tN;
                cblas_scopy(N,&X[tN4],1,C,1); cblas_scopy(N,&X[tN4+N],1,I,1);
                cblas_scopy(N,&X[tN4+N2],1,F,1); cblas_scopy(N,&X[tN4+N3],1,O,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uc,N,&Y[tN-N],1,o,C,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Ui,N,&Y[tN-N],1,o,I,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uf,N,&Y[tN-N],1,o,F,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uo,N,&Y[tN-N],1,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanhf(C[n])/(1.0f+expf(-I[n])) + H[n]/(1.0f+expf(-F[n]));
                    Y[tN+n] = tanhf(H[n]) / (1.0f+expf(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                H[n] = tanhf(X[nT]) / (1.0f+expf(-X[NT+nT]));
                Y[nT] = tanhf(H[n]) / (1.0f+expf(-X[NT3+nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&X[t],T,C,1); cblas_scopy(N,&X[NT+t],T,I,1);
                cblas_scopy(N,&X[NT2+t],T,F,1); cblas_scopy(N,&X[NT3+t],T,O,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uc,N,&Y[t-1],T,o,C,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Ui,N,&Y[t-1],T,o,I,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uf,N,&Y[t-1],T,o,F,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uo,N,&Y[t-1],T,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanhf(C[n])/(1.0f+expf(-I[n])) + H[n]/(1.0f+expf(-F[n]));
                    Y[t+n*T] = tanhf(H[n]) / (1.0f+expf(-O[n]));
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
                H[n] = tanhf(X[nT]) / (1.0f+expf(-X[NT+nT]));
                Y[nT] = tanhf(H[n]) / (1.0f+expf(-X[NT3+nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&X[t],T,C,1); cblas_scopy(N,&X[NT+t],T,I,1);
                cblas_scopy(N,&X[NT2+t],T,F,1); cblas_scopy(N,&X[NT3+t],T,O,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Uc,N,&Y[t-1],T,o,C,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Ui,N,&Y[t-1],T,o,I,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Uf,N,&Y[t-1],T,o,F,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Uo,N,&Y[t-1],T,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanhf(C[n])/(1.0f+expf(-I[n])) + H[n]/(1.0f+expf(-F[n]));
                    Y[t+n*T] = tanhf(H[n]) / (1.0f+expf(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                H[n] = tanhf(X[n]) / (1.0f+expf(-X[N+n]));
                Y[n] = tanhf(H[n]) / (1.0f+expf(-X[N3+n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N; tN4 = 4*tN;
                cblas_scopy(N,&X[tN4],1,C,1); cblas_scopy(N,&X[tN4+N],1,I,1);
                cblas_scopy(N,&X[tN4+N2],1,F,1); cblas_scopy(N,&X[tN4+N3],1,O,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Uc,N,&Y[tN-N],1,o,C,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Ui,N,&Y[tN-N],1,o,I,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Uf,N,&Y[tN-N],1,o,F,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Uo,N,&Y[tN-N],1,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanhf(C[n])/(1.0f+expf(-I[n])) + H[n]/(1.0f+expf(-F[n]));
                    Y[tN+n] = tanhf(H[n]) / (1.0f+expf(-O[n]));
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in lstm_s: dim must be 0 or 1.\n"); return 1;
    }

    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);
    return 0;
}


int lstm_d (double *Y, const double *X, const double *Uc, const double *Ui, const double *Uf, const double *Uo, const int N, const int T, const int dim, const char iscolmajor)
{
    const double o = 1.0;
    const int N2 = 2*N, N3 = 3*N, NT = N*T, NT2 = 2*N*T, NT3 = 3*N*T;
    int n, t, nT, tN, tN4;
    double *C, *I, *F, *O, *H;

    //Checks
    if (N<1) { fprintf(stderr,"error in lstm_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in lstm_d: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(C=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(I=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(F=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(O=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm_d: problem with malloc. "); perror("malloc"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                H[n] = tanh(X[n]) / (1.0+exp(-X[N+n]));
                Y[n] = tanh(H[n]) / (1.0+exp(-X[N3+n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N; tN4 = 4*tN;
                cblas_dcopy(N,&X[tN4],1,C,1); cblas_dcopy(N,&X[tN4+N],1,I,1);
                cblas_dcopy(N,&X[tN4+N2],1,F,1); cblas_dcopy(N,&X[tN4+N3],1,O,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Uc,N,&Y[tN-N],1,o,C,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Ui,N,&Y[tN-N],1,o,I,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Uf,N,&Y[tN-N],1,o,F,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Uo,N,&Y[tN-N],1,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanh(C[n])/(1.0+exp(-I[n])) + H[n]/(1.0+exp(-F[n]));
                    Y[tN+n] = tanh(H[n]) / (1.0+exp(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                H[n] = tanh(X[nT]) / (1.0+exp(-X[NT+nT]));
                Y[nT] = tanh(H[n]) / (1.0+exp(-X[NT3+nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&X[t],T,C,1); cblas_dcopy(N,&X[NT+t],T,I,1);
                cblas_dcopy(N,&X[NT2+t],T,F,1); cblas_dcopy(N,&X[NT3+t],T,O,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uc,N,&Y[t-1],T,o,C,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Ui,N,&Y[t-1],T,o,I,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uf,N,&Y[t-1],T,o,F,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uo,N,&Y[t-1],T,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanh(C[n])/(1.0+exp(-I[n])) + H[n]/(1.0+exp(-F[n]));
                    Y[t+n*T] = tanh(H[n]) / (1.0+exp(-O[n]));
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
                H[n] = tanh(X[nT]) / (1.0+exp(-X[NT+nT]));
                Y[nT] = tanh(H[n]) / (1.0+exp(-X[NT3+nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&X[t],T,C,1); cblas_dcopy(N,&X[NT+t],T,I,1);
                cblas_dcopy(N,&X[NT2+t],T,F,1); cblas_dcopy(N,&X[NT3+t],T,O,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Uc,N,&Y[t-1],T,o,C,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Ui,N,&Y[t-1],T,o,I,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Uf,N,&Y[t-1],T,o,F,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Uo,N,&Y[t-1],T,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanh(C[n])/(1.0+exp(-I[n])) + H[n]/(1.0+exp(-F[n]));
                    Y[t+n*T] = tanh(H[n]) / (1.0+exp(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                H[n] = tanh(X[n]) / (1.0+exp(-X[N+n]));
                Y[n] = tanh(H[n]) / (1.0+exp(-X[N3+n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N; tN4 = 4*tN;
                cblas_dcopy(N,&X[tN4],1,C,1); cblas_dcopy(N,&X[tN4+N],1,I,1);
                cblas_dcopy(N,&X[tN4+N2],1,F,1); cblas_dcopy(N,&X[tN4+N3],1,O,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Uc,N,&Y[tN-N],1,o,C,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Ui,N,&Y[tN-N],1,o,I,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Uf,N,&Y[tN-N],1,o,F,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Uo,N,&Y[tN-N],1,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanh(C[n])/(1.0+exp(-I[n])) + H[n]/(1.0+exp(-F[n]));
                    Y[tN+n] = tanh(H[n]) / (1.0+exp(-O[n]));
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in lstm_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}



int lstm_inplace_s (float *X, const float *Uc, const float *Ui, const float *Uf, const float *Uo, const int N, const int T, const int dim, const char iscolmajor)
{
    const float o = 1.0f;
    const int N2 = 2*N, N3 = 3*N, N4 = 4*N, NT = N*T, NT2 = 2*N*T, NT3 = 3*N*T;
    int n, t, nT, tN, tN4;
    float *C, *I, *F, *O, *H;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in lstm_inplace_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in lstm_inplace_s: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(C=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm_inplace_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(I=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm_inplace_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(F=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm_inplace_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(O=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm_inplace_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm_inplace_s: problem with malloc. "); perror("malloc"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                H[n] = tanhf(X[n]) / (1.0f+expf(-X[N+n]));
                X[n] = tanhf(H[n]) / (1.0f+expf(-X[N3+n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N; tN4 = 4*tN;
                cblas_scopy(N,&X[tN4],1,C,1); cblas_scopy(N,&X[tN4+N],1,I,1);
                cblas_scopy(N,&X[tN4+N2],1,F,1); cblas_scopy(N,&X[tN4+N3],1,O,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uc,N,&X[tN4-N4],1,o,C,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Ui,N,&X[tN4-N4],1,o,I,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uf,N,&X[tN4-N4],1,o,F,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uo,N,&X[tN4-N4],1,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanhf(C[n])/(1.0f+expf(-I[n])) + H[n]/(1.0f+expf(-F[n]));
                    X[tN4+n] = tanhf(H[n]) / (1.0f+expf(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                H[n] = tanhf(X[nT]) / (1.0f+expf(-X[NT+nT]));
                X[nT] = tanhf(H[n]) / (1.0f+expf(-X[NT3+nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&X[t],T,C,1); cblas_scopy(N,&X[NT+t],T,I,1);
                cblas_scopy(N,&X[NT2+t],T,F,1); cblas_scopy(N,&X[NT3+t],T,O,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uc,N,&X[t-1],T,o,C,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Ui,N,&X[t-1],T,o,I,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uf,N,&X[t-1],T,o,F,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uo,N,&X[t-1],T,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanhf(C[n])/(1.0f+expf(-I[n])) + H[n]/(1.0f+expf(-F[n]));
                    X[t+n*T] = tanhf(H[n]) / (1.0f+expf(-O[n]));
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
                H[n] = tanhf(X[nT]) / (1.0f+expf(-X[NT+nT]));
                X[nT] = tanhf(H[n]) / (1.0f+expf(-X[NT3+nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&X[t],T,C,1); cblas_scopy(N,&X[NT+t],T,I,1);
                cblas_scopy(N,&X[NT2+t],T,F,1); cblas_scopy(N,&X[NT3+t],T,O,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Uc,N,&X[t-1],T,o,C,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Ui,N,&X[t-1],T,o,I,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Uf,N,&X[t-1],T,o,F,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Uo,N,&X[t-1],T,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanhf(C[n])/(1.0f+expf(-I[n])) + H[n]/(1.0f+expf(-F[n]));
                    X[t+n*T] = tanhf(H[n]) / (1.0f+expf(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                H[n] = tanhf(X[n]) / (1.0f+expf(-X[N+n]));
                X[n] = tanhf(H[n]) / (1.0f+expf(-X[N3+n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N; tN4 = 4*tN;
                cblas_scopy(N,&X[tN4],1,C,1); cblas_scopy(N,&X[tN4+N],1,I,1);
                cblas_scopy(N,&X[tN4+N2],1,F,1); cblas_scopy(N,&X[tN4+N3],1,O,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Uc,N,&X[tN4-N4],1,o,C,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Ui,N,&X[tN4-N4],1,o,I,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Uf,N,&X[tN4-N4],1,o,F,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Uo,N,&X[tN4-N4],1,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanhf(C[n])/(1.0f+expf(-I[n])) + H[n]/(1.0f+expf(-F[n]));
                    X[tN4+n] = tanhf(H[n]) / (1.0f+expf(-O[n]));
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in lstm_inplace_s: dim must be 0 or 1.\n"); return 1;
    }

    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);
    return 0;
}


int lstm_inplace_d (double *X, const double *Uc, const double *Ui, const double *Uf, const double *Uo, const int N, const int T, const int dim, const char iscolmajor)
{
    const double o = 1.0;
    const int N2 = 2*N, N3 = 3*N, N4 = 4*N, NT = N*T, NT2 = 2*N*T, NT3 = 3*N*T;
    int n, t, nT, tN, tN4;
    double *C, *I, *F, *O, *H;

    //Checks
    if (N<1) { fprintf(stderr,"error in lstm_inplace_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in lstm_inplace_d: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(C=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm_inplace_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(I=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm_inplace_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(F=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm_inplace_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(O=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm_inplace_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm_inplace_d: problem with malloc. "); perror("malloc"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                H[n] = tanh(X[n]) / (1.0+exp(-X[N+n]));
                X[n] = tanh(H[n]) / (1.0+exp(-X[N3+n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N; tN4 = 4*tN;
                cblas_dcopy(N,&X[tN4],1,C,1); cblas_dcopy(N,&X[tN4+N],1,I,1);
                cblas_dcopy(N,&X[tN4+N2],1,F,1); cblas_dcopy(N,&X[tN4+N3],1,O,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Uc,N,&X[tN4-N4],1,o,C,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Ui,N,&X[tN4-N4],1,o,I,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Uf,N,&X[tN4-N4],1,o,F,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Uo,N,&X[tN4-N4],1,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanh(C[n])/(1.0+exp(-I[n])) + H[n]/(1.0+exp(-F[n]));
                    X[tN4+n] = tanh(H[n]) / (1.0+exp(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                H[n] = tanh(X[nT]) / (1.0+exp(-X[NT+nT]));
                X[nT] = tanh(H[n]) / (1.0+exp(-X[NT3+nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&X[t],T,C,1); cblas_dcopy(N,&X[NT+t],T,I,1);
                cblas_dcopy(N,&X[NT2+t],T,F,1); cblas_dcopy(N,&X[NT3+t],T,O,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uc,N,&X[t-1],T,o,C,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Ui,N,&X[t-1],T,o,I,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uf,N,&X[t-1],T,o,F,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uo,N,&X[t-1],T,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanh(C[n])/(1.0+exp(-I[n])) + H[n]/(1.0+exp(-F[n]));
                    X[t+n*T] = tanh(H[n]) / (1.0+exp(-O[n]));
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
                H[n] = tanh(X[nT]) / (1.0+exp(-X[NT+nT]));
                X[nT] = tanh(H[n]) / (1.0+exp(-X[NT3+nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&X[t],T,C,1); cblas_dcopy(N,&X[NT+t],T,I,1);
                cblas_dcopy(N,&X[NT2+t],T,F,1); cblas_dcopy(N,&X[NT3+t],T,O,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Uc,N,&X[t-1],T,o,C,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Ui,N,&X[t-1],T,o,I,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Uf,N,&X[t-1],T,o,F,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Uo,N,&X[t-1],T,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanh(C[n])/(1.0+exp(-I[n])) + H[n]/(1.0+exp(-F[n]));
                    X[t+n*T] = tanh(H[n]) / (1.0+exp(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                H[n] = tanh(X[n]) / (1.0+exp(-X[N+n]));
                X[n] = tanh(H[n]) / (1.0+exp(-X[N3+n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N; tN4 = 4*tN;
                cblas_dcopy(N,&X[tN4],1,C,1); cblas_dcopy(N,&X[tN4+N],1,I,1);
                cblas_dcopy(N,&X[tN4+N2],1,F,1); cblas_dcopy(N,&X[tN4+N3],1,O,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Uc,N,&X[tN4-N4],1,o,C,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Ui,N,&X[tN4-N4],1,o,I,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Uf,N,&X[tN4-N4],1,o,F,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Uo,N,&X[tN4-N4],1,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanh(C[n])/(1.0+exp(-I[n])) + H[n]/(1.0+exp(-F[n]));
                    X[tN4+n] = tanh(H[n]) / (1.0+exp(-O[n]));
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in lstm_inplace_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
