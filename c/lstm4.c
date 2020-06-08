//This does CELL (~soma) stage of LSTM (long short-term memory) model.
//This requires each neuron to have 4 input time series, Xc, Xi, Xf, Xo,
//where Xc is the usual (or "cellular") input and Xi, Xf, Xo the inputs for the input, forget, output gates.
//Xc, Xi, Xf, Xo are the output of separate linear IN stages (weights and baises).

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

int lstm4_s (float *Y, const float *Xc, const float *Xi, const float *Xf, const float *Xo, const float *Uc, const float *Ui, const float *Uf, const float *Uo, const int N, const int T, const int dim, const char iscolmajor);
int lstm4_d (double *Y, const double *Xc, const double *Xi, const double *Xf, const double *Xo, const double *Uc, const double *Ui, const double *Uf, const double *Uo, const int N, const int T, const int dim, const char iscolmajor);

int lstm4_inplace_s (float *Xc, const float *Xi, const float *Xf, const float *Xo, const float *Uc, const float *Ui, const float *Uf, const float *Uo, const int N, const int T, const int dim, const char iscolmajor);
int lstm4_inplace_d (double *Xc, const double *Xi, const double *Xf, const double *Xo, const double *Uc, const double *Ui, const double *Uf, const double *Uo, const int N, const int T, const int dim, const char iscolmajor);


int lstm4_s (float *Y, const float *Xc, const float *Xi, const float *Xf, const float *Xo, const float *Uc, const float *Ui, const float *Uf, const float *Uo, const int N, const int T, const int dim, const char iscolmajor)
{
    const float o = 1.0f;
    int n, t, nT, tN;
    float *C, *I, *F, *O, *H;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in lstm4_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in lstm4_s: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(C=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm4_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(I=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm4_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(F=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm4_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(O=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm4_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm4_s: problem with malloc. "); perror("malloc"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (N==1)
    {
        //C[0] = tanhf(Xc[0]);
        //I[0] = 1.0f / (1.0f+expf(-Xi[0]));
        //F[0] = 1.0f / (1.0f+expf(-Xf[0]));
        //O[0] = 1.0f / (1.0f+expf(-Xo[0]));
        //H[0] = I[0] * C[0];
        //Y[0] = O[0] * tanhf(H[0]);
        H[0] = tanhf(Xc[0]) / (1.0f+expf(-Xi[0]));
        Y[0] = tanhf(H[0]) / (1.0f+expf(-Xo[0]));
        for (t=1; t<T; t++)
        {
            //C[0] = tanhf(Xc[t]+Uc[0]*Y[t-1]);
            //I[0] = 1.0f / (1.0f+expf(-Xi[t]-Ui[0]*Y[t-1]));
            //F[0] = 1.0f / (1.0f+expf(-Xf[t]-Uf[0]*Y[t-1]));
            //O[0] = 1.0f / (1.0f+expf(-Xo[t]-Uo[0]*Y[t-1]));
            //H[0] = I[0]*C[0] + F[0]*H[0];
            //Y[t] = O[0] * tanhf(H[0]);
            H[0] = tanhf(Xc[t]+Uc[0]*Y[t-1])/(1.0f+expf(-Xi[t]-Ui[0]*Y[t-1])) + H[0]/(1.0f+expf(-Xf[t]-Uf[0]*Y[t-1]));
            Y[t] = tanhf(H[0]) / (1.0f+expf(-Xo[t]-Uo[0]*Y[t-1]));
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                //C[n] = tanhf(Xc[n]);
                //I[n] = 1.0f / (1.0f+expf(-Xi[n]));
                //F[n] = 1.0f / (1.0f+expf(-Xf[n]));
                //O[n] = 1.0f / (1.0f+expf(-Xo[n]));
                //H[n] = I[n] * C[n];
                //Y[n] = O[n] * tanhf(H[n]);
                H[n] = tanhf(Xc[n]) / (1.0f+expf(-Xi[n]));
                Y[n] = tanhf(H[n]) / (1.0f+expf(-Xo[n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_scopy(N,&Xc[tN],1,C,1); cblas_scopy(N,&Xi[tN],1,I,1);
                cblas_scopy(N,&Xf[tN],1,F,1); cblas_scopy(N,&Xo[tN],1,O,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uc,N,&Y[tN-N],1,o,C,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Ui,N,&Y[tN-N],1,o,I,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uf,N,&Y[tN-N],1,o,F,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uo,N,&Y[tN-N],1,o,O,1);
                for (n=0; n<N; n++)
                {
                    //C[n] = tanhf(C[n]);
                    //I[n] = 1.0f / (1.0f+expf(-I[n]));
                    //F[n] = 1.0f / (1.0f+expf(-F[n]));
                    //O[n] = 1.0f / (1.0f+expf(-O[n]));
                    //H[n] = I[n]*C[n] + F[n]*H[n];
                    //Y[tN+n] = O[n] * tanhf(H[n]);
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
                //C[n] = tanhf(Xc[nT])
                //I[n] = 1.0f / (1.0f+expf(-Xi[nT]));
                //F[n] = 1.0f / (1.0f+expf(-Xf[nT]));
                //O[n] = 1.0f / (1.0f+expf(-Xo[nT]));
                //H[n] = I[n] * C[n];
                //Y[nT] = O[n] * tanhf(H[n]);
                H[n] = tanhf(Xc[nT]) / (1.0f+expf(-Xi[nT]));
                Y[nT] = tanhf(H[n]) / (1.0f+expf(-Xo[nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&Xc[t],T,C,1); cblas_scopy(N,&Xi[t],T,I,1);
                cblas_scopy(N,&Xf[t],T,F,1); cblas_scopy(N,&Xo[t],T,O,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uc,N,&Y[t-1],T,o,C,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Ui,N,&Y[t-1],T,o,I,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uf,N,&Y[t-1],T,o,F,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uo,N,&Y[t-1],T,o,O,1);
                for (n=0; n<N; n++)
                {
                    //C[n] = tanhf(C[n]);
                    //I[n] = 1.0f / (1.0f+expf(-I[n]));
                    //F[n] = 1.0f / (1.0f+expf(-F[n]));
                    //O[n] = 1.0f / (1.0f+expf(-O[n]));
                    //H[n] = I[n]*C[n] + F[n]*H[n];
                    //Y[t+n*T] = O[n] * tanhf(H[n]);
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
                H[n] = tanhf(Xc[nT]) / (1.0f+expf(-Xi[nT]));
                Y[nT] = tanhf(H[n]) / (1.0f+expf(-Xo[nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&Xc[t],T,C,1); cblas_scopy(N,&Xi[t],T,I,1);
                cblas_scopy(N,&Xf[t],T,F,1); cblas_scopy(N,&Xo[t],T,O,1);
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
                H[n] = tanhf(Xc[n]) / (1.0f+expf(-Xi[n]));
                Y[n] = tanhf(H[n]) / (1.0f+expf(-Xo[n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_scopy(N,&Xc[tN],1,C,1); cblas_scopy(N,&Xi[tN],1,I,1);
                cblas_scopy(N,&Xf[tN],1,F,1); cblas_scopy(N,&Xo[tN],1,O,1);
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
        fprintf(stderr,"error in lstm4_s: dim must be 0 or 1.\n"); return 1;
    }

    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);
    return 0;
}


int lstm4_d (double *Y, const double *Xc, const double *Xi, const double *Xf, const double *Xo, const double *Uc, const double *Ui, const double *Uf, const double *Uo, const int N, const int T, const int dim, const char iscolmajor)
{
    const double o = 1.0;
    int n, t, nT, tN;
    double *C, *I, *F, *O, *H;

    //Checks
    if (N<1) { fprintf(stderr,"error in lstm4_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in lstm4_d: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(C=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm4_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(I=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm4_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(F=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm4_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(O=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm4_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm4_d: problem with malloc. "); perror("malloc"); return 1; }

    if (N==1)
    {
        H[0] = tanh(Xc[0]) / (1.0+exp(-Xi[0]));
        Y[0] = tanh(H[0]) / (1.0+exp(-Xo[0]));
        for (t=1; t<T; t++)
        {
            H[0] = tanh(Xc[t]+Uc[0]*Y[t-1])/(1.0+exp(-Xi[t]-Ui[0]*Y[t-1])) + H[0]/(1.0+exp(-Xf[t]-Uf[0]*Y[t-1]));
            Y[t] = tanh(H[0]) / (1.0+exp(-Xo[t]-Uo[0]*Y[t-1]));
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                H[n] = tanh(Xc[n]) / (1.0+exp(-Xi[n]));
                Y[n] = tanh(H[n]) / (1.0+exp(-Xo[n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_dcopy(N,&Xc[tN],1,C,1); cblas_dcopy(N,&Xi[tN],1,I,1);
                cblas_dcopy(N,&Xf[tN],1,F,1); cblas_dcopy(N,&Xo[tN],1,O,1);
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
                H[n] = tanh(Xc[nT]) / (1.0+exp(-Xi[nT]));
                Y[nT] = tanh(H[n]) / (1.0+exp(-Xo[nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&Xc[t],T,C,1); cblas_dcopy(N,&Xi[t],T,I,1);
                cblas_dcopy(N,&Xf[t],T,F,1); cblas_dcopy(N,&Xo[t],T,O,1);
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
                H[n] = tanh(Xc[nT]) / (1.0+exp(-Xi[nT]));
                Y[nT] = tanh(H[n]) / (1.0+exp(-Xo[nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&Xc[t],T,C,1); cblas_dcopy(N,&Xi[t],T,I,1);
                cblas_dcopy(N,&Xf[t],T,F,1); cblas_dcopy(N,&Xo[t],T,O,1);
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
                H[n] = tanh(Xc[n]) / (1.0+exp(-Xi[n]));
                Y[n] = tanh(H[n]) / (1.0+exp(-Xo[n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_dcopy(N,&Xc[tN],1,C,1); cblas_dcopy(N,&Xi[tN],1,I,1);
                cblas_dcopy(N,&Xf[tN],1,F,1); cblas_dcopy(N,&Xo[tN],1,O,1);
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
        fprintf(stderr,"error in lstm4_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}



int lstm4_inplace_s (float *Xc, const float *Xi, const float *Xf, const float *Xo, const float *Uc, const float *Ui, const float *Uf, const float *Uo, const int N, const int T, const int dim, const char iscolmajor)
{
    const float o = 1.0f;
    int n, t, nT, tN;
    float *C, *I, *F, *O, *H;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in lstm4_inplace_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in lstm4_inplace_s: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(C=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm4_inplace_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(I=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm4_inplace_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(F=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm4_inplace_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(O=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm4_inplace_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in lstm4_inplace_s: problem with malloc. "); perror("malloc"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (N==1)
    {
        H[0] = tanhf(Xc[0]) / (1.0f+expf(-Xi[0]));
        Xc[0] = tanhf(H[0]) / (1.0f+expf(-Xo[0]));
        for (t=1; t<T; t++)
        {
            H[0] = tanhf(Xc[t]+Uc[0]*Xc[t-1])/(1.0f+expf(-Xi[t]-Ui[0]*Xc[t-1])) + H[0]/(1.0f+expf(-Xf[t]-Uf[0]*Xc[t-1]));
            Xc[t] = tanhf(H[0]) / (1.0f+expf(-Xo[t]-Uo[0]*Xc[t-1]));
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                H[n] = tanhf(Xc[n]) / (1.0f+expf(-Xi[n]));
                Xc[n] = tanhf(H[n]) / (1.0f+expf(-Xo[n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_scopy(N,&Xc[tN],1,C,1); cblas_scopy(N,&Xi[tN],1,I,1);
                cblas_scopy(N,&Xf[tN],1,F,1); cblas_scopy(N,&Xo[tN],1,O,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uc,N,&Xc[tN-N],1,o,C,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Ui,N,&Xc[tN-N],1,o,I,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uf,N,&Xc[tN-N],1,o,F,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,Uo,N,&Xc[tN-N],1,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanhf(C[n])/(1.0f+expf(-I[n])) + H[n]/(1.0f+expf(-F[n]));
                    Xc[tN+n] = tanhf(H[n]) / (1.0f+expf(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                H[n] = tanhf(Xc[nT]) / (1.0f+expf(-Xi[nT]));
                Xc[nT] = tanhf(H[n]) / (1.0f+expf(-Xo[nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&Xc[t],T,C,1); cblas_scopy(N,&Xi[t],T,I,1);
                cblas_scopy(N,&Xf[t],T,F,1); cblas_scopy(N,&Xo[t],T,O,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uc,N,&Xc[t-1],T,o,C,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Ui,N,&Xc[t-1],T,o,I,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uf,N,&Xc[t-1],T,o,F,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uo,N,&Xc[t-1],T,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanhf(C[n])/(1.0f+expf(-I[n])) + H[n]/(1.0f+expf(-F[n]));
                    Xc[t+n*T] = tanhf(H[n]) / (1.0f+expf(-O[n]));
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
                H[n] = tanhf(Xc[nT]) / (1.0f+expf(-Xi[nT]));
                Xc[nT] = tanhf(H[n]) / (1.0f+expf(-Xo[nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_scopy(N,&Xc[t],T,C,1); cblas_scopy(N,&Xi[t],T,I,1);
                cblas_scopy(N,&Xf[t],T,F,1); cblas_scopy(N,&Xo[t],T,O,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Uc,N,&Xc[t-1],T,o,C,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Ui,N,&Xc[t-1],T,o,I,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Uf,N,&Xc[t-1],T,o,F,1);
                cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,Uo,N,&Xc[t-1],T,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanhf(C[n])/(1.0f+expf(-I[n])) + H[n]/(1.0f+expf(-F[n]));
                    Xc[t+n*T] = tanhf(H[n]) / (1.0f+expf(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                H[n] = tanhf(Xc[n]) / (1.0f+expf(-Xi[n]));
                Xc[n] = tanhf(H[n]) / (1.0f+expf(-Xo[n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_scopy(N,&Xc[tN],1,C,1); cblas_scopy(N,&Xi[tN],1,I,1);
                cblas_scopy(N,&Xf[tN],1,F,1); cblas_scopy(N,&Xo[tN],1,O,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Uc,N,&Xc[tN-N],1,o,C,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Ui,N,&Xc[tN-N],1,o,I,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Uf,N,&Xc[tN-N],1,o,F,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,Uo,N,&Xc[tN-N],1,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanhf(C[n])/(1.0f+expf(-I[n])) + H[n]/(1.0f+expf(-F[n]));
                    Xc[tN+n] = tanhf(H[n]) / (1.0f+expf(-O[n]));
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in lstm4_inplace_s: dim must be 0 or 1.\n"); return 1;
    }

    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);
    return 0;
}


int lstm4_inplace_d (double *Xc, const double *Xi, const double *Xf, const double *Xo, const double *Uc, const double *Ui, const double *Uf, const double *Uo, const int N, const int T, const int dim, const char iscolmajor)
{
    const double o = 1.0;
    int n, t, nT, tN;
    double *C, *I, *F, *O, *H;

    //Checks
    if (N<1) { fprintf(stderr,"error in lstm4_inplace_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in lstm4_inplace_d: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(C=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm4_inplace_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(I=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm4_inplace_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(F=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm4_inplace_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(O=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm4_inplace_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in lstm4_inplace_d: problem with malloc. "); perror("malloc"); return 1; }

    if (N==1)
    {
        H[0] = tanh(Xc[0]) / (1.0+exp(-Xi[0]));
        Xc[0] = tanh(H[0]) / (1.0+exp(-Xo[0]));
        for (t=1; t<T; t++)
        {
            H[0] = tanh(Xc[t]+Uc[0]*Xc[t-1])/(1.0+exp(-Xi[t]-Ui[0]*Xc[t-1])) + H[0]/(1.0+exp(-Xf[t]-Uf[0]*Xc[t-1]));
            Xc[t] = tanh(H[0]) / (1.0+exp(-Xo[t]-Uo[0]*Xc[t-1]));
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                H[n] = tanh(Xc[n]) / (1.0+exp(-Xi[n]));
                Xc[n] = tanh(H[n]) / (1.0+exp(-Xo[n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_dcopy(N,&Xc[tN],1,C,1); cblas_dcopy(N,&Xi[tN],1,I,1);
                cblas_dcopy(N,&Xf[tN],1,F,1); cblas_dcopy(N,&Xo[tN],1,O,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Uc,N,&Xc[tN-N],1,o,C,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Ui,N,&Xc[tN-N],1,o,I,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Uf,N,&Xc[tN-N],1,o,F,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,Uo,N,&Xc[tN-N],1,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanh(C[n])/(1.0+exp(-I[n])) + H[n]/(1.0+exp(-F[n]));
                    Xc[tN+n] = tanh(H[n]) / (1.0+exp(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                H[n] = tanh(Xc[nT]) / (1.0+exp(-Xi[nT]));
                Xc[nT] = tanh(H[n]) / (1.0+exp(-Xo[nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&Xc[t],T,C,1); cblas_dcopy(N,&Xi[t],T,I,1);
                cblas_dcopy(N,&Xf[t],T,F,1); cblas_dcopy(N,&Xo[t],T,O,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uc,N,&Xc[t-1],T,o,C,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Ui,N,&Xc[t-1],T,o,I,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uf,N,&Xc[t-1],T,o,F,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,Uo,N,&Xc[t-1],T,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanh(C[n])/(1.0+exp(-I[n])) + H[n]/(1.0+exp(-F[n]));
                    Xc[t+n*T] = tanh(H[n]) / (1.0+exp(-O[n]));
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
                H[n] = tanh(Xc[nT]) / (1.0+exp(-Xi[nT]));
                Xc[nT] = tanh(H[n]) / (1.0+exp(-Xo[nT]));
            }
            for (t=1; t<T; t++)
            {
                cblas_dcopy(N,&Xc[t],T,C,1); cblas_dcopy(N,&Xi[t],T,I,1);
                cblas_dcopy(N,&Xf[t],T,F,1); cblas_dcopy(N,&Xo[t],T,O,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Uc,N,&Xc[t-1],T,o,C,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Ui,N,&Xc[t-1],T,o,I,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Uf,N,&Xc[t-1],T,o,F,1);
                cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,Uo,N,&Xc[t-1],T,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanh(C[n])/(1.0+exp(-I[n])) + H[n]/(1.0+exp(-F[n]));
                    Xc[t+n*T] = tanh(H[n]) / (1.0+exp(-O[n]));
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                H[n] = tanh(Xc[n]) / (1.0+exp(-Xi[n]));
                Xc[n] = tanh(H[n]) / (1.0+exp(-Xo[n]));
            }
            for (t=1; t<T; t++)
            {
                tN = t*N;
                cblas_dcopy(N,&Xc[tN],1,C,1); cblas_dcopy(N,&Xi[tN],1,I,1);
                cblas_dcopy(N,&Xf[tN],1,F,1); cblas_dcopy(N,&Xo[tN],1,O,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Uc,N,&Xc[tN-N],1,o,C,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Ui,N,&Xc[tN-N],1,o,I,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Uf,N,&Xc[tN-N],1,o,F,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,Uo,N,&Xc[tN-N],1,o,O,1);
                for (n=0; n<N; n++)
                {
                    H[n] = tanh(C[n])/(1.0+exp(-I[n])) + H[n]/(1.0+exp(-F[n]));
                    Xc[tN+n] = tanh(H[n]) / (1.0+exp(-O[n]));
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in lstm4_inplace_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
