//This does CELL (~soma) stage of GRU (gated recurrent unit) model.
//This requires each neuron to have 3 input time series, X, Xr, Xz,
//where X is the usual input and Xr, Xz the inputs for the reset and update gates,
//stacked into X = [X; Xr; Xz] for dim=0, or X = [X Xr Xz] for dim=1.

//For dim=0, R[:,t] = sig{Xr[:,t] + Ur*Y[:,t-1]} \n";
//           Z[:,t] = sig{Xz[:,t] + Uz*Y[:,t-1]} \n";
//           H[:,t] = R[:,t].*Y[:,t-1] \n";
//           Y[:,t] = Z[:,t].*Y[:,t-1] + (1-Z[:,t]).*tanh{X[:,t] + U*H[:,t]} \n";
//with sizes X, Xr, Xz: N x T \n";
//           U, Ur, Uz: N x N \n";
//           Y        : N x T \n";
//
//For dim=1, R[t,:] = sig{Xr[t,:] + Y[t-1,:]*Ur} \n";
//           Z[t,:] = sig{Xz[t,:] + Y[t-1,:]*Uz} \n";
//           H[t,:] = R[t,:].*Y[t-1,:] \n";
//           Y[t,:] = Z[t,:].*Y[t-1,:] + (1-Z[t,:]).*tanh{X[t,:] + H[t,:]*U} \n";
//with sizes X, Xr, Xz: T x N \n";
//           U, Ur, Uz: N x N \n";
//           Y        : T x N \n";
//
//where sig is the logistic (sigmoid) nonlinearity = 1/(1+exp(-x)),
//R is the reset gate, Z is the update gate, H is an intermediate (hidden) vector,
//U, Ur, Uz are NxN matrices, and Y is the output.

//Note that, the neurons of a layer are independent only if U, Ur, Uz are diagonal matrices.
//This is only really a CELL (~soma) stage in that case.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int gru_s (float *Y, const float *X, const float *U, const float *Ur, const float *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int gru_d (double *Y, const double *X, const double *U, const double *Ur, const double *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim);

int gru_inplace_s (float *X, const float *U, const float *Ur, const float *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int gru_inplace_d (double *X, const double *U, const double *Ur, const double *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim);


int gru_s (float *Y, const float *X, const float *U, const float *Ur, const float *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const float o = 1.0f;
    const size_t N2 = 2*N, NT = N*T, NT2 = 2*N*T;
    size_t nT, tN, tN3;

    float *R, *Z, *H;
    if (!(R=(float *)malloc(N*sizeof(float)))) { fprintf(stderr,"error in gru_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(Z=(float *)malloc(N*sizeof(float)))) { fprintf(stderr,"error in gru_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(float *)malloc(N*sizeof(float)))) { fprintf(stderr,"error in gru_s: problem with malloc. "); perror("malloc"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                //R[n] = 1.0f / (1.0f+expf(-X[N+n]));
                Z[n] = 1.0f / (1.0f+expf(-X[N2+n]));
                Y[n] = (1.0f-Z[n]) * tanhf(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N; tN3 = 3*tN;
                cblas_scopy((int)N,&X[tN3+N],1,R,1); cblas_scopy((int)N,&X[tN3+N2],1,Z,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Ur,(int)N,&Y[tN-N],1,o,R,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Uz,(int)N,&Y[tN-N],1,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    //R[n] = 1.0f / (1.0f+expf(-R[n]));
                    Z[n] = 1.0f / (1.0f+expf(-Z[n]));
                    //H[n] = R[n] * Y[tN-N+n];
                    H[n] = Y[tN-N+n] / (1.0f+expf(-R[n]));
                }
                cblas_scopy((int)N,&X[tN3],1,&Y[tN],1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&Y[tN],1);
                for (size_t n=0; n<N; ++n)
                {
                    Y[tN+n] = Z[n]*Y[tN-N+n] + (1.0f-Z[n])*tanhf(Y[tN+n]);
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                Z[n] = 1.0f / (1.0f+expf(-X[NT2+nT]));
                Y[nT] = (1.0f-Z[n]) * tanhf(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_scopy((int)N,&X[NT+t],(int)T,R,1); cblas_scopy((int)N,&X[NT2+t],(int)T,Z,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,Ur,(int)N,&Y[t-1],(int)T,o,R,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,Uz,(int)N,&Y[t-1],(int)T,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0f / (1.0f+expf(-Z[n]));
                    H[n] = Y[t-1+n*T] / (1.0f+expf(-R[n]));
                }
                cblas_scopy((int)N,&X[t],(int)T,&Y[t],(int)T);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&Y[t],(int)T);
                for (size_t n=0; n<N; ++n)
                {
                    nT = n*T;
                    Y[t+nT] = Z[n]*Y[t-1+nT] + (1.0f-Z[n])*tanhf(Y[t+nT]);
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
                Z[n] = 1.0f / (1.0f+expf(-X[NT2+nT]));
                Y[nT] = (1.0f-Z[n]) * tanhf(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_scopy((int)N,&X[NT+t],(int)T,R,1); cblas_scopy((int)N,&X[NT2+t],(int)T,Z,1);
                cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,Ur,(int)N,&Y[t-1],(int)T,o,R,1);
                cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,Uz,(int)N,&Y[t-1],(int)T,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0f / (1.0f+expf(-Z[n]));
                    H[n] = Y[t-1+n*T] / (1.0f+expf(-R[n]));
                }
                cblas_scopy((int)N,&X[t],(int)T,&Y[t],(int)T);
                cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&Y[t],(int)T);
                for (size_t n=0; n<N; ++n)
                {
                    nT = n*T;
                    Y[t+nT] = Z[n]*Y[t-1+nT] + (1.0f-Z[n])*tanhf(Y[t+nT]);
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                Z[n] = 1.0f / (1.0f+expf(-X[N2+n]));
                Y[n] = (1.0f-Z[n]) * tanhf(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N; tN3 = 3*tN;
                cblas_scopy((int)N,&X[tN3+N],1,R,1); cblas_scopy((int)N,&X[tN3+N2],1,Z,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Ur,(int)N,&Y[tN-N],1,o,R,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Uz,(int)N,&Y[tN-N],1,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0f / (1.0f+expf(-Z[n]));
                    H[n] = Y[tN-N+n] / (1.0f+expf(-R[n]));
                }
                cblas_scopy((int)N,&X[tN3],1,&Y[tN],1);
                cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&Y[tN],1);
                for (size_t n=0; n<N; ++n)
                {
                    Y[tN+n] = Z[n]*Y[tN-N+n] + (1.0f-Z[n])*tanhf(Y[tN+n]);
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in gru_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int gru_d (double *Y, const double *X, const double *U, const double *Ur, const double *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const double o = 1.0;
    const size_t N2 = 2*N, NT = N*T, NT2 = 2*N*T;
    size_t nT, tN, tN3;

    double *R, *Z, *H;
    if (!(R=(double *)malloc(N*sizeof(double)))) { fprintf(stderr,"error in gru_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(Z=(double *)malloc(N*sizeof(double)))) { fprintf(stderr,"error in gru_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(double *)malloc(N*sizeof(double)))) { fprintf(stderr,"error in gru_d: problem with malloc. "); perror("malloc"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                Z[n] = 1.0 / (1.0+exp(-X[N2+n]));
                Y[n] = (1.0-Z[n]) * tanh(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N; tN3 = 3*tN;
                cblas_dcopy((int)N,&X[tN3+N],1,R,1); cblas_dcopy((int)N,&X[tN3+N2],1,Z,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Ur,(int)N,&Y[tN-N],1,o,R,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Uz,(int)N,&Y[tN-N],1,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0 / (1.0+exp(-Z[n]));
                    H[n] = Y[tN-N+n] / (1.0+exp(-R[n]));
                }
                cblas_dcopy((int)N,&X[tN3],1,&Y[tN],1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&Y[tN],1);
                for (size_t n=0; n<N; ++n)
                {
                    Y[tN+n] = Z[n]*Y[tN-N+n] + (1.0-Z[n])*tanh(Y[tN+n]);
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                Z[n] = 1.0 / (1.0+exp(-X[NT2+nT]));
                Y[nT] = (1.0-Z[n]) * tanh(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_dcopy((int)N,&X[NT+t],(int)T,R,1); cblas_dcopy((int)N,&X[NT2+t],(int)T,Z,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,Ur,(int)N,&Y[t-1],(int)T,o,R,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,Uz,(int)N,&Y[t-1],(int)T,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0 / (1.0+exp(-Z[n]));
                    H[n] = Y[t-1+n*T] / (1.0+exp(-R[n]));
                }
                cblas_dcopy((int)N,&X[t],(int)T,&Y[t],(int)T);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&Y[t],(int)T);
                for (size_t n=0; n<N; ++n)
                {
                    nT = n*T;
                    Y[t+nT] = Z[n]*Y[t-1+nT] + (1.0-Z[n])*tanh(Y[t+nT]);
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
                Z[n] = 1.0 / (1.0+exp(-X[NT2+nT]));
                Y[nT] = (1.0-Z[n]) * tanh(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_dcopy((int)N,&X[NT+t],(int)T,R,1); cblas_dcopy((int)N,&X[NT2+t],(int)T,Z,1);
                cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,Ur,(int)N,&Y[t-1],(int)T,o,R,1);
                cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,Uz,(int)N,&Y[t-1],(int)T,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0 / (1.0+exp(-Z[n]));
                    H[n] = Y[t-1+n*T] / (1.0+exp(-R[n]));
                }
                cblas_dcopy((int)N,&X[t],(int)T,&Y[t],(int)T);
                cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&Y[t],(int)T);
                for (size_t n=0; n<N; ++n)
                {
                    nT = n*T;
                    Y[t+nT] = Z[n]*Y[t-1+nT] + (1.0-Z[n])*tanh(Y[t+nT]);
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                Z[n] = 1.0 / (1.0+exp(-X[N2+n]));
                Y[n] = (1.0-Z[n]) * tanh(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N; tN3 = 3*tN;
                cblas_dcopy((int)N,&X[tN3+N],1,R,1); cblas_dcopy((int)N,&X[tN3+N2],1,Z,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Ur,(int)N,&Y[tN-N],1,o,R,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Uz,(int)N,&Y[tN-N],1,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0 / (1.0+exp(-Z[n]));
                    H[n] = Y[tN-N+n] / (1.0+exp(-R[n]));
                }
                cblas_dcopy((int)N,&X[tN3],1,&Y[tN],1);
                cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&Y[tN],1);
                for (size_t n=0; n<N; ++n)
                {
                    Y[tN+n] = Z[n]*Y[tN-N+n] + (1.0-Z[n])*tanh(Y[tN+n]);
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in gru_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}



int gru_inplace_s (float *X, const float *U, const float *Ur, const float *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const float o = 1.0f;
    const size_t N2 = 2*N, N3 = 3*N, NT = N*T, NT2 = 2*N*T;
    size_t nT, tN, tN3;

    float *R, *Z, *H;
    if (!(R=(float *)malloc(N*sizeof(float)))) { fprintf(stderr,"error in gru_inplace_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(Z=(float *)malloc(N*sizeof(float)))) { fprintf(stderr,"error in gru_inplace_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(float *)malloc(N*sizeof(float)))) { fprintf(stderr,"error in gru_inplace_s: problem with malloc. "); perror("malloc"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                Z[n] = 1.0f / (1.0f+expf(-X[N2+n]));
                X[n] = (1.0f-Z[n]) * tanhf(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N; tN3 = 3*tN;
                cblas_scopy((int)N,&X[tN3+N],1,R,1); cblas_scopy((int)N,&X[tN3+N2],1,Z,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Ur,(int)N,&X[tN3-N3],1,o,R,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Uz,(int)N,&X[tN3-N3],1,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0f / (1.0f+expf(-Z[n]));
                    H[n] = X[tN3-N3+n] / (1.0f+expf(-R[n]));
                }
                cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[tN3],1);
                for (size_t n=0; n<N; ++n)
                {
                    X[tN3+n] = Z[n]*X[tN-N3+n] + (1.0f-Z[n])*tanhf(X[tN3+n]);
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                Z[n] = 1.0f / (1.0f+expf(-X[NT2+nT]));
                X[nT] = (1.0f-Z[n]) * tanhf(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_scopy((int)N,&X[NT+t],(int)T,R,1); cblas_scopy((int)N,&X[NT2+t],(int)T,Z,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,Ur,(int)N,&X[t-1],(int)T,o,R,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,Uz,(int)N,&X[t-1],(int)T,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0f / (1.0f+expf(-Z[n]));
                    H[n] = X[t-1+n*T] / (1.0f+expf(-R[n]));
                }
                cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[t],(int)T);
                for (size_t n=0; n<N; ++n)
                {
                    nT = n*T;
                    X[t+nT] = Z[n]*X[t-1+nT] + (1.0f-Z[n])*tanhf(X[t+nT]);
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
                Z[n] = 1.0f / (1.0f+expf(-X[NT2+nT]));
                X[nT] = (1.0f-Z[n]) * tanhf(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_scopy((int)N,&X[NT+t],(int)T,R,1); cblas_scopy((int)N,&X[NT2+t],(int)T,Z,1);
                cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,Ur,(int)N,&X[t-1],(int)T,o,R,1);
                cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,Uz,(int)N,&X[t-1],(int)T,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0f / (1.0f+expf(-Z[n]));
                    H[n] = X[t-1+n*T] / (1.0f+expf(-R[n]));
                }
                cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[t],(int)T);
                for (size_t n=0; n<N; ++n)
                {
                    nT = n*T;
                    X[t+nT] = Z[n]*X[t-1+nT] + (1.0f-Z[n])*tanhf(X[t+nT]);
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                Z[n] = 1.0f / (1.0f+expf(-X[N2+n]));
                X[n] = (1.0f-Z[n]) * tanhf(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N; tN3 = 3*tN;
                cblas_scopy((int)N,&X[tN3+N],1,R,1); cblas_scopy((int)N,&X[tN3+N2],1,Z,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Ur,(int)N,&X[tN3-N3],1,o,R,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Uz,(int)N,&X[tN3-N3],1,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0f / (1.0f+expf(-Z[n]));
                    H[n] = X[tN3-N3+n] / (1.0f+expf(-R[n]));
                }
                cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[tN3],1);
                for (size_t n=0; n<N; ++n)
                {
                    X[tN3+n] = Z[n]*X[tN-N3+n] + (1.0f-Z[n])*tanhf(X[tN3+n]);
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in gru_inplace_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int gru_inplace_d (double *X, const double *U, const double *Ur, const double *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const double o = 1.0;
    const size_t N2 = 2*N, N3 = 3*N, NT = N*T, NT2 = 2*N*T;
    size_t nT, tN, tN3;

    double *R, *Z, *H;
    if (!(R=(double *)malloc(N*sizeof(double)))) { fprintf(stderr,"error in gru_inplace_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(Z=(double *)malloc(N*sizeof(double)))) { fprintf(stderr,"error in gru_inplace_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(double *)malloc(N*sizeof(double)))) { fprintf(stderr,"error in gru_inplace_d: problem with malloc. "); perror("malloc"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                Z[n] = 1.0 / (1.0+exp(-X[N2+n]));
                X[n] = (1.0-Z[n]) * tanh(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N; tN3 = 3*tN;
                cblas_dcopy((int)N,&X[tN3+N],1,R,1); cblas_dcopy((int)N,&X[tN3+N2],1,Z,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Ur,(int)N,&X[tN3-N3],1,o,R,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Uz,(int)N,&X[tN3-N3],1,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0 / (1.0+exp(-Z[n]));
                    H[n] = X[tN3-N3+n] / (1.0+exp(-R[n]));
                }
                cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[tN3],1);
                for (size_t n=0; n<N; ++n)
                {
                    X[tN3+n] = Z[n]*X[tN-N3+n] + (1.0-Z[n])*tanh(X[tN3+n]);
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                Z[n] = 1.0 / (1.0+exp(-X[NT2+nT]));
                X[nT] = (1.0-Z[n]) * tanh(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_dcopy((int)N,&X[NT+t],(int)T,R,1); cblas_dcopy((int)N,&X[NT2+t],(int)T,Z,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,Ur,(int)N,&X[t-1],(int)T,o,R,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,Uz,(int)N,&X[t-1],(int)T,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0 / (1.0+exp(-Z[n]));
                    H[n] = X[t-1+n*T] / (1.0+exp(-R[n]));
                }
                cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[t],(int)T);
                for (size_t n=0; n<N; ++n)
                {
                    nT = n*T;
                    X[t+nT] = Z[n]*X[t-1+nT] + (1.0-Z[n])*tanh(X[t+nT]);
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
                Z[n] = 1.0 / (1.0+exp(-X[NT2+nT]));
                X[nT] = (1.0-Z[n]) * tanh(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_dcopy((int)N,&X[NT+t],(int)T,R,1); cblas_dcopy((int)N,&X[NT2+t],(int)T,Z,1);
                cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,Ur,(int)N,&X[t-1],(int)T,o,R,1);
                cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,Uz,(int)N,&X[t-1],(int)T,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0 / (1.0+exp(-Z[n]));
                    H[n] = X[t-1+n*T] / (1.0+exp(-R[n]));
                }
                cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[t],(int)T);
                for (size_t n=0; n<N; ++n)
                {
                    nT = n*T;
                    X[t+nT] = Z[n]*X[t-1+nT] + (1.0-Z[n])*tanh(X[t+nT]);
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                Z[n] = 1.0 / (1.0+exp(-X[N2+n]));
                X[n] = (1.0-Z[n]) * tanh(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N; tN3 = 3*tN;
                cblas_dcopy((int)N,&X[tN3+N],1,R,1); cblas_dcopy((int)N,&X[tN3+N2],1,Z,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Ur,(int)N,&X[tN3-N3],1,o,R,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Uz,(int)N,&X[tN3-N3],1,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0 / (1.0+exp(-Z[n]));
                    H[n] = X[tN3-N3+n] / (1.0+exp(-R[n]));
                }
                cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[tN3],1);
                for (size_t n=0; n<N; ++n)
                {
                    X[tN3+n] = Z[n]*X[tN-N3+n] + (1.0-Z[n])*tanh(X[tN3+n]);
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in gru_inplace_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
