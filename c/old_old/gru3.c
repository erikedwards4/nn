//This does CELL (~soma) stage of GRU (gated recurrent unit) model.
//This requires each neuron to have 3 input time series, X, Xr, Xz,
//where X is the usual input and Xr, Xz the inputs for the reset and update gates.
//X, Xr, Xz are the output of a separate linear IN stage (weights and baises).

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
#include <math.h>
#include <cblas.h>
//#include <time.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int gru3_s (float *Y, const float *X, const float *Xr, const float *Xz, const float *U, const float *Ur, const float *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int gru3_d (double *Y, const double *X, const double *Xr, const double *Xz, const double *U, const double *Ur, const double *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim);

int gru3_inplace_s (float *X, const float *Xr, const float *Xz, const float *U, const float *Ur, const float *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int gru3_inplace_d (double *X, const double *Xr, const double *Xz, const double *U, const double *Ur, const double *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim);


int gru3_s (float *Y, const float *X, const float *Xr, const float *Xz, const float *U, const float *Ur, const float *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const float o = 1.0f;
    int nT, tN;
    float *R, *Z, *H;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in gru3_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in gru3_s: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(R=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in gru3_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(Z=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in gru3_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in gru3_s: problem with malloc. "); perror("malloc"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (N==1)
    {
        //R[0] = 1.0f / (1.0f+expf(-Xr[0]));
        Z[0] = 1.0f / (1.0f+expf(-Xz[0]));
        Y[0] = (1.0f-Z[0]) * tanhf(X[0]);
        for (size_t t=1; t<T; ++t)
        {
            //R[0] = 1.0f / (1.0f+expf(-Xr[t]-Ur[0]*Y[t-1]));
            Z[0] = 1.0f / (1.0f+expf(-Xz[t]-Uz[0]*Y[t-1]));
            //H[0] = R[0] * Y[t-1];
            H[0] = Y[t-1] / (1.0f+expf(-Xr[t]-Ur[0]*Y[t-1]));
            Y[t] = Z[0]*Y[t-1] + (1.0f-Z[0])*tanhf(X[t]+U[0]*H[0]);
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                //R[n] = 1.0f / (1.0f+expf(-Xr[n]));
                Z[n] = 1.0f / (1.0f+expf(-Xz[n]));
                Y[n] = (1.0f-Z[n]) * tanhf(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N;
                cblas_scopy((int)N,&Xr[tN],1,(int)R,1); cblas_scopy((int)N,&Xz[tN],1,Z,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Ur,(int)N,&Y[tN-N],1,o,(int)R,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Uz,(int)N,&Y[tN-N],1,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    //R[n] = 1.0f / (1.0f+expf(-R[n]));
                    Z[n] = 1.0f / (1.0f+expf(-Z[n]));
                    //H[n] = R[n] * Y[tN-N+n];
                    H[n] = Y[tN-N+n] / (1.0f+expf(-R[n]));
                }
                cblas_scopy((int)N,&X[tN],1,&Y[tN],1);
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
                //R[n] = 1.0f / (1.0f+expf(-Xr[nT]));
                Z[n] = 1.0f / (1.0f+expf(-Xz[nT]));
                Y[nT] = (1.0f-Z[n]) * tanhf(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_scopy((int)N,&Xr[t],(int)T,(int)R,1); cblas_scopy((int)N,&Xz[t],(int)T,Z,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,Ur,(int)N,&Y[t-1],(int)T,o,(int)R,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,Uz,(int)N,&Y[t-1],(int)T,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    //R[n] = 1.0f / (1.0f+expf(-R[n]));
                    Z[n] = 1.0f / (1.0f+expf(-Z[n]));
                    //H[n] = R[n] * Y[t-1+n*T];
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
                Z[n] = 1.0f / (1.0f+expf(-Xz[nT]));
                Y[nT] = (1.0f-Z[n]) * tanhf(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_scopy((int)N,&Xr[t],(int)T,(int)R,1); cblas_scopy((int)N,&Xz[t],(int)T,Z,1);
                cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,Ur,(int)N,&Y[t-1],(int)T,o,(int)R,1);
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
                Z[n] = 1.0f / (1.0f+expf(-Xz[n]));
                Y[n] = (1.0f-Z[n]) * tanhf(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N;
                cblas_scopy((int)N,&Xr[tN],1,(int)R,1); cblas_scopy((int)N,&Xz[tN],1,Z,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Ur,(int)N,&Y[tN-N],1,o,(int)R,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Uz,(int)N,&Y[tN-N],1,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0f / (1.0f+expf(-Z[n]));
                    H[n] = Y[tN-N+n] / (1.0f+expf(-R[n]));
                }
                cblas_scopy((int)N,&X[tN],1,&Y[tN],1);
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
        fprintf(stderr,"error in gru3_s: dim must be 0 or 1.\n"); return 1;
    }

    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);
    return 0;
}


int gru3_d (double *Y, const double *X, const double *Xr, const double *Xz, const double *U, const double *Ur, const double *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const double o = 1.0;
    int nT, tN;
    double *R, *Z, *H;

    //Checks
    if (N<1) { fprintf(stderr,"error in gru3_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in gru3_d: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(R=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in gru3_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(Z=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in gru3_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in gru3_d: problem with malloc. "); perror("malloc"); return 1; }

    if (N==1)
    {
        Z[0] = 1.0 / (1.0+exp(-Xz[0]));
        Y[0] = (1.0-Z[0]) * tanh(X[0]);
        for (size_t t=1; t<T; ++t)
        {
            Z[0] = 1.0 / (1.0+exp(-Xz[t]-Uz[0]*Y[t-1]));
            H[0] = Y[t-1] / (1.0+exp(-Xr[t]-Ur[0]*Y[t-1]));
            Y[t] = Z[0]*Y[t-1] + (1.0-Z[0])*tanh(X[t]+U[0]*H[0]);
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                Z[n] = 1.0 / (1.0+exp(-Xz[n]));
                Y[n] = (1.0-Z[n]) * tanh(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N;
                cblas_dcopy((int)N,&Xr[tN],1,(int)R,1); cblas_dcopy((int)N,&Xz[tN],1,Z,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Ur,(int)N,&Y[tN-N],1,o,(int)R,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Uz,(int)N,&Y[tN-N],1,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0 / (1.0+exp(-Z[n]));
                    H[n] = Y[tN-N+n] / (1.0+exp(-R[n]));
                }
                cblas_dcopy((int)N,&X[tN],1,&Y[tN],1);
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
                Z[n] = 1.0 / (1.0+exp(-Xz[nT]));
                Y[nT] = (1.0-Z[n]) * tanh(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_dcopy((int)N,&Xr[t],(int)T,(int)R,1); cblas_dcopy((int)N,&Xz[t],(int)T,Z,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,Ur,(int)N,&Y[t-1],(int)T,o,(int)R,1);
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
                Z[n] = 1.0 / (1.0+exp(-Xz[nT]));
                Y[nT] = (1.0-Z[n]) * tanh(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_dcopy((int)N,&Xr[t],(int)T,(int)R,1); cblas_dcopy((int)N,&Xz[t],(int)T,Z,1);
                cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,Ur,(int)N,&Y[t-1],(int)T,o,(int)R,1);
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
                Z[n] = 1.0 / (1.0+exp(-Xz[n]));
                Y[n] = (1.0-Z[n]) * tanh(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N;
                cblas_dcopy((int)N,&Xr[tN],1,(int)R,1); cblas_dcopy((int)N,&Xz[tN],1,Z,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Ur,(int)N,&Y[tN-N],1,o,(int)R,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Uz,(int)N,&Y[tN-N],1,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0 / (1.0+exp(-Z[n]));
                    H[n] = Y[tN-N+n] / (1.0+exp(-R[n]));
                }
                cblas_dcopy((int)N,&X[tN],1,&Y[tN],1);
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
        fprintf(stderr,"error in gru3_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}



int gru3_inplace_s (float *X, const float *Xr, const float *Xz, const float *U, const float *Ur, const float *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const float o = 1.0f;
    int nT, tN;
    float *R, *Z, *H;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in gru3_inplace_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in gru3_inplace_s: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(R=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in gru3_inplace_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(Z=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in gru3_inplace_s: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in gru3_inplace_s: problem with malloc. "); perror("malloc"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (N==1)
    {
        //R[0] = 1.0f / (1.0f+expf(-Xr[0]));
        Z[0] = 1.0f / (1.0f+expf(-Xz[0]));
        X[0] = (1.0f-Z[0]) * tanhf(X[0]);
        for (size_t t=1; t<T; ++t)
        {
            //R[0] = 1.0f / (1.0f+expf(-Xr[t]-Ur[0]*X[t-1]));
            Z[0] = 1.0f / (1.0f+expf(-Xz[t]-Uz[0]*X[t-1]));
            //H[0] = R[0] * X[t-1];
            H[0] = X[t-1] / (1.0f+expf(-Xr[t]-Ur[0]*X[t-1]));
            X[t] = Z[0]*X[t-1] + (1.0f-Z[0])*tanhf(X[t]+U[0]*H[0]);
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                //R[n] = 1.0f / (1.0f+expf(-Xr[n]));
                Z[n] = 1.0f / (1.0f+expf(-Xz[n]));
                X[n] = (1.0f-Z[n]) * tanhf(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N;
                cblas_scopy((int)N,&Xr[tN],1,(int)R,1); cblas_scopy((int)N,&Xz[tN],1,Z,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Ur,(int)N,&X[tN-N],1,o,(int)R,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Uz,(int)N,&X[tN-N],1,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    //R[n] = 1.0f / (1.0f+expf(-R[n]));
                    Z[n] = 1.0f / (1.0f+expf(-Z[n]));
                    //H[n] = R[n] * X[tN-N+n];
                    H[n] = X[tN-N+n] / (1.0f+expf(-R[n]));
                }
                cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[tN],1);
                for (size_t n=0; n<N; ++n)
                {
                    X[tN+n] = Z[n]*X[tN-N+n] + (1.0f-Z[n])*tanhf(X[tN+n]);
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                Z[n] = 1.0f / (1.0f+expf(-Xz[nT]));
                X[nT] = (1.0f-Z[n]) * tanhf(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_scopy((int)N,&Xr[t],(int)T,(int)R,1); cblas_scopy((int)N,&Xz[t],(int)T,Z,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,Ur,(int)N,&X[t-1],(int)T,o,(int)R,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,Ur,(int)N,&X[t-1],(int)T,o,Z,1);
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
                Z[n] = 1.0f / (1.0f+expf(-Xz[nT]));
                X[nT] = (1.0f-Z[n]) * tanhf(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_scopy((int)N,&Xr[t],(int)T,(int)R,1); cblas_scopy((int)N,&Xz[t],(int)T,Z,1);
                cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,Ur,(int)N,&X[t-1],(int)T,o,(int)R,1);
                cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,Ur,(int)N,&X[t-1],(int)T,o,Z,1);
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
                Z[n] = 1.0f / (1.0f+expf(-Xz[n]));
                X[n] = (1.0f-Z[n]) * tanhf(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N;
                cblas_scopy((int)N,&Xr[tN],1,(int)R,1); cblas_scopy((int)N,&Xz[tN],1,Z,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Ur,(int)N,&X[tN-N],1,o,(int)R,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Uz,(int)N,&X[tN-N],1,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0f / (1.0f+expf(-Z[n]));
                    H[n] = X[tN-N+n] / (1.0f+expf(-R[n]));
                }
                cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[tN],1);
                for (size_t n=0; n<N; ++n)
                {
                    X[tN+n] = Z[n]*X[tN-N+n] + (1.0f-Z[n])*tanhf(X[tN+n]);
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in gru3_inplace_s: dim must be 0 or 1.\n"); return 1;
    }

    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);
    return 0;
}


int gru3_inplace_d (double *X, const double *Xr, const double *Xz, const double *U, const double *Ur, const double *Uz, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const double o = 1.0;
    int nT, tN;
    double *R, *Z, *H;

    //Checks
    if (N<1) { fprintf(stderr,"error in gru3_inplace_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in gru3_inplace_d: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(R=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in gru3_inplace_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(Z=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in gru3_inplace_d: problem with malloc. "); perror("malloc"); return 1; }
    if (!(H=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in gru3_inplace_d: problem with malloc. "); perror("malloc"); return 1; }

    if (N==1)
    {
        Z[0] = 1.0 / (1.0+exp(-Xz[0]));
        X[0] = (1.0-Z[0]) * tanh(X[0]);
        for (size_t t=1; t<T; ++t)
        {
            Z[0] = 1.0 / (1.0+exp(-Xz[t]-Uz[0]*X[t-1]));
            H[0] = X[t-1] / (1.0+exp(-Xr[t]-Ur[0]*X[t-1]));
            X[t] = Z[0]*X[t-1] + (1.0-Z[0])*tanh(X[t]+U[0]*H[0]);
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                Z[n] = 1.0 / (1.0+exp(-Xz[n]));
                X[n] = (1.0-Z[n]) * tanh(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N;
                cblas_dcopy((int)N,&Xr[tN],1,(int)R,1); cblas_dcopy((int)N,&Xz[tN],1,Z,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Ur,(int)N,&X[tN-N],1,o,(int)R,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,Uz,(int)N,&X[tN-N],1,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0 / (1.0+exp(-Z[n]));
                    H[n] = X[tN-N+n] / (1.0+exp(-R[n]));
                }
                cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[tN],1);
                for (size_t n=0; n<N; ++n)
                {
                    X[tN+n] = Z[n]*X[tN-N+n] + (1.0-Z[n])*tanh(X[tN+n]);
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                Z[n] = 1.0 / (1.0+exp(-Xz[nT]));
                X[nT] = (1.0-Z[n]) * tanh(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_dcopy((int)N,&Xr[t],(int)T,(int)R,1); cblas_dcopy((int)N,&Xz[t],(int)T,Z,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,Ur,(int)N,&X[t-1],(int)T,o,(int)R,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,Ur,(int)N,&X[t-1],(int)T,o,Z,1);
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
                Z[n] = 1.0 / (1.0+exp(-Xz[nT]));
                X[nT] = (1.0-Z[n]) * tanh(X[nT]);
            }
            for (size_t t=1; t<T; ++t)
            {
                cblas_dcopy((int)N,&Xr[t],(int)T,(int)R,1); cblas_dcopy((int)N,&Xz[t],(int)T,Z,1);
                cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,Ur,(int)N,&X[t-1],(int)T,o,(int)R,1);
                cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,Ur,(int)N,&X[t-1],(int)T,o,Z,1);
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
                Z[n] = 1.0 / (1.0+exp(-Xz[n]));
                X[n] = (1.0-Z[n]) * tanh(X[n]);
            }
            for (size_t t=1; t<T; ++t)
            {
                tN = t*N;
                cblas_dcopy((int)N,&Xr[tN],1,(int)R,1); cblas_dcopy((int)N,&Xz[tN],1,Z,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Ur,(int)N,&X[tN-N],1,o,(int)R,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,Uz,(int)N,&X[tN-N],1,o,Z,1);
                for (size_t n=0; n<N; ++n)
                {
                    Z[n] = 1.0 / (1.0+exp(-Z[n]));
                    H[n] = X[tN-N+n] / (1.0+exp(-R[n]));
                }
                cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,H,1,o,&X[tN],1);
                for (size_t n=0; n<N; ++n)
                {
                    X[tN+n] = Z[n]*X[tN-N+n] + (1.0-Z[n])*tanh(X[tN+n]);
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in gru3_inplace_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
