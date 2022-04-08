//Gets output side of Elman RNN layer.
//There is no strict definition of Elman RNN in the literature.
//For example, the Wikipedia article doesn't specify the dims of the hidden or other vectors.
//Here I implement an interpretation where there are N neurons and thus N inputs/outputs.
//The inputs here are from the WB stage, so reduced to N driving input time-series in X.

//Again, this is not an Elman "network", rather a layer of N neurons
//that I have named "Elman" neurons due to their great similarity to an Elman RNN.

//To do: should I allow other output activations other than logistic?
//I could make that a separate stage in the 

#include <stdio.h>
#include <math.h>
#include <cblas.h>
//#include <time.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int jordan_s (float *Y, const float *X, const float *U, float *Y1, const float *W, const float *B, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int jordan_d (double *Y, const double *X, const double *U, double *Y1, const double *W, const double *B, const size_t N, const size_t T, const char iscolmajor, const size_t dim);


int jordan_s (float *Y, const float *X, const float *U, float *Y1, const float *W, const float *B, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const float o = 1.0f;
    int n, t;
    float *H;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in jordan_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in jordan_s: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(H=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in jordan_s: problem with malloc. "); perror("malloc"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (N==1)
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
        if (dim==0)
        {
            if (iscolmajor)
            {
                for (size_t n=0; n<N; ++n) { cblas_scopy((int)T,&B[n],0,&Y[n],(int)N); }
                cblas_scopy((int)N,&X[0],1,H,1);
                cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,Y1,1,o,H,1);
                for (size_t n=0; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-H[n])); }
                cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[0],1);
                for (size_t n=0; n<N; ++n) { Y[n] = 1.0f/(1.0f+expf(-Y[n])); }
                for (size_t t=1; t<T; ++t)
                {
                    cblas_scopy((int)N,&X[t*N],1,H,1);
                    cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,&Y[(t-1)*N],1,o,H,1);
                    for (size_t n=0; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-H[n])); }
                    cblas_sgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[t*N],1);
                    for (size_t n=0; n<N; ++n) { Y[t*N+n] = 1.0f/(1.0f+expf(-Y[t*N+n])); }
                }
            }
            else
            {
                for (size_t n=0; n<N; ++n) { cblas_scopy((int)T,&B[n],0,&Y[n*T],1); }
                cblas_scopy((int)N,&X[0],(int)T,H,1);
                cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,Y1,1,o,H,1);
                for (size_t n=0; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-H[n])); }
                cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[0],(int)T);
                for (size_t n=0; n<N; ++n) { Y[n*T] = 1.0f/(1.0f+expf(-Y[n*T])); }
                for (size_t t=1; t<T; ++t)
                {
                    cblas_scopy((int)N,&X[t],(int)T,H,1);
                    cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,&Y[t-1],(int)T,o,H,1);
                    for (size_t n=0; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-H[n])); }
                    cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[t],(int)T);
                    for (size_t n=0; n<N; ++n) { Y[t+n*T] = 1.0f/(1.0f+expf(-Y[t+n*T])); }
                }
            }
        }
        else if (dim==1)
        {
            if (iscolmajor)
            {
                for (size_t n=0; n<N; ++n) { cblas_scopy((int)T,&B[n],0,&Y[n*T],1); }
                cblas_scopy((int)N,&X[0],(int)T,H,1);
                cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,Y1,1,o,H,1);
                for (size_t n=0; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-H[n])); }
                cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[0],(int)T);
                for (size_t n=0; n<N; ++n) { Y[n*T] = 1.0f/(1.0f+expf(-Y[n*T])); }
                for (size_t t=1; t<T; ++t)
                {
                    cblas_scopy((int)N,&X[t],(int)T,H,1);
                    cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,&Y[t-1],(int)T,o,H,1);
                    for (size_t n=0; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-H[n])); }
                    cblas_sgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[t],(int)T);
                    for (size_t n=0; n<N; ++n) { Y[t+n*T] = 1.0f/(1.0f+expf(-Y[t+n*T])); }
                }
            }
            else
            {
                for (size_t n=0; n<N; ++n) { cblas_scopy((int)T,&B[n],0,&Y[n],(int)N); }
                cblas_scopy((int)N,&X[0],1,H,1);
                cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,Y1,1,o,H,1);
                for (size_t n=0; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-H[n])); }
                cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[0],1);
                for (size_t n=0; n<N; ++n) { Y[n] = 1.0f/(1.0f+expf(-Y[n])); }
                for (size_t t=1; t<T; ++t)
                {
                    cblas_scopy((int)N,&X[t*N],1,H,1);
                    cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,&Y[(t-1)*N],1,o,H,1);
                    for (size_t n=0; n<N; ++n) { H[n] = 1.0f/(1.0f+expf(-H[n])); }
                    cblas_sgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[t*N],1);
                    for (size_t n=0; n<N; ++n) { Y[t*N+n] = 1.0f/(1.0f+expf(-Y[t*N+n])); }
                }
            }
        }
        else
        {
            fprintf(stderr,"error in jordan_s: dim must be 0 or 1.\n"); return 1;
        }
    }
    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);

    return 0;
}


int jordan_d (double *Y, const double *X, const double *U, double *Y1, const double *W, const double *B, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const double o = 1.0;
    int n, t;
    double *H;

    //Checks
    if (N<1) { fprintf(stderr,"error in jordan_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in jordan_d: T (num time points) must be positive\n"); return 1; }

    //Allocate
    if (!(H=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in jordan_d: problem with malloc. "); perror("malloc"); return 1; }

    if (N==1)
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
        if (dim==0)
        {
            if (iscolmajor)
            {
                for (size_t n=0; n<N; ++n) { cblas_dcopy((int)T,&B[n],0,&Y[n],(int)N); }
                cblas_dcopy((int)N,&X[0],1,H,1);
                cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,Y1,1,o,H,1);
                for (size_t n=0; n<N; ++n) { H[n] = 1.0/(1.0+exp(-H[n])); }
                cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[0],1);
                for (size_t n=0; n<N; ++n) { Y[n] = 1.0/(1.0+exp(-Y[n])); }
                for (size_t t=1; t<T; ++t)
                {
                    cblas_dcopy((int)N,&X[t*N],1,H,1);
                    cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,&Y[(t-1)*N],1,o,H,1);
                    for (size_t n=0; n<N; ++n) { H[n] = 1.0/(1.0+exp(-H[n])); }
                    cblas_dgemv(CblasColMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[t*N],1);
                    for (size_t n=0; n<N; ++n) { Y[t*N+n] = 1.0/(1.0+exp(-Y[t*N+n])); }
                }
            }
            else
            {
                for (size_t n=0; n<N; ++n) { cblas_dcopy((int)T,&B[n],0,&Y[n*T],1); }
                cblas_dcopy((int)N,&X[0],(int)T,H,1);
                cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,Y1,1,o,H,1);
                for (size_t n=0; n<N; ++n) { H[n] = 1.0/(1.0+exp(-H[n])); }
                cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[0],(int)T);
                for (size_t n=0; n<N; ++n) { Y[n*T] = 1.0/(1.0+exp(-Y[n*T])); }
                for (size_t t=1; t<T; ++t)
                {
                    cblas_dcopy((int)N,&X[t],(int)T,H,1);
                    cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,U,(int)N,&Y[t-1],(int)T,o,H,1);
                    for (size_t n=0; n<N; ++n) { H[n] = 1.0/(1.0+exp(-H[n])); }
                    cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[t],(int)T);
                    for (size_t n=0; n<N; ++n) { Y[t+n*T] = 1.0/(1.0+exp(-Y[t+n*T])); }
                }
            }
        }
        else if (dim==1)
        {
            if (iscolmajor)
            {
                for (size_t n=0; n<N; ++n) { cblas_dcopy((int)T,&B[n],0,&Y[n*T],1); }
                cblas_dcopy((int)N,&X[0],(int)T,H,1);
                cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,Y1,1,o,H,1);
                for (size_t n=0; n<N; ++n) { H[n] = 1.0/(1.0+exp(-H[n])); }
                cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[0],(int)T);
                for (size_t n=0; n<N; ++n) { Y[n*T] = 1.0/(1.0+exp(-Y[n*T])); }
                for (size_t t=1; t<T; ++t)
                {
                    cblas_dcopy((int)N,&X[t],(int)T,H,1);
                    cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,&Y[t-1],(int)T,o,H,1);
                    for (size_t n=0; n<N; ++n) { H[n] = 1.0/(1.0+exp(-H[n])); }
                    cblas_dgemv(CblasColMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[t],(int)T);
                    for (size_t n=0; n<N; ++n) { Y[t+n*T] = 1.0/(1.0+exp(-Y[t+n*T])); }
                }
            }
            else
            {
                for (size_t n=0; n<N; ++n) { cblas_dcopy((int)T,&B[n],0,&Y[n],(int)N); }
                cblas_dcopy((int)N,&X[0],1,H,1);
                cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,Y1,1,o,H,1);
                for (size_t n=0; n<N; ++n) { H[n] = 1.0/(1.0+exp(-H[n])); }
                cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[0],1);
                for (size_t n=0; n<N; ++n) { Y[n] = 1.0/(1.0+exp(-Y[n])); }
                for (size_t t=1; t<T; ++t)
                {
                    cblas_dcopy((int)N,&X[t*N],1,H,1);
                    cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,U,(int)N,&Y[(t-1)*N],1,o,H,1);
                    for (size_t n=0; n<N; ++n) { H[n] = 1.0/(1.0+exp(-H[n])); }
                    cblas_dgemv(CblasRowMajor,CblasTrans,(int)N,(int)N,o,W,(int)N,H,1,o,&Y[t*N],1);
                    for (size_t n=0; n<N; ++n) { Y[t*N+n] = 1.0/(1.0+exp(-Y[t*N+n])); }
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
