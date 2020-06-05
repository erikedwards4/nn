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
namespace openn {
extern "C" {
#endif

int elman_s (float *Y, const float *X, const float *U, float *H, const float *W, const float *B, const int N, const int T, const int dim, const char iscolmajor);
int elman_d (double *Y, const double *X, const double *U, double *H, const double *W, const double *B, const int N, const int T, const int dim, const char iscolmajor);


int elman_s (float *Y, const float *X, const float *U, float *H, const float *W, const float *B, const int N, const int T, const int dim, const char iscolmajor)
{
    const float o = 1.0f;
    int n, t;
    float *tmp;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in elman_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in elman_s: T (num time points) must be positive\n"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (N==1)
    {
        for (t=0; t<T; t++)
        {
            H[0] = 1.0f / (1.0f+expf(-X[t]-U[0]*H[0]));
            Y[t] = 1.0f / (1.0f+expf(-B[0]-W[0]*H[0]));
        }
    }
    else
    {
        //Allocate
        if (!(tmp=(float *)malloc((size_t)(N)*sizeof(float)))) { fprintf(stderr,"error in elman_s: problem with malloc. "); perror("malloc"); return 1; }

        if (dim==0)
        {
            if (iscolmajor)
            {
                for (t=0; t<T; t++)
                {
                    cblas_scopy(N,&X[t*N],1,tmp,1);
                    cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,U,N,H,1,o,tmp,1);
                    for (n=0; n<N; n++) { H[n] = 1.0f/(1.0f+expf(-tmp[n])); }
                    //cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,U,N,H,1,o,&X[t*N],1);
                    //for (n=0; n<N; n++) { H[n] = 1.0f/(1.0f+expf(-X[t*N+n])); }
                    //for (n=0; n<N; n++) { Y[t*N+n] = 1.0f/(1.0f+expf(-cblas_sdot(N,&W[n],N,H,1)-B[n])); }
                    cblas_scopy(N,B,1,tmp,1);
                    cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,o,W,N,H,1,o,tmp,1);
                    for (n=0; n<N; n++) { Y[t*N+n] = 1.0f/(1.0f+expf(-tmp[n])); }
                }
            }
            else
            {
                for (t=0; t<T; t++)
                {
                    cblas_scopy(N,&X[t],T,tmp,1);
                    cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,U,N,H,1,o,tmp,1);
                    for (n=0; n<N; n++) { H[n] = 1.0f/(1.0f+expf(-tmp[n])); }
                    cblas_scopy(N,B,1,tmp,1);
                    cblas_sgemv(CblasRowMajor,CblasNoTrans,N,N,o,W,N,H,1,o,tmp,1);
                    for (n=0; n<N; n++) { Y[t+n*T] = 1.0f/(1.0f+expf(-tmp[n])); }
                }
            }
        }
        else if (dim==1)
        {
            if (iscolmajor)
            {
                for (t=0; t<T; t++)
                {
                    cblas_scopy(N,&X[t],T,tmp,1);
                    cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,U,N,H,1,o,tmp,1);
                    for (n=0; n<N; n++) { H[n] = 1.0f/(1.0f+expf(-tmp[n])); }
                    cblas_scopy(N,B,1,tmp,1);
                    cblas_sgemv(CblasColMajor,CblasTrans,N,N,o,W,N,H,1,o,tmp,1);
                    for (n=0; n<N; n++) { Y[t+n*T] = 1.0f/(1.0f+expf(-tmp[n])); }
                }
            }
            else
            {
                for (t=0; t<T; t++)
                {
                    cblas_scopy(N,&X[t*N],1,tmp,1);
                    cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,U,N,H,1,o,tmp,1);
                    for (n=0; n<N; n++) { H[n] = 1.0f/(1.0f+expf(-tmp[n])); }
                    cblas_scopy(N,B,1,tmp,1);
                    cblas_sgemv(CblasRowMajor,CblasTrans,N,N,o,W,N,H,1,o,tmp,1);
                    for (n=0; n<N; n++) { Y[t*N+n] = 1.0f/(1.0f+expf(-tmp[n])); }
                }
            }
        }
        else
        {
            fprintf(stderr,"error in elman_s: dim must be 0 or 1.\n"); return 1;
        }
    }
    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);

    return 0;
}


int elman_d (double *Y, const double *X, const double *U, double *H, const double *W, const double *B, const int N, const int T, const int dim, const char iscolmajor)
{
    const double o = 1.0;
    int n, t;
    double *tmp;

    //Checks
    if (N<1) { fprintf(stderr,"error in elman_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in elman_d: T (num time points) must be positive\n"); return 1; }

    if (N==1)
    {
        for (t=0; t<T; t++)
        {
            H[0] = 1.0 / (1.0+exp(-X[t]-U[0]*H[0]));
            Y[t] = 1.0 / (1.0+exp(-B[0]-W[0]*H[0]));
        }
    }
    else
    {
        //Allocate
        if (!(tmp=(double *)malloc((size_t)(N)*sizeof(double)))) { fprintf(stderr,"error in elman_d: problem with malloc. "); perror("malloc"); return 1; }

        if (dim==0)
        {
            if (iscolmajor)
            {
                for (t=0; t<T; t++)
                {
                    cblas_dcopy(N,&X[t*N],1,tmp,1);
                    cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,U,N,H,1,o,tmp,1);
                    for (n=0; n<N; n++) { H[n] = 1.0/(1.0+exp(-tmp[n])); }
                    cblas_dcopy(N,B,1,tmp,1);
                    cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,o,W,N,H,1,o,tmp,1);
                    for (n=0; n<N; n++) { Y[t*N+n] = 1.0/(1.0+exp(-tmp[n])); }
                }
            }
            else
            {
                for (t=0; t<T; t++)
                {
                    cblas_dcopy(N,&X[t],T,tmp,1);
                    cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,U,N,H,1,o,tmp,1);
                    for (n=0; n<N; n++) { H[n] = 1.0/(1.0+exp(-tmp[n])); }
                    cblas_dcopy(N,B,1,tmp,1);
                    cblas_dgemv(CblasRowMajor,CblasNoTrans,N,N,o,W,N,H,1,o,tmp,1);
                    for (n=0; n<N; n++) { Y[t+n*T] = 1.0/(1.0+exp(-tmp[n])); }
                }
            }
        }
        else if (dim==1)
        {
            if (iscolmajor)
            {
                for (t=0; t<T; t++)
                {
                    cblas_dcopy(N,&X[t],T,tmp,1);
                    cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,U,N,H,1,o,tmp,1);
                    for (n=0; n<N; n++) { H[n] = 1.0/(1.0+exp(-tmp[n])); }
                    cblas_dcopy(N,B,1,tmp,1);
                    cblas_dgemv(CblasColMajor,CblasTrans,N,N,o,W,N,H,1,o,tmp,1);
                    for (n=0; n<N; n++) { Y[t+n*T] = 1.0/(1.0+exp(-tmp[n])); }
                }
            }
            else
            {
                for (t=0; t<T; t++)
                {
                    cblas_dcopy(N,&X[t*N],1,tmp,1);
                    cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,U,N,H,1,o,tmp,1);
                    for (n=0; n<N; n++) { H[n] = 1.0/(1.0+exp(-tmp[n])); }
                    cblas_dcopy(N,B,1,tmp,1);
                    cblas_dgemv(CblasRowMajor,CblasTrans,N,N,o,W,N,H,1,o,tmp,1);
                    for (n=0; n<N; n++) { Y[t*N+n] = 1.0/(1.0+exp(-tmp[n])); }
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
