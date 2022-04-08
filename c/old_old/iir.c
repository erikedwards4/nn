//CELL (~soma) stage: IIR filtering of each row or col of X according to dim.
//The IIR filters are specified by an Nx(Q+1) or (Q+1)xN matrix A,
//where Q is the IIR filter order (Q=0 means only a0; Q=1 means a0 and a1; etc.).

//The calling program must ensure that the sizes are correct, the filter is stable, etc.

//I just started this... finish later!! (Or skip.)

#include <stdio.h>
#include <cblas.h>

#ifdef __cplusplus
namespace ov {
extern "C" {
#endif

int iir_s (float *Y, const float *X, const float *A, const size_t N, const size_t T, const int Q, const char iscolmajor, const size_t dim);
int iir_d (double *Y, const double *X, const double *A, const size_t N, const size_t T, const int Q, const char iscolmajor, const size_t dim);
int iir_c (float *Y, const float *X, const float *A, const size_t N, const size_t T, const int Q, const char iscolmajor, const size_t dim);
int iir_z (double *Y, const double *X, const double *A, const size_t N, const size_t T, const int Q, const char iscolmajor, const size_t dim);

int iir_inplace_s (float *X, const float *A, const size_t N, const size_t T, const int Q, const char iscolmajor, const size_t dim);
int iir_inplace_d (double *X, const double *A, const size_t N, const size_t T, const int Q, const char iscolmajor, const size_t dim);
int iir_inplace_c (float *X, const float *A, const size_t N, const size_t T, const int Q, const char iscolmajor, const size_t dim);
int iir_inplace_z (double *X, const double *A, const size_t N, const size_t T, const int Q, const char iscolmajor, const size_t dim);


int iir_inplace_s (float *X, const float *A, const size_t N, const size_t T, const int Q, const char iscolmajor, const size_t dim)
{
    const size_t M = Q - 1;
    int n, t;

    //Checks
    if (N<1) { fprintf(stderr,"error in iir_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in iir_s: T (num time points) must be positive\n"); return 1; }
    if (Q<0) { fprintf(stderr,"error in iir_s: Q (filter order) must be nonnegative\n"); return 1; }

    if (N==1)
    {
        if (A[0]!=1.0f) { cblas_sscal((int)T,1.0f/A[0],A,1); cblas_sscal(N*T,1.0f/A[0],X,1); }
        for (size_t t=1; t<M; ++t) { X[t] -= cblas_sdot(t,&A[M-t],1,&X[0],1); }
        for (size_t t=M; t<T; ++t) { X[t] -= cblas_sdot(M,&A[0],1,&X[t-M],1); }
    }
    else if (dim==0)
    {
        if (N<3)
        {
            if (iscolmajor)
            {
                for (size_t n=0; n<N; ++n)
                {
                    if (A[n*Q1]!=1.0f) { cblas_sscal((int)T,1.0f/A[n*Q1],&A[n*Q1],1); }
                }
                for (size_t n=1; n<M; ++n)
                {
                    X[n] -= cblas_sdot(n,&A[M-n],1,&X[0],1);
                    X[n+R] -= cblas_sdot(n,&A[M-n],1,&X[R],1);
                }
                for (size_t n=M; n<R; ++n)
                {
                    X[n] -= cblas_sdot(M,&A[0],1,&X[n-M],1);
                    X[n+R] -= cblas_sdot(M,&A[0],1,&X[n-M+R],1);
                }
            }
            else
            {
                for (size_t n=1; n<M; ++n)
                {
                    X[n*C] -= cblas_sdot(n,&A[M-n],1,&X[0],(int)C);
                    X[1+n*C] -= cblas_sdot(n,&A[M-n],1,&X[1],(int)C);
                }
                for (size_t n=M; n<R; ++n)
                {
                    X[n*C] -= cblas_sdot(M,&A[0],1,&X[(n-M)*C],(int)C);
                    X[1+n*C] -= cblas_sdot(M,&A[0],1,&X[1+(n-M)*C],(int)C);
                }
            }
        }
        else
        {
            if (iscolmajor)
            {
                for (size_t n=1; n<M; ++n) { cblas_sgemv(CblasColMajor,CblasTrans,n,(int)C,-1.0f,&X[0],(int)R,&A[M-n],1,1.0f,&X[n],(int)R); }
                for (size_t n=M; n<R; ++n) { cblas_sgemv(CblasColMajor,CblasTrans,M,(int)C,-1.0f,&X[n-M],(int)R,&A[0],1,1.0f,&X[n],(int)R); }
            }
            else
            {
                for (size_t n=1; n<M; ++n) { cblas_sgemv(CblasRowMajor,CblasTrans,n,(int)C,-1.0f,&X[0],(int)C,&A[M-n],1,1.0f,&X[n*C],1); }
                for (size_t n=M; n<R; ++n) { cblas_sgemv(CblasRowMajor,CblasTrans,M,(int)C,-1.0f,&X[(n-M)*C],(int)C,&A[0],1,1.0f,&X[n*C],1); }
            }
        }
    }
    else if (dim==1)
    {
        if (R==1)
        {
            for (size_t n=1; n<M; ++n) { X[n] -= cblas_sdot(n,&A[M-n],1,&X[0],1); }
            for (size_t n=M; n<C; ++n) { X[n] -= cblas_sdot(M,&A[0],1,&X[n-M],1); }
        }
        else if (R==2)
        {
            if (iscolmajor)
            {
                for (size_t n=1; n<M; ++n)
                {
                    X[n*R] -= cblas_sdot(n,&A[M-n],1,&X[0],(int)R);
                    X[1+n*R] -= cblas_sdot(n,&A[M-n],1,&X[1],(int)R);
                }
                for (size_t n=M; n<C; ++n)
                {
                    X[n*R] -= cblas_sdot(M,&A[0],1,&X[(n-M)*R],(int)R);
                    X[1+n*R] -= cblas_sdot(M,&A[0],1,&X[1+(n-M)*R],(int)R);
                }
            }
            else
            {
                for (size_t n=1; n<M; ++n)
                {
                    X[n] -= cblas_sdot(n,&A[M-n],1,&X[0],1);
                    X[n+C] -= cblas_sdot(n,&A[M-n],1,&X[C],1);
                }
                for (size_t n=M; n<C; ++n)
                {
                    X[n] -= cblas_sdot(M,&A[0],1,&X[n-M],1);
                    X[n+C] -= cblas_sdot(M,&A[0],1,&X[n-M+C],1);
                }
            }
        }
        else
        {
            if (iscolmajor)
            {
                for (size_t n=1; n<M; ++n) { cblas_sgemv(CblasColMajor,CblasNoTrans,(int)R,n,-1.0f,&X[0],(int)R,&A[M-n],1,1.0f,&X[n*R],1); }
                for (size_t n=M; n<C; ++n) { cblas_sgemv(CblasColMajor,CblasNoTrans,(int)R,M,-1.0f,&X[(n-M)*R],(int)R,&A[0],1,1.0f,&X[n*R],1); }
            }
            else
            {
                for (size_t n=1; n<M; ++n) { cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)R,n,-1.0f,&X[0],(int)C,&A[M-n],1,1.0f,&X[n],(int)C); }
                for (size_t n=M; n<C; ++n) { cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)R,M,-1.0f,&X[n-M],(int)C,&A[0],1,1.0f,&X[n],(int)C); }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in iir_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int iir_d (double *X, const char iscolmajor, const size_t R, const size_t C, const double *A, const size_t N, const size_t dim)
{
    const size_t M = N - 1;
    int n;

    //Checks
    if (R<1) { fprintf(stderr,"error in iir_d: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in iir_d: C (ncols X) must be positive\n"); return 1; }
    if (N<1) { fprintf(stderr,"error in iir_d: N (filter order) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (C==1)
        {
            for (size_t n=1; n<M; ++n) { X[n] -= cblas_ddot(n,&A[M-n],1,&X[0],1); }
            for (size_t n=M; n<R; ++n) { X[n] -= cblas_ddot(M,&A[0],1,&X[n-M],1); }
        }
        else if (C==2)
        {
            if (iscolmajor)
            {
                for (size_t n=1; n<M; ++n)
                {
                    X[n] -= cblas_ddot(n,&A[M-n],1,&X[0],1);
                    X[n+R] -= cblas_ddot(n,&A[M-n],1,&X[R],1);
                }
                for (size_t n=M; n<R; ++n)
                {
                    X[n] -= cblas_ddot(M,&A[0],1,&X[n-M],1);
                    X[n+R] -= cblas_ddot(M,&A[0],1,&X[n-M+R],1);
                }
            }
            else
            {
                for (size_t n=1; n<M; ++n)
                {
                    X[n*C] -= cblas_ddot(n,&A[M-n],1,&X[0],(int)C);
                    X[1+n*C] -= cblas_ddot(n,&A[M-n],1,&X[1],(int)C);
                }
                for (size_t n=M; n<R; ++n)
                {
                    X[n*C] -= cblas_ddot(M,&A[0],1,&X[(n-M)*C],(int)C);
                    X[1+n*C] -= cblas_ddot(M,&A[0],1,&X[1+(n-M)*C],(int)C);
                }
            }
        }
        else
        {
            if (iscolmajor)
            {
                for (size_t n=1; n<M; ++n) { cblas_dgemv(CblasColMajor,CblasTrans,n,(int)C,-1.0,&X[0],(int)R,&A[M-n],1,1.0,&X[n],(int)R); }
                for (size_t n=M; n<R; ++n) { cblas_dgemv(CblasColMajor,CblasTrans,M,(int)C,-1.0,&X[n-M],(int)R,&A[0],1,1.0,&X[n],(int)R); }
            }
            else
            {
                for (size_t n=1; n<M; ++n) { cblas_dgemv(CblasRowMajor,CblasTrans,n,(int)C,-1.0,&X[0],(int)C,&A[M-n],1,1.0,&X[n*C],1); }
                for (size_t n=M; n<R; ++n) { cblas_dgemv(CblasRowMajor,CblasTrans,M,(int)C,-1.0,&X[(n-M)*C],(int)C,&A[0],1,1.0,&X[n*C],1); }
            }
        }
    }
    else if (dim==1)
    {
        if (R==1)
        {
            for (size_t n=1; n<M; ++n) { X[n] -= cblas_ddot(n,&A[M-n],1,&X[0],1); }
            for (size_t n=M; n<C; ++n) { X[n] -= cblas_ddot(M,&A[0],1,&X[n-M],1); }
        }
        else if (R==2)
        {
            if (iscolmajor)
            {
                for (size_t n=1; n<M; ++n)
                {
                    X[n*R] -= cblas_ddot(n,&A[M-n],1,&X[0],(int)R);
                    X[1+n*R] -= cblas_ddot(n,&A[M-n],1,&X[1],(int)R);
                }
                for (size_t n=M; n<C; ++n)
                {
                    X[n*R] -= cblas_ddot(M,&A[0],1,&X[(n-M)*R],(int)R);
                    X[1+n*R] -= cblas_ddot(M,&A[0],1,&X[1+(n-M)*R],(int)R);
                }
            }
            else
            {
                for (size_t n=1; n<M; ++n)
                {
                    X[n] -= cblas_ddot(n,&A[M-n],1,&X[0],1);
                    X[n+C] -= cblas_ddot(n,&A[M-n],1,&X[C],1);
                }
                for (size_t n=M; n<C; ++n)
                {
                    X[n] -= cblas_ddot(M,&A[0],1,&X[n-M],1);
                    X[n+C] -= cblas_ddot(M,&A[0],1,&X[n-M+C],1);
                }
            }
        }
        else
        {
            if (iscolmajor)
            {
                for (size_t n=1; n<M; ++n) { cblas_dgemv(CblasColMajor,CblasNoTrans,(int)R,n,-1.0,&X[0],(int)R,&A[M-n],1,1.0,&X[n*R],1); }
                for (size_t n=M; n<C; ++n) { cblas_dgemv(CblasColMajor,CblasNoTrans,(int)R,M,-1.0,&X[(n-M)*R],(int)R,&A[0],1,1.0,&X[n*R],1); }
            }
            else
            {
                for (size_t n=1; n<M; ++n) { cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)R,n,-1.0,&X[0],(int)C,&A[M-n],1,1.0,&X[n],(int)C); }
                for (size_t n=M; n<C; ++n) { cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)R,M,-1.0,&X[n-M],(int)C,&A[0],1,1.0,&X[n],(int)C); }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in iir_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int iir_c (float *X, const char iscolmajor, const size_t R, const size_t C, const float *A, const size_t N, const size_t dim)
{
    const float a[2] = {-1.0f,0.0f}, b[2] = {1.0f,0.0f};
    const size_t M = N - 1;
    int n;

    //Checks
    if (R<1) { fprintf(stderr,"error in iir_c: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in iir_c: C (ncols X) must be positive\n"); return 1; }
    if (N<1) { fprintf(stderr,"error in iir_c: N (filter order) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=1; n<M; ++n) { cblas_cgemv(CblasColMajor,CblasTrans,n,(int)C,&a[0],&X[0],(int)R,&A[2*(M-n)],1,&b[0],&X[2*n],(int)R); }
            for (size_t n=M; n<R; ++n) { cblas_cgemv(CblasColMajor,CblasTrans,M,(int)C,&a[0],&X[2*(n-M)],(int)R,&A[0],1,&b[0],&X[2*n],(int)R); }
        }
        else
        {
            for (size_t n=1; n<M; ++n) { cblas_cgemv(CblasRowMajor,CblasTrans,n,(int)C,&a[0],&X[0],(int)C,&A[2*(M-n)],1,&b[0],&X[2*n*C],1); }
            for (size_t n=M; n<R; ++n) { cblas_cgemv(CblasRowMajor,CblasTrans,M,(int)C,&a[0],&X[2*(n-M)*C],(int)C,&A[0],1,&b[0],&X[2*n*C],1); }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (size_t n=1; n<M; ++n) { cblas_cgemv(CblasColMajor,CblasNoTrans,(int)R,n,&a[0],&X[0],(int)R,&A[2*(M-n)],1,&b[0],&X[2*n*R],1); }
            for (size_t n=M; n<C; ++n) { cblas_cgemv(CblasColMajor,CblasNoTrans,(int)R,M,&a[0],&X[2*(n-M)*R],(int)R,&A[0],1,&b[0],&X[2*n*R],1); }
        }
        else
        {
            for (size_t n=1; n<M; ++n) { cblas_cgemv(CblasRowMajor,CblasNoTrans,(int)R,n,&a[0],&X[0],(int)C,&A[2*(M-n)],1,&b[0],&X[2*n],(int)C); }
            for (size_t n=M; n<C; ++n) { cblas_cgemv(CblasRowMajor,CblasNoTrans,(int)R,M,&a[0],&X[2*(n-M)],(int)C,&A[0],1,&b[0],&X[2*n],(int)C); }
        }
    }
    else
    {
        fprintf(stderr,"error in iir_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int iir_z (double *X, const char iscolmajor, const size_t R, const size_t C, const double *A, const size_t N, const size_t dim)
{
    const double a[2] = {-1.0,0.0}, b[2] = {1.0,0.0};
    const size_t M = N - 1;
    int n;

    //Checks
    if (R<1) { fprintf(stderr,"error in iir_z: R (nrows X) must be positive\n"); return 1; }
    if (C<1) { fprintf(stderr,"error in iir_z: C (ncols X) must be positive\n"); return 1; }
    if (N<1) { fprintf(stderr,"error in iir_z: N (filter order) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=1; n<M; ++n) { cblas_zgemv(CblasColMajor,CblasTrans,n,(int)C,&a[0],&X[0],(int)R,&A[2*(M-n)],1,&b[0],&X[2*n],(int)R); }
            for (size_t n=M; n<R; ++n) { cblas_zgemv(CblasColMajor,CblasTrans,M,(int)C,&a[0],&X[2*(n-M)],(int)R,&A[0],1,&b[0],&X[2*n],(int)R); }
        }
        else
        {
            for (size_t n=1; n<M; ++n) { cblas_zgemv(CblasRowMajor,CblasTrans,n,(int)C,&a[0],&X[0],(int)C,&A[2*(M-n)],1,&b[0],&X[2*n*C],1); }
            for (size_t n=M; n<R; ++n) { cblas_zgemv(CblasRowMajor,CblasTrans,M,(int)C,&a[0],&X[2*(n-M)*C],(int)C,&A[0],1,&b[0],&X[2*n*C],1); }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (size_t n=1; n<M; ++n) { cblas_zgemv(CblasColMajor,CblasNoTrans,(int)R,n,&a[0],&X[0],(int)R,&A[2*(M-n)],1,&b[0],&X[2*n*R],1); }
            for (size_t n=M; n<C; ++n) { cblas_zgemv(CblasColMajor,CblasNoTrans,(int)R,M,&a[0],&X[2*(n-M)*R],(int)R,&A[0],1,&b[0],&X[2*n*R],1); }
        }
        else
        {
            for (size_t n=1; n<M; ++n) { cblas_zgemv(CblasRowMajor,CblasNoTrans,(int)R,n,&a[0],&X[0],(int)C,&A[2*(M-n)],1,&b[0],&X[2*n],(int)C); }
            for (size_t n=M; n<C; ++n) { cblas_zgemv(CblasRowMajor,CblasNoTrans,(int)R,M,&a[0],&X[2*(n-M)],(int)C,&A[0],1,&b[0],&X[2*n],(int)C); }
        }
    }
    else
    {
        fprintf(stderr,"error in iir_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif

