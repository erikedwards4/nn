//Causal FIR filtering of each row or col of X according to dim.

//FIR impulse responses are given in matrix B.
//For dim=0, X is NxT and B is NxL
//For dim=1, X is TxN and B is LxN
//where N is the number of neurons, T the number of time points, and L the FIR filter order.

#include <stdio.h>
#include <cblas.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int fir_s (float *Y, const float *X, const float *B, const size_t N, const size_t T, const size_t L, const char iscolmajor, const size_t dim)
{
    const float z = 0.0f;
    
    //Initialize Y to 0
    cblas_scopy((int)(N*T),&z,0,Y,1);

    if (N==1u)
    {
        for (size_t l=0u; l<L; l++) { cblas_saxpy((int)(T-l),B[l],&X[0],1,&Y[l],1); }
    }
    else if (dim==0u)
    {
        if (iscolmajor)
        {
            for (size_t n=0u; n<N; ++n)
            {
                for (size_t l=0u; l<L; l++) { cblas_saxpy((int)(T-l),B[n+l*N],&X[n],(int)N,&Y[n+l*N],(int)N); }
            }
        }
        else
        {
            for (size_t n=0u; n<N; ++n)
            {
                for (size_t l=0u; l<L; l++) { cblas_saxpy((int)(T-l),B[n*L+l],&X[n*T],1,&Y[n*T+l],1); }
            }
        }
    }
    else if (dim==1u)
    {
        if (iscolmajor)
        {
            for (size_t n=0u; n<N; ++n)
            {
                for (size_t l=0u; l<L; l++) { cblas_saxpy((int)(T-l),B[n*L+l],&X[n*T],1,&Y[n*T+l],1); }
            }
        }
        else
        {
            for (size_t n=0u; n<N; ++n)
            {
                for (size_t l=0u; l<L; l++) { cblas_saxpy((int)(T-l),B[n+l*N],&X[n],(int)N,&Y[n+l*N],(int)N); }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in fir_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int fir_d (double *Y, const double *X, const double *B, const size_t N, const size_t T, const size_t L, const char iscolmajor, const size_t dim)
{
    const double z = 0.0;
    
    //Initialize Y to 0
    cblas_dcopy((int)(N*T),&z,0,Y,1);

    if (N==1u)
    {
        for (size_t l=0u; l<L; l++) { cblas_daxpy((int)(T-l),B[l],&X[0],1,&Y[l],1); }
    }
    else if (dim==0u)
    {
        if (iscolmajor)
        {
            for (size_t n=0u; n<N; ++n)
            {
                for (size_t l=0u; l<L; l++) { cblas_daxpy((int)(T-l),B[n+l*N],&X[n],(int)N,&Y[n+l*N],(int)N); }
            }
        }
        else
        {
            for (size_t n=0u; n<N; ++n)
            {
                for (size_t l=0u; l<L; l++) { cblas_daxpy((int)(T-l),B[n*L+l],&X[n*T],1,&Y[n*T+l],1); }
            }
        }
    }
    else if (dim==1u)
    {
        if (iscolmajor)
        {
            for (size_t n=0u; n<N; ++n)
            {
                for (size_t l=0u; l<L; l++) { cblas_daxpy((int)(T-l),B[n*L+l],&X[n*T],1,&Y[n*T+l],1); }
            }
        }
        else
        {
            for (size_t n=0u; n<N; ++n)
            {
                for (size_t l=0u; l<L; l++) { cblas_daxpy((int)(T-l),B[n+l*N],&X[n],(int)N,&Y[n+l*N],(int)N); }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in fir_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int fir_c (float *Y, const float *X, const float *B, const size_t N, const size_t T, const size_t L, const char iscolmajor, const size_t dim)
{
    const float z[2] = {0.0f,0.0f};
    
    //Initialize Y to 0
    cblas_ccopy((int)(N*T),z,0,Y,1);

    if (N==1u)
    {
        for (size_t l=0u; l<L; l++) { cblas_caxpy((int)(T-l),&B[2*l],&X[0],1,&Y[2*l],1); }
    }
    else if (dim==0u)
    {
        if (iscolmajor)
        {
            for (size_t n=0u; n<N; ++n)
            {
                for (size_t l=0u; l<L; l++) { cblas_caxpy((int)(T-l),&B[2*(n+l*N)],&X[2*n],(int)N,&Y[2*(n+l*N)],(int)N); }
            }
        }
        else
        {
            for (size_t n=0u; n<N; ++n)
            {
                for (size_t l=0u; l<L; l++) { cblas_caxpy((int)(T-l),&B[2*(n*L+l)],&X[2*n*T],1,&Y[2*(n*T+l)],1); }
            }
        }
    }
    else if (dim==1u)
    {
        if (iscolmajor)
        {
            for (size_t n=0u; n<N; ++n)
            {
                for (size_t l=0u; l<L; l++) { cblas_caxpy((int)(T-l),&B[2*(n*L+l)],&X[2*n*T],1,&Y[2*(n*T+l)],1); }
            }
        }
        else
        {
            for (size_t n=0u; n<N; ++n)
            {
                for (size_t l=0u; l<L; l++) { cblas_caxpy((int)(T-l),&B[2*(n+l*N)],&X[2*n],(int)N,&Y[2*(n+l*N)],(int)N); }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in fir_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int fir_z (double *Y, const double *X, const double *B, const size_t N, const size_t T, const size_t L, const char iscolmajor, const size_t dim)
{
    const double z[2] = {0.0,0.0};
    
    //Initialize Y to 0
    cblas_zcopy((int)(N*T),z,0,Y,1);

    if (N==1u)
    {
        for (size_t l=0u; l<L; l++) { cblas_zaxpy((int)(T-l),&B[2*l],&X[0],1,&Y[2*l],1); }
    }
    else if (dim==0u)
    {
        if (iscolmajor)
        {
            for (size_t n=0u; n<N; ++n)
            {
                for (size_t l=0u; l<L; l++) { cblas_zaxpy((int)(T-l),&B[2*(n+l*N)],&X[2*n],(int)N,&Y[2*(n+l*N)],(int)N); }
            }
        }
        else
        {
            for (size_t n=0u; n<N; ++n)
            {
                for (size_t l=0u; l<L; l++) { cblas_zaxpy((int)(T-l),&B[2*(n*L+l)],&X[2*n*T],1,&Y[2*(n*T+l)],1); }
            }
        }
    }
    else if (dim==1u)
    {
        if (iscolmajor)
        {
            for (size_t n=0u; n<N; ++n)
            {
                for (size_t l=0u; l<L; l++) { cblas_zaxpy((int)(T-l),&B[2*(n*L+l)],&X[2*n*T],1,&Y[2*(n*T+l)],1); }
            }
        }
        else
        {
            for (size_t n=0u; n<N; ++n)
            {
                for (size_t l=0u; l<L; l++) { cblas_zaxpy((int)(T-l),&B[2*(n+l*N)],&X[2*n],(int)N,&Y[2*(n+l*N)],(int)N); }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in fir_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
