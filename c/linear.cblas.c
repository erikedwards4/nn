//IN method.
//Linear transformation (weights only, no biases) of Ni inputs to No outputs.
//This version uses CBLAS

//Input X has Ni neurons and output Y has No neurons.

//The vecs of length Ni are always contiguous in memory, such that:

//If col-major: Y[:,l] = W' * X[:,l]
//where:
//X has size Ni x L
//Y has size No x L
//W has size Ni x No

//If row-major: Y[l,:] = X[l,:] * W'
//X has size L x Ni
//Y has size L x No
//W has size No x Ni

//For a different set-up that allows linear transformation of vecs in
//any orientation, use the linear function from math.

//I retain the for loop through L for compatibility with real-time streaming.

#include <cblas.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int linear_cblas_s (float *Y, const float *X, const float *W, const size_t Ni, const size_t No, const size_t L);
int linear_cblas_d (double *Y, const double *X, const double *W, const size_t Ni, const size_t No, const size_t L);
int linear_cblas_c (float *Y, const float *X, const float *W, const size_t Ni, const size_t No, const size_t L);
int linear_cblas_z (double *Y, const double *X, const double *W, const size_t Ni, const size_t No, const size_t L);


int linear_cblas_s (float *Y, const float *X, const float *W, const size_t Ni, const size_t No, const size_t L)
{
    for (size_t l=L; l>0u; --l, X+=Ni, Y+=No)
    {
        cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)No,(int)Ni,1.0f,W,(int)Ni,X,1,1.0f,Y,1);
    }

    return 0;
}


int linear_cblas_d (double *Y, const double *X, const double *W, const size_t Ni, const size_t No, const size_t L)
{
    for (size_t l=L; l>0u; --l, X+=Ni, Y+=No)
    {
        cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)No,(int)Ni,1.0,W,(int)Ni,X,1,1.0,Y,1);
    }

    return 0;
}


int linear_cblas_c (float *Y, const float *X, const float *W, const size_t Ni, const size_t No, const size_t L)
{
    const float o[2] = {1.0f,0.0f};

    for (size_t l=L; l>0u; --l, X+=2u*Ni, Y+=2u*No)
    {
        cblas_cgemv(CblasRowMajor,CblasNoTrans,(int)No,(int)Ni,o,W,(int)Ni,X,1,o,Y,1);
    }

    return 0;
}


int linear_cblas_z (double *Y, const double *X, const double *W, const size_t Ni, const size_t No, const size_t L)
{
    const double o[2] = {1.0,0.0};

    for (size_t l=L; l>0u; --l, X+=2u*Ni, Y+=2u*No)
    {
        cblas_zgemv(CblasRowMajor,CblasNoTrans,(int)No,(int)Ni,o,W,(int)Ni,X,1,o,Y,1);
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
