//IN method.
//Affine transformation (weights and biases) of Ni inputs to No outputs.
//This version uses CBLAS.

//Input X has Ni neurons and output Y has No neurons.
//Each output neuron has a bias term, so B is a vector of length No.

//The vecs of length Ni are always contiguous in memory, such that:

//If col-major: Y[:,l] = W' * X[:,l] + B
//where:
//X has size Ni x L
//Y has size No x L
//W has size Ni x No
//B has size No x 1

//If row-major: Y[l,:] = X[l,:] * W' + B
//X has size L x Ni
//Y has size L x No
//W has size No x Ni
//B has size 1 x No

//For a different set-up that allows affine transformation of vecs in
//any orientation, use the affine function from math.

#include <cblas.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int affine_cblas_s (float *Y, const float *X, const float *W, const float *B, const size_t Ni, const size_t No, const size_t L)
{
    for (size_t l=L; l>0u; --l, X+=Ni, Y+=No)
    {
        cblas_scopy((int)No,B,1,Y,1);
        cblas_sgemv(CblasRowMajor,CblasNoTrans,(int)No,(int)Ni,1.0f,W,(int)Ni,X,1,1.0f,Y,1);
    }

    return 0;
}


int affine_cblas_d (double *Y, const double *X, const double *W, const double *B, const size_t Ni, const size_t No, const size_t L)
{
    for (size_t l=L; l>0u; --l, X+=Ni, Y+=No)
    {
        cblas_dcopy((int)No,B,1,Y,1);
        cblas_dgemv(CblasRowMajor,CblasNoTrans,(int)No,(int)Ni,1.0,W,(int)Ni,X,1,1.0,Y,1);
    }

    return 0;
}


int affine_cblas_c (float *Y, const float *X, const float *W, const float *B, const size_t Ni, const size_t No, const size_t L)
{
    const float o[2] = {1.0f,0.0f};

    for (size_t l=L; l>0u; --l, X+=2u*Ni, Y+=2u*No)
    {
        cblas_ccopy((int)No,B,1,Y,1);
        cblas_cgemv(CblasRowMajor,CblasNoTrans,(int)No,(int)Ni,o,W,(int)Ni,X,1,o,Y,1);
    }

    return 0;
}


int affine_cblas_z (double *Y, const double *X, const double *W, const double *B, const size_t Ni, const size_t No, const size_t L)
{
    const double o[2] = {1.0,0.0};

    for (size_t l=L; l>0u; --l, X+=2u*Ni, Y+=2u*No)
    {
        cblas_zcopy((int)No,B,1,Y,1);
        cblas_zgemv(CblasRowMajor,CblasNoTrans,(int)No,(int)Ni,o,W,(int)Ni,X,1,o,Y,1);
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
