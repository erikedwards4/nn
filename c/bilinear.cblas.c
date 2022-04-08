//IN method.
//Bilinear transformation (3D tensor of weights, 1D vector of biases)
//of Ni1 inputs from X1 and Ni2 inputs from X2 to No outputs in Y.
//This version uses CBLAS.

//Input X1 has Ni1 neurons, X2 has Ni2 neurons, and output Y has No neurons.

//If col-major: Y[o,l] = X1[:,l]' * W[:,:,o] * X2[:,l] + B[o]
//where:
//X1 has size Ni1 x L
//X2 has size Ni2 x L
//Y  has size No x L
//W  has size Ni1 x Ni2 x No
//B  has size No x 1

//If row-major: Y[l,o] = X1[l,:] * W[o,:,:] * X2[l,:]' + B[o]
//X1 has size L x Ni1
//X2 has size L x Ni2
//Y  has size L x No
//W  has size No x Ni2 x Ni1
//B  has size 1 x No

//This is equal to the outer product of X1 and X2 at each l (time point),
//and then take the weighted sum by each of the No matrices within W,
//producing No scalars, one for each output neuron in Y.

//I retain the for loop through L for compatibility with real-time streaming.

//This CBLAS version not working yet!!

#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int bilinear_cblas_s (float *Y, const float *X1, const float *X2, const float *W, const float *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L)
{
    const size_t N12 = Ni1*Ni2, Nw = N12*No;

    float *outer_prod;
    outer_prod = (float *)malloc(N12*sizeof(float));
    if (!outer_prod) { fprintf(stderr,"error in bilinear_cblas_s: problem with malloc. "); perror("malloc"); return 1; }

    for (size_t l=L; l>0u; --l, X1+=Ni1, X2+=Ni2, B-=No, W-=Nw)
    {
        //Re-zero outer_prod
        for (size_t n=N12; n>0u; --n) { outer_prod[n] = 0.0f; }

        //Outer product of X1, X2
        cblas_sger(CblasRowMajor,(int)Ni1,(int)Ni2,1.0f,X1,1,X2,1,outer_prod,(int)Ni2);

        //Weight by W for each output
        for (size_t o=No; o>0u; --o, ++B, ++Y, W+=N12)
        {
            *Y = *B + cblas_sdot((int)N12,W,1,outer_prod,1);
            //*Y = cblas_sdsdot((int)N12,*B,W,1,outer_prod,1);
        }
    }

    free(outer_prod);

    return 0;
}


int bilinear_cblas_d (double *Y, const double *X1, const double *X2, const double *W, const double *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L)
{
    const size_t N12 = Ni1*Ni2, Nw = N12*No;

    double *outer_prod;
    outer_prod = (double *)malloc(N12*sizeof(double));
    if (!outer_prod) { fprintf(stderr,"error in bilinear_cblas_d: problem with malloc. "); perror("malloc"); return 1; }

    for (size_t l=L; l>0u; --l, X1+=Ni1, X2+=Ni2, B-=No, W-=Nw)
    {
        //Re-zero outer_prod
        for (size_t n=N12; n>0u; --n) { outer_prod[n] = 0.0; }

        //Outer product of X1, X2
        cblas_dger(CblasRowMajor,(int)Ni1,(int)Ni2,1.0,X1,1,X2,1,outer_prod,(int)Ni2);

        //Weight by W for each output
        for (size_t o=No; o>0u; --o, ++B, ++Y, W+=N12)
        {
            *Y = *B + cblas_ddot((int)N12,W,1,outer_prod,1);
        }
    }

    free(outer_prod);

    return 0;
}


int bilinear_cblas_c (float *Y, const float *X1, const float *X2, const float *W, const float *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L)
{
    const size_t N12 = 2u*Ni1*Ni2, Nw = N12*No;
    const float oz[2] = {1.0f,0.0f};

    float *outer_prod;
    outer_prod = (float *)malloc(N12*sizeof(float));
    if (!outer_prod) { fprintf(stderr,"error in bilinear_cblas_c: problem with malloc. "); perror("malloc"); return 1; }

    for (size_t l=L; l>0u; --l, X1+=2u*Ni1, X2+=2u*Ni2, B-=2u*No, W-=Nw)
    {
        //Re-zero outer_prod
        for (size_t n=N12; n>0u; --n) { outer_prod[n] = 0.0f; }

        //Outer product of X1, X2
        cblas_cgeru(CblasRowMajor,(int)Ni1,(int)Ni2,oz,X1,1,X2,1,outer_prod,(int)Ni2);

        //Weight by W for each output
        for (size_t o=No; o>0u; --o, W+=N12)
        {
            cblas_cdotu_sub((int)N12,W,1,outer_prod,1,Y);
            *Y++ += *B++; *Y++ += *B++;
        }
    }

    free(outer_prod);

    return 0;
}


int bilinear_cblas_z (double *Y, const double *X1, const double *X2, const double *W, const double *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L)
{
    const size_t N12 = 2u*Ni1*Ni2, Nw = N12*No;
    const double oz[2] = {1.0,0.0};

    double *outer_prod;
    outer_prod = (double *)malloc(N12*sizeof(double));
    if (!outer_prod) { fprintf(stderr,"error in bilinear_cblas_z: problem with malloc. "); perror("malloc"); return 1; }

    for (size_t l=L; l>0u; --l, X1+=2u*Ni1, X2+=2u*Ni2, B-=2u*No, W-=Nw)
    {
        //Re-zero outer_prod
        for (size_t n=N12; n>0u; --n) { outer_prod[n] = 0.0; }

        //Outer product of X1, X2
        cblas_zgeru(CblasRowMajor,(int)Ni1,(int)Ni2,oz,X1,1,X2,1,outer_prod,(int)Ni2);

        //Weight by W for each output
        for (size_t o=No; o>0u; --o, W+=N12)
        {
            cblas_zdotu_sub((int)N12,W,1,outer_prod,1,Y);
            *Y++ += *B++; *Y++ += *B++;
        }
    }

    free(outer_prod);

    return 0;
}


#ifdef __cplusplus
}
}
#endif
