//IN method.
//Bilinear transformation (3D tensor of weights, 1D vector of biases)
//of Ni1 inputs from X1 and Ni2 inputs from X2 to No outputs in Y.

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

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int bilinear_s (float *Y, const float *X1, const float *X2, const float *W, const float *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L);
int bilinear_d (double *Y, const double *X1, const double *X2, const double *W, const double *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L);
int bilinear_c (float *Y, const float *X1, const float *X2, const float *W, const float *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L);
int bilinear_z (double *Y, const double *X1, const double *X2, const double *W, const double *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L);


int bilinear_s (float *Y, const float *X1, const float *X2, const float *W, const float *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L)
{
    const size_t N12 = Ni1*Ni2, Nw = N12*No;

    float sm, *outer_prod;
    outer_prod = (float *)malloc(N12*sizeof(float));
    if (!outer_prod) { fprintf(stderr,"error in bilinear_s: problem with malloc. "); perror("malloc"); return 1; }

    for (size_t l=L; l>0u; --l, X2+=Ni2, B-=No, W-=Nw)
    {
        //Re-zero outer_prod
        for (size_t n=N12; n>0u; --n) { outer_prod[n] = 0.0f; }

        //Outer product of X1, X2
        for (size_t i=Ni1; i>0u; --i, ++X1, X2-=Ni2)
        {
            for (size_t j=Ni2; j>0u; --j, ++X2, ++outer_prod)
            {
                *outer_prod = *X1 * *X2;
            }
        }
        outer_prod -= N12;

        //Weight by W for each output
        for (size_t o=No; o>0u; --o, ++B, ++Y, outer_prod-=N12)
        {
            sm = *B;
            for (size_t n=N12; n>0u; --n, ++W, ++outer_prod)
            {
                sm += *W * *outer_prod;
            }
            *Y = sm;
        }
    }

    free(outer_prod);

    return 0;
}


int bilinear_d (double *Y, const double *X1, const double *X2, const double *W, const double *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L)
{
    const size_t N12 = Ni1*Ni2, Nw = N12*No;

    double sm, *outer_prod;
    outer_prod = (double *)malloc(N12*sizeof(double));
    if (!outer_prod) { fprintf(stderr,"error in bilinear_d: problem with malloc. "); perror("malloc"); return 1; }

    for (size_t l=L; l>0u; --l, X2+=Ni2, B-=No, W-=Nw)
    {
        //Re-zero outer_prod
        for (size_t n=N12; n>0u; --n) { outer_prod[n] = 0.0; }

        //Outer product of X1, X2
        for (size_t i=Ni1; i>0u; --i, ++X1, X2-=Ni2)
        {
            for (size_t j=Ni2; j>0u; --j, ++X2, ++outer_prod)
            {
                *outer_prod = *X1 * *X2;
            }
        }
        outer_prod -= N12;

        //Weight by W for each output
        for (size_t o=No; o>0u; --o, ++B, ++Y, outer_prod-=N12)
        {
            sm = *B;
            for (size_t n=N12; n>0u; --n, ++W, ++outer_prod)
            {
                sm += *W * *outer_prod;
            }
            *Y = sm;
        }
    }

    free(outer_prod);

    return 0;
}


int bilinear_c (float *Y, const float *X1, const float *X2, const float *W, const float *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L)
{
    const size_t N12 = Ni1*Ni2, Nw = N12*No;

    float smr, smi, *outer_prod;
    outer_prod = (float *)malloc(2u*N12*sizeof(float));
    if (!outer_prod) { fprintf(stderr,"error in bilinear_c: problem with malloc. "); perror("malloc"); return 1; }

    for (size_t l=L; l>0u; --l, X2+=2u*Ni2, B-=2u*No, W-=2u*Nw)
    {
        //Re-zero outer_prod
        for (size_t n=2u*N12; n>0u; --n) { outer_prod[n] = 0.0f; }

        //Outer product of X1, X2
        for (size_t i=Ni1; i>0u; --i, X1+=2, X2-=2u*Ni2)
        {
            for (size_t j=Ni2; j>0u; --j, X2+=2)
            {
                *outer_prod++ = *X1**X2 - *(X1+1)**(X2+1);
                *outer_prod++ = *X1**(X2+1) + *(X1+1)**X2;
            }
        }
        outer_prod -= 2u*N12;

        //Weight by W for each output
        for (size_t o=No; o>0u; --o, outer_prod-=2u*N12)
        {
            smr = *B++; smi = *B++;
            for (size_t n=N12; n>0u; --n, W+=2, outer_prod+=2)
            {
                smr += *W**outer_prod - *(W+1)**(outer_prod+1);
                smi += *W**(outer_prod+1) + *(W+1)**outer_prod;
            }
            *Y++ = smr; *Y++ = smi;
        }
    }

    free(outer_prod);

    return 0;
}


int bilinear_z (double *Y, const double *X1, const double *X2, const double *W, const double *B, const size_t Ni1, const size_t Ni2, const size_t No, const size_t L)
{
    const size_t N12 = Ni1*Ni2, Nw = N12*No;

    double smr, smi, *outer_prod;
    outer_prod = (double *)malloc(2u*N12*sizeof(double));
    if (!outer_prod) { fprintf(stderr,"error in bilinear_z: problem with malloc. "); perror("malloc"); return 1; }

    for (size_t l=L; l>0u; --l, X2+=2u*Ni2, B-=2u*No, W-=2u*Nw)
    {
        //Re-zero outer_prod
        for (size_t n=2u*N12; n>0u; --n) { outer_prod[n] = 0.0; }

        //Outer product of X1, X2
        for (size_t i=Ni1; i>0u; --i, X1+=2, X2-=2u*Ni2)
        {
            for (size_t j=Ni2; j>0u; --j, X2+=2)
            {
                *outer_prod++ = *X1**X2 - *(X1+1)**(X2+1);
                *outer_prod++ = *X1**(X2+1) + *(X1+1)**X2;
            }
        }
        outer_prod -= 2u*N12;

        //Weight by W for each output
        for (size_t o=No; o>0u; --o, outer_prod-=2u*N12)
        {
            smr = *B++; smi = *B++;
            for (size_t n=N12; n>0u; --n, W+=2, outer_prod+=2)
            {
                smr += *W**outer_prod - *(W+1)**(outer_prod+1);
                smi += *W**(outer_prod+1) + *(W+1)**outer_prod;
            }
            *Y++ = smr; *Y++ = smi;
        }
    }

    free(outer_prod);

    return 0;
}


#ifdef __cplusplus
}
}
#endif
