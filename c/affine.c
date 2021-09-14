//IN method.
//Affine transformation (weights and biases) of Ni inputs to No outputs.

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

//#include <omp.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int affine_s (float *Y, const float *X, const float *W, const float *B, const size_t Ni, const size_t No, const size_t L);
int affine_d (double *Y, const double *X, const double *W, const double *B, const size_t Ni, const size_t No, const size_t L);
int affine_c (float *Y, const float *X, const float *W, const float *B, const size_t Ni, const size_t No, const size_t L);
int affine_z (double *Y, const double *X, const double *W, const double *B, const size_t Ni, const size_t No, const size_t L);


int affine_s (float *Y, const float *X, const float *W, const float *B, const size_t Ni, const size_t No, const size_t L)
{
    const size_t Nw = Ni*No;
    float sm;

    for (size_t l=L; l>0u; --l, B-=No, W-=Nw, X+=Ni)
    {
        for (size_t o=No; o>0u; --o, X-=Ni, ++B, ++Y)
        {
            sm = *B;
            for (size_t i=Ni; i>0u; --i, ++X, ++W)
            {
                sm += *X * *W;
            }
            *Y = sm;
        }
    }

    return 0;
}


int affine_d (double *Y, const double *X, const double *W, const double *B, const size_t Ni, const size_t No, const size_t L)
{
    const size_t Nw = Ni*No;
    double sm;

    for (size_t l=L; l>0u; --l, B-=No, W-=Nw, X+=Ni)
    {
        for (size_t o=No; o>0u; --o, X-=Ni, ++B, ++Y)
        {
            sm = *B;
            for (size_t i=Ni; i>0u; --i, ++X, ++W)
            {
                sm += *X * *W;
            }
            *Y = sm;
        }
    }

    return 0;
}


int affine_c (float *Y, const float *X, const float *W, const float *B, const size_t Ni, const size_t No, const size_t L)
{
    const size_t Nw = Ni*No;
    float smr, smi;

    for (size_t l=L; l>0u; --l, B-=2u*No, W-=2u*Nw, X+=2u*Ni)
    {
        for (size_t o=No; o>0u; --o, X-=2u*Ni)
        {
            smr = *B++; smi = *B++;
            for (size_t i=Ni; i>0u; --i, X+=2, W+=2)
            {
                smr += *X**W - *(X+1)**(W+1);
                smi += *X**(W+1) + *(X+1)**W;
            }
            *Y++ = smr; *Y++ = smi;
        }
    }

    return 0;
}


int affine_z (double *Y, const double *X, const double *W, const double *B, const size_t Ni, const size_t No, const size_t L)
{
    const size_t Nw = Ni*No;
    double smr, smi;

    for (size_t l=L; l>0u; --l, B-=2u*No, W-=2u*Nw, X+=2u*Ni)
    {
        for (size_t o=No; o>0u; --o, X-=2u*Ni)
        {
            smr = *B++; smi = *B++;
            for (size_t i=Ni; i>0u; --i, X+=2, W+=2)
            {
                smr += *X**W - *(X+1)**(W+1);
                smi += *X**(W+1) + *(X+1)**W;
            }
            *Y++ = smr; *Y++ = smi;
        }
    }

    return 0;
}


//Although this compiles and runs, it does not give the right output
// int affine_omp_s (float *Y, const float *X, const float *W, const float *B, const size_t Ni, const size_t No, const size_t L)
// {
//     for (size_t l=L; l>0u; --l)
//     {
//         #pragma omp parallel for
//         for (size_t o=No; o>0u; --o)
//         {
//             float sm = B[o];
//             for (size_t i=Ni; i>0u; --i)
//             {
//                 sm += X[i+l*Ni] * W[i+o*Ni];
//             }
//             Y[o] = sm;
//         }
//     }

//     return 0;
// }


#ifdef __cplusplus
}
}
#endif
