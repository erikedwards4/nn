//IN method.
//Linear transformation (weights only, no biases) of Ni inputs to No outputs.

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

//#include <omp.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int linear_s (float *Y, const float *X, const float *W, const size_t Ni, const size_t No, const size_t L);
int linear_d (double *Y, const double *X, const double *W, const size_t Ni, const size_t No, const size_t L);
int linear_c (float *Y, const float *X, const float *W, const size_t Ni, const size_t No, const size_t L);
int linear_z (double *Y, const double *X, const double *W, const size_t Ni, const size_t No, const size_t L);


int linear_s (float *Y, const float *X, const float *W, const size_t Ni, const size_t No, const size_t L)
{
    const size_t Nw = Ni*No;
    float sm;

    for (size_t l=L; l>0u; --l, W-=Nw, X+=Ni)
    {
        for (size_t o=No; o>0u; --o, X-=Ni, ++Y)
        {
            sm = 0.0f;
            for (size_t i=Ni; i>0u; --i, ++X, ++W)
            {
                sm += *X * *W;
            }
            *Y = sm;
        }
    }

    return 0;
}


int linear_d (double *Y, const double *X, const double *W, const size_t Ni, const size_t No, const size_t L)
{
    const size_t Nw = Ni*No;
    double sm;

    for (size_t l=L; l>0u; --l, W-=Nw, X+=Ni)
    {
        for (size_t o=No; o>0u; --o, X-=Ni, ++Y)
        {
            sm = 0.0;
            for (size_t i=Ni; i>0u; --i, ++X, ++W)
            {
                sm += *X * *W;
            }
            *Y = sm;
        }
    }

    return 0;
}


int linear_c (float *Y, const float *X, const float *W, const size_t Ni, const size_t No, const size_t L)
{
    const size_t Nw = Ni*No;
    float smr, smi;

    for (size_t l=L; l>0u; --l, W-=2u*Nw, X+=2u*Ni)
    {
        for (size_t o=No; o>0u; --o, X-=2u*Ni)
        {
            smr = smi = 0.0f;
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


int linear_z (double *Y, const double *X, const double *W, const size_t Ni, const size_t No, const size_t L)
{
    const size_t Nw = Ni*No;
    double smr, smi;

    for (size_t l=L; l>0u; --l, W-=2u*Nw, X+=2u*Ni)
    {
        for (size_t o=No; o>0u; --o, X-=2u*Ni)
        {
            smr = smi = 0.0;
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
// int linear_omp_s (float *Y, const float *X, const float *W, const size_t Ni, const size_t No, const size_t L)
// {
//     for (size_t l=L; l>0u; --l)
//     {
//         #pragma omp parallel for
//         for (size_t o=No; o>0u; --o)
//         {
//             float sm = 0.0f;
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
