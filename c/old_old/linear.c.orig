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
<<<<<<< HEAD
    const size_t Nw = Ni*No;
    float sm;

    for (size_t l=L; l>0u; --l, W-=Nw, X+=Ni)
=======
    const size_t N = R*C*S*H;
    const size_t Lx = (dim==0u) ? R : (dim==1u) ? C : (dim==2u) ? S : H;
    const size_t Nw = Lx*Ly;
    float sm2;

    if (N==0u) {}
    else if (Lx==N)
>>>>>>> 2a96d2fcaa1a48d4fa925a3955e8b99860ee5123
    {
        for (size_t o=No; o>0u; --o, X-=Ni, ++Y)
        {
            sm = 0.0f;
            for (size_t i=Ni; i>0u; --i, ++X, ++W)
            {
<<<<<<< HEAD
                sm += *X * *W;
=======
                sm2 = 0.0f;
                for (size_t lx=0; lx<Lx; ++lx, ++X, ++W) { sm2 = fmaf(*X,*W,sm2); }
                *Y = sm2;
            }
        }
        else
        {
            cblas_sgemv(CblasColMajor,CblasTrans,(int)Lx,(int)Ly,1.0f,W,(int)Lx,X,1,0.0f,Y,1);
        }
    }
    else
    {
        const size_t K = (iscolmajor) ? ((dim==0u) ? 1u : (dim==1u) ? R : (dim==2u) ? R*C : R*C*S) : ((dim==0u) ? C*S*H : (dim==1u) ? S*H : (dim==2u) ? H : 1u);
        const size_t B = (iscolmajor && dim==0u) ? C*S*H : K;
        const size_t V = N/Lx, G = V/B;

        if (K==1u && (G==1u || B==1u))
        {
            if (Nw<4500)
            {
                for (size_t v=0; v<V; ++v, W-=Nw)
                {
                    for (size_t ly=0; ly<Ly; ++ly, ++Y)
                    {
                        sm2 = 0.0f;
                        for (size_t lx=0; lx<Lx; ++lx, ++X, ++W) { sm2 = fmaf(*X,*W,sm2); }
                        *Y = sm2;
                        if (ly<Ly-1) { X -= Lx; }
                    }
                }
            }
            else
            {
                cblas_sgemm(CblasColMajor,CblasTrans,CblasNoTrans,(int)Ly,(int)V,(int)Lx,1.0f,W,(int)Lx,X,(int)Lx,0.0f,Y,(int)Ly);
            }
        }
        else
        {
            for (size_t g=G; g>0u; --g, X+=B*(Lx-1), Y+=B*(Ly-1))
            {
                for (size_t b=B; b>0u; --b, ++X, W-=Nw, Y-=K*Ly-1)
                {
                    for (size_t ly=0; ly<Ly; ++ly, X-=K*Lx, Y+=K)
                    {
                        sm2 = 0.0f;
                        for (size_t lx=0; lx<Lx; ++lx, X+=K, ++W) { sm2 = fmaf(*X,*W,sm2); }
                        *Y = sm2;
                    }
                }
>>>>>>> 2a96d2fcaa1a48d4fa925a3955e8b99860ee5123
            }
            *Y = sm;
        }
    }

    return 0;
}


int linear_d (double *Y, const double *X, const double *W, const size_t Ni, const size_t No, const size_t L)
{
<<<<<<< HEAD
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
=======
    const size_t N = R*C*S*H;
    const size_t Lx = (dim==0u) ? R : (dim==1u) ? C : (dim==2u) ? S : H;
    const size_t Nw = Lx*Ly;
    double sm2;

    if (N==0u) {}
    else if (Lx==N)
    {
        if (Nw<30000)
        {
            for (size_t ly=0; ly<Ly; ++ly, X-=Lx, ++Y)
            {
                sm2 = 0.0;
                for (size_t lx=0; lx<Lx; ++lx, ++X, ++W) { sm2 = fma(*X,*W,sm2); }
                *Y = sm2;
            }
        }
        else
        {
            cblas_dgemv(CblasColMajor,CblasTrans,(int)Lx,(int)Ly,1.0,W,(int)Lx,X,1,0.0,Y,1);
        }
    }
    else
    {
        const size_t K = (iscolmajor) ? ((dim==0u) ? 1u : (dim==1u) ? R : (dim==2u) ? R*C : R*C*S) : ((dim==0u) ? C*S*H : (dim==1u) ? S*H : (dim==2u) ? H : 1u);
        const size_t B = (iscolmajor && dim==0u) ? C*S*H : K;
        const size_t V = N/Lx, G = V/B;

        if (K==1u && (G==1u || B==1u))
        {
            if (Nw<4500)
            {
                for (size_t v=0; v<V; ++v, W-=Nw)
                {
                    for (size_t ly=0; ly<Ly; ++ly, ++Y)
                    {
                        sm2 = 0.0;
                        for (size_t lx=0; lx<Lx; ++lx, ++X, ++W) { sm2 = fma(*X,*W,sm2); }
                        *Y = sm2;
                        if (ly<Ly-1) { X -= Lx; }
                    }
                }
            }
            else
            {
                cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,(int)Ly,(int)V,(int)Lx,1.0,W,(int)Lx,X,(int)Lx,0.0,Y,(int)Ly);
            }
        }
        else
        {
            for (size_t g=G; g>0u; --g, X+=B*(Lx-1), Y+=B*(Ly-1))
            {
                for (size_t b=B; b>0u; --b, ++X, W-=Nw, Y-=K*Ly-1)
                {
                    for (size_t ly=0; ly<Ly; ++ly, X-=K*Lx, Y+=K)
                    {
                        sm2 = 0.0;
                        for (size_t lx=0; lx<Lx; ++lx, X+=K, ++W) { sm2 = fma(*X,*W,sm2); }
                        *Y = sm2;
                    }
                }
>>>>>>> 2a96d2fcaa1a48d4fa925a3955e8b99860ee5123
            }
            *Y = sm;
        }
    }

    return 0;
}


int linear_c (float *Y, const float *X, const float *W, const size_t Ni, const size_t No, const size_t L)
{
<<<<<<< HEAD
    const size_t Nw = Ni*No;
    float smr, smi;

    for (size_t l=L; l>0u; --l, W-=2u*Nw, X+=2u*Ni)
=======
    const size_t N = R*C*S*H;
    const size_t Lx = (dim==0u) ? R : (dim==1u) ? C : (dim==2u) ? S : H;
    const size_t Nw = Lx*Ly;
    float sm2r, sm2i, xr, xi, ar, ai;

    if (N==0u) {}
    else if (Lx==N)
>>>>>>> 2a96d2fcaa1a48d4fa925a3955e8b99860ee5123
    {
        for (size_t o=No; o>0u; --o, X-=2u*Ni)
        {
            smr = smi = 0.0f;
            for (size_t i=Ni; i>0u; --i, X+=2, W+=2)
            {
<<<<<<< HEAD
                smr += *X**W - *(X+1)**(W+1);
                smi += *X**(W+1) + *(X+1)**W;
=======
                sm2r = sm2i = 0.0f;
                for (size_t lx=0; lx<Lx; ++lx, ++X, ++W)
                {
                    xr = *X; xi = *++X;
                    ar = *W; ai = *++W;
                    sm2r += xr*ar - xi*ai;
                    sm2i += xr*ai + xi*ar;
                }
                *Y = sm2r; *++Y = sm2i;
            }
        }
        else
        {
            const float z[2] = {0.0f,0.0f}, o[2] = {1.0f,0.0f};
            cblas_cgemv(CblasColMajor,CblasTrans,(int)Lx,(int)Ly,o,W,(int)Lx,X,1,z,Y,1);
        }
    }
    else
    {
        const size_t K = (iscolmajor) ? ((dim==0u) ? 1u : (dim==1u) ? R : (dim==2u) ? R*C : R*C*S) : ((dim==0u) ? C*S*H : (dim==1u) ? S*H : (dim==2u) ? H : 1u);
        const size_t B = (iscolmajor && dim==0u) ? C*S*H : K;
        const size_t V = N/Lx, G = V/B;

        if (K==1u && (G==1u || B==1u))
        {
            if (Nw<4500)
            {
                for (size_t v=0; v<V; ++v, W-=2*Nw)
                {
                    for (size_t ly=0; ly<Ly; ++ly, ++Y)
                    {
                        sm2r = sm2i = 0.0f;
                        for (size_t lx=0; lx<Lx; ++lx, ++X, ++W)
                        {
                            xr = *X; xi = *++X;
                            ar = *W; ai = *++W;
                            sm2r += xr*ar - xi*ai;
                            sm2i += xr*ai + xi*ar;
                        }
                        *Y = sm2r; *++Y = sm2i;
                        if (ly<Ly-1) { X -= 2*Lx; }
                    }
                }
            }
            else
            {
                const float z[2] = {0.0f,0.0f}, o[2] = {1.0f,0.0f};
                cblas_cgemm(CblasColMajor,CblasTrans,CblasNoTrans,(int)Ly,(int)V,(int)Lx,o,W,(int)Lx,X,(int)Lx,z,Y,(int)Ly);
            }
        }
        else
        {
            for (size_t g=G; g>0u; --g, X+=2*B*(Lx-1), Y+=2*B*(Ly-1))
            {
                for (size_t b=B; b>0u; --b, ++X, W-=2*Nw, Y-=2*K*Ly-2)
                {
                    for (size_t ly=0; ly<Ly; ++ly, X-=2*K*Lx, Y+=2*K-1)
                    {
                        sm2r = sm2i = 0.0f;
                        for (size_t lx=0; lx<Lx; ++lx, X+=2*K-1, ++W)
                        {
                            xr = *X; xi = *++X;
                            ar = *W; ai = *++W;
                            sm2r += xr*ar - xi*ai;
                            sm2i += xr*ai + xi*ar;
                        }
                        *Y = sm2r; *++Y = sm2i;
                    }
                }
>>>>>>> 2a96d2fcaa1a48d4fa925a3955e8b99860ee5123
            }
            *Y++ = smr; *Y++ = smi;
        }
    }

    return 0;
}


int linear_z (double *Y, const double *X, const double *W, const size_t Ni, const size_t No, const size_t L)
{
<<<<<<< HEAD
    const size_t Nw = Ni*No;
    double smr, smi;

    for (size_t l=L; l>0u; --l, W-=2u*Nw, X+=2u*Ni)
    {
        for (size_t o=No; o>0u; --o, X-=2u*Ni)
=======
    const size_t N = R*C*S*H;
    const size_t Lx = (dim==0u) ? R : (dim==1u) ? C : (dim==2u) ? S : H;
    const size_t Nw = Lx*Ly;
    double sm2r, sm2i, xr, xi, ar, ai;

    if (N==0u) {}
    else if (Lx==N)
    {
        if (Nw<150)
        {
            for (size_t ly=0; ly<Ly; ++ly, X-=2*Lx, ++Y)
            {
                sm2r = sm2i = 0.0;
                for (size_t lx=0; lx<Lx; ++lx, ++X, ++W)
                {
                    xr = *X; xi = *++X;
                    ar = *W; ai = *++W;
                    sm2r += xr*ar - xi*ai;
                    sm2i += xr*ai + xi*ar;
                }
                *Y = sm2r; *++Y = sm2i;
            }
        }
        else
        {
            const double z[2] = {0.0,0.0}, o[2] = {1.0,0.0};
            cblas_zgemv(CblasColMajor,CblasTrans,(int)Lx,(int)Ly,o,W,(int)Lx,X,1,z,Y,1);
        }
    }
    else
    {
        const size_t K = (iscolmajor) ? ((dim==0u) ? 1u : (dim==1u) ? R : (dim==2u) ? R*C : R*C*S) : ((dim==0u) ? C*S*H : (dim==1u) ? S*H : (dim==2u) ? H : 1u);
        const size_t B = (iscolmajor && dim==0u) ? C*S*H : K;
        const size_t V = N/Lx, G = V/B;

        if (K==1u && (G==1u || B==1u))
>>>>>>> 2a96d2fcaa1a48d4fa925a3955e8b99860ee5123
        {
            smr = smi = 0.0;
            for (size_t i=Ni; i>0u; --i, X+=2, W+=2)
            {
<<<<<<< HEAD
                smr += *X**W - *(X+1)**(W+1);
                smi += *X**(W+1) + *(X+1)**W;
=======
                for (size_t v=0; v<V; ++v, W-=2*Nw)
                {
                    for (size_t ly=0; ly<Ly; ++ly, ++Y)
                    {
                        sm2r = sm2i = 0.0;
                        for (size_t lx=0; lx<Lx; ++lx, ++X, ++W)
                        {
                            xr = *X; xi = *++X;
                            ar = *W; ai = *++W;
                            sm2r += xr*ar - xi*ai;
                            sm2i += xr*ai + xi*ar;
                        }
                        *Y = sm2r; *++Y = sm2i;
                        if (ly<Ly-1) { X -= 2*Lx; }
                    }
                }
            }
            else
            {
                const double z[2] = {0.0,0.0}, o[2] = {1.0,0.0};
                cblas_zgemm(CblasColMajor,CblasTrans,CblasNoTrans,(int)Ly,(int)V,(int)Lx,o,W,(int)Lx,X,(int)Lx,z,Y,(int)Ly);
            }
        }
        else
        {
            for (size_t g=G; g>0u; --g, X+=2*B*(Lx-1), Y+=2*B*(Ly-1))
            {
                for (size_t b=B; b>0u; --b, ++X, W-=2*Nw, Y-=2*K*Ly-2)
                {
                    for (size_t ly=0; ly<Ly; ++ly, X-=2*K*Lx, Y+=2*K-1)
                    {
                        sm2r = sm2i = 0.0;
                        for (size_t lx=0; lx<Lx; ++lx, X+=2*K-1, ++W)
                        {
                            xr = *X; xi = *++X;
                            ar = *W; ai = *++W;
                            sm2r += xr*ar - xi*ai;
                            sm2i += xr*ai + xi*ar;
                        }
                        *Y = sm2r; *++Y = sm2i;
                    }
                }
>>>>>>> 2a96d2fcaa1a48d4fa925a3955e8b99860ee5123
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
