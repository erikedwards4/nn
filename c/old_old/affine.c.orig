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
<<<<<<< HEAD
    const size_t Nw = Ni*No;
    float sm;

    for (size_t l=L; l>0u; --l, B-=No, W-=Nw, X+=Ni)
    {
        for (size_t o=No; o>0u; --o, X-=Ni, ++B, ++Y)
        {
            sm = *B;
            for (size_t i=Ni; i>0u; --i, ++X, ++W)
=======
    const size_t N = R*C*S*H;
    const size_t Lx = (dim==0u) ? R : (dim==1u) ? C : (dim==2u) ? S : H;
    const size_t Nw = Lx*Ly;
    float sm2;

    if (N==0u) {}
    else if (Lx==N)
    {
        if (Nw<30000)
        {
            for (size_t ly=0; ly<Ly; ++ly, X-=Lx, ++B, ++Y)
            {
                sm2 = *B;
                for (size_t lx=0; lx<Lx; ++lx, ++X, ++W) { sm2 = fmaf(*X,*W,sm2); }
                *Y = sm2;
            }
        }
        else
        {
            for (size_t ly=0; ly<Ly; ++ly, ++B, ++Y) { *Y = *B; }
            Y -= Ly;
            cblas_sgemv(CblasColMajor,CblasTrans,(int)Lx,(int)Ly,1.0f,W,(int)Lx,X,1,1.0f,Y,1);
        }
    }
    else
    {
        const size_t K = (iscolmajor) ? ((dim==0u) ? 1u : (dim==1u) ? R : (dim==2u) ? R*C : R*C*S) : ((dim==0u) ? C*S*H : (dim==1u) ? S*H : (dim==2u) ? H : 1u);
        const size_t BS = (iscolmajor && dim==0u) ? C*S*H : K;
        const size_t V = N/Lx, G = V/BS;

        if (K==1u && (G==1u || BS==1))
        {
            if (Nw<1000)
            {
                for (size_t v=0; v<V; ++v, W-=Lx*Ly, B-=Ly)
                {
                    for (size_t ly=0; ly<Ly; ++ly, ++B, ++Y)
                    {
                        sm2 = *B;
                        for (size_t lx=0; lx<Lx; ++lx, ++X, ++W) { sm2 = fmaf(*X,*W,sm2); }
                        *Y = sm2;
                        if (ly<Ly-1) { X -= Lx; }
                    }
                }
            }
            else
            {
                for (size_t v=0; v<V; ++v, X+=Lx, B-=Ly, Y+=Ly)
                {
                    for (size_t ly=0; ly<Ly; ++ly, ++B, ++Y) { *Y = *B; }
                    Y -= Ly;
                    cblas_sgemv(CblasColMajor,CblasTrans,(int)Lx,(int)Ly,1.0f,W,(int)Lx,X,1,1.0f,Y,1);
                }
            }
        }
        else
        {
            for (size_t g=G; g>0u; --g, X+=BS*(Lx-1), Y+=BS*(Ly-1))
>>>>>>> 2a96d2fcaa1a48d4fa925a3955e8b99860ee5123
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
<<<<<<< HEAD
    const size_t Nw = Ni*No;
    double sm;

    for (size_t l=L; l>0u; --l, B-=No, W-=Nw, X+=Ni)
    {
        for (size_t o=No; o>0u; --o, X-=Ni, ++B, ++Y)
        {
            sm = *B;
            for (size_t i=Ni; i>0u; --i, ++X, ++W)
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
            for (size_t ly=0; ly<Ly; ++ly, X-=Lx, ++B, ++Y)
            {
                sm2 = *B;
                for (size_t lx=0; lx<Lx; ++lx, ++X, ++W) { sm2 = fma(*X,*W,sm2); }
                *Y = sm2;
            }
        }
        else
        {
            for (size_t ly=0; ly<Ly; ++ly, ++B, ++Y) { *Y = *B; }
            Y -= Ly;
            cblas_dgemv(CblasColMajor,CblasTrans,(int)Lx,(int)Ly,1.0,W,(int)Lx,X,1,1.0,Y,1);
        }
    }
    else
    {
        const size_t K = (iscolmajor) ? ((dim==0u) ? 1u : (dim==1u) ? R : (dim==2u) ? R*C : R*C*S) : ((dim==0u) ? C*S*H : (dim==1u) ? S*H : (dim==2u) ? H : 1u);
        const size_t BS = (iscolmajor && dim==0u) ? C*S*H : K;
        const size_t V = N/Lx, G = V/BS;

        if (K==1u && (G==1u || BS==1))
        {
            if (Nw<1000)
            {
                for (size_t v=0; v<V; ++v, W-=Lx*Ly, B-=Ly)
                {
                    for (size_t ly=0; ly<Ly; ++ly, ++B, ++Y)
                    {
                        sm2 = *B;
                        for (size_t lx=0; lx<Lx; ++lx, ++X, ++W) { sm2 = fma(*X,*W,sm2); }
                        *Y = sm2;
                        if (ly<Ly-1) { X -= Lx; }
                    }
                }
            }
            else
            {
                for (size_t v=0; v<V; ++v, X+=Lx, B-=Ly, Y+=Ly)
                {
                    for (size_t ly=0; ly<Ly; ++ly, ++B, ++Y) { *Y = *B; }
                    Y -= Ly;
                    cblas_dgemv(CblasColMajor,CblasTrans,(int)Lx,(int)Ly,1.0,W,(int)Lx,X,1,1.0,Y,1);
                }
            }
        }
        else
        {
            for (size_t g=G; g>0u; --g, X+=BS*(Lx-1), Y+=BS*(Ly-1))
>>>>>>> 2a96d2fcaa1a48d4fa925a3955e8b99860ee5123
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
<<<<<<< HEAD
    const size_t Nw = Ni*No;
    float smr, smi;

    for (size_t l=L; l>0u; --l, B-=2u*No, W-=2u*Nw, X+=2u*Ni)
=======
    const size_t N = R*C*S*H;
    const size_t Lx = (dim==0u) ? R : (dim==1u) ? C : (dim==2u) ? S : H;
    const size_t Nw = Lx*Ly;
    float xr, xi, ar, ai, sm2r, sm2i;

    if (N==0u) {}
    else if (Lx==N)
>>>>>>> 2a96d2fcaa1a48d4fa925a3955e8b99860ee5123
    {
        for (size_t o=No; o>0u; --o, X-=2u*Ni)
        {
            smr = *B++; smi = *B++;
            for (size_t i=Ni; i>0u; --i, X+=2, W+=2)
            {
<<<<<<< HEAD
                smr += *X**W - *(X+1)**(W+1);
                smi += *X**(W+1) + *(X+1)**W;
=======
                sm2r = *B; sm2i = *++B;
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
            const float o[2] = {1.0f,0.0f};
            for (size_t ly=0; ly<2*Ly; ++ly, ++B, ++Y) { *Y = *B; }
            Y -= 2*Ly;
            cblas_cgemv(CblasColMajor,CblasTrans,(int)Lx,(int)Ly,o,W,(int)Lx,X,1,o,Y,1);
        }
    }
    else
    {
        const size_t K = (iscolmajor) ? ((dim==0u) ? 1u : (dim==1u) ? R : (dim==2u) ? R*C : R*C*S) : ((dim==0u) ? C*S*H : (dim==1u) ? S*H : (dim==2u) ? H : 1u);
        const size_t BS = (iscolmajor && dim==0u) ? C*S*H : K;
        const size_t V = N/Lx, G = V/BS;

        if (K==1u && (G==1u || BS==1))
        {
            if (Nw<4500)
            {
                for (size_t v=0; v<V; ++v, W-=2*Nw, B-=2*Ly)
                {
                    for (size_t ly=0; ly<Ly; ++ly, ++B, ++Y)
                    {
                        sm2r = *B; sm2i = *++B;
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
                const float o[2] = {1.0f,0.0f};
                for (size_t v=0; v<V; ++v, X+=2*Lx, B-=2*Ly, Y+=2*Ly)
                {
                    for (size_t ly=0; ly<2*Ly; ++ly, ++B, ++Y) { *Y = *B; }
                    Y -= 2*Ly;
                    cblas_cgemv(CblasColMajor,CblasTrans,(int)Lx,(int)Ly,o,W,(int)Lx,X,1,o,Y,1);
                }
            }
        }
        else
        {
            for (size_t g=G; g>0u; --g, X+=2*BS*(Lx-1), Y+=2*BS*(Ly-1))
            {
                for (size_t b=0; b<BS; ++b, X+=2, W-=2*Nw, B-=2*Ly, Y-=2*K*Ly-2)
                {
                    for (size_t ly=0; ly<Ly; ++ly, X-=2*K*Lx, ++B, Y+=2*K-1)
                    {
                        sm2r = *B; sm2i = *++B;
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


int affine_z (double *Y, const double *X, const double *W, const double *B, const size_t Ni, const size_t No, const size_t L)
{
<<<<<<< HEAD
    const size_t Nw = Ni*No;
    double smr, smi;

    for (size_t l=L; l>0u; --l, B-=2u*No, W-=2u*Nw, X+=2u*Ni)
    {
        for (size_t o=No; o>0u; --o, X-=2u*Ni)
=======
    const size_t N = R*C*S*H;
    const size_t Lx = (dim==0u) ? R : (dim==1u) ? C : (dim==2u) ? S : H;
    const size_t Nw = Lx*Ly;
    double xr, xi, ar, ai, sm2r, sm2i;

    if (N==0u) {}
    else if (Lx==N)
    {
        if (Lx<30000)
        {
            for (size_t ly=0; ly<Ly; ++ly, X-=2*Lx, ++B, ++Y)
            {
                sm2r = *B; sm2i = *++B;
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
            const double o[2] = {1.0,0.0};
            for (size_t ly=0; ly<2*Ly; ++ly, ++B, ++Y) { *Y = *B; }
            Y -= 2*Ly;
            cblas_zgemv(CblasColMajor,CblasTrans,(int)Lx,(int)Ly,o,W,(int)Lx,X,1,o,Y,1);
        }
    }
    else
    {
        const size_t K = (iscolmajor) ? ((dim==0u) ? 1u : (dim==1u) ? R : (dim==2u) ? R*C : R*C*S) : ((dim==0u) ? C*S*H : (dim==1u) ? S*H : (dim==2u) ? H : 1u);
        const size_t BS = (iscolmajor && dim==0u) ? C*S*H : K;
        const size_t V = N/Lx, G = V/BS;

        if (K==1u && (G==1u || BS==1))
>>>>>>> 2a96d2fcaa1a48d4fa925a3955e8b99860ee5123
        {
            smr = *B++; smi = *B++;
            for (size_t i=Ni; i>0u; --i, X+=2, W+=2)
            {
<<<<<<< HEAD
                smr += *X**W - *(X+1)**(W+1);
                smi += *X**(W+1) + *(X+1)**W;
=======
                for (size_t v=0; v<V; ++v, W-=2*Nw, B-=2*Ly)
                {
                    for (size_t ly=0; ly<Ly; ++ly, ++B, ++Y)
                    {
                        sm2r = *B; sm2i = *++B;
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
                const double o[2] = {1.0,0.0};
                for (size_t v=0; v<V; ++v, X+=2*Lx, B-=2*Ly, Y+=2*Ly)
                {
                    for (size_t ly=0; ly<2*Ly; ++ly, ++B, ++Y) { *Y = *B; }
                    Y -= 2*Ly;
                    cblas_zgemv(CblasColMajor,CblasTrans,(int)Lx,(int)Ly,o,W,(int)Lx,X,1,o,Y,1);
                }
            }
        }
        else
        {
            for (size_t g=G; g>0u; --g, X+=2*BS*(Lx-1), Y+=2*BS*(Ly-1))
            {
                for (size_t b=0; b<BS; ++b, X+=2, W-=2*Nw, B-=2*Ly, Y-=2*K*Ly-2)
                {
                    for (size_t ly=0; ly<Ly; ++ly, X-=2*K*Lx, ++B, Y+=2*K-1)
                    {
                        sm2r = *B; sm2i = *++B;
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
