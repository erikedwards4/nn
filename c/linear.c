//IN method.
//Linear transformation of each vector in X using matrix W.
//W transforms Ni inputs to No outputs.

//This is different overall setup than usual matmul2: X can be 4D tensor,
//with the vecs along any dimension, and each is transformed (vec2vec operation).
//Thus, the code is much like the code for other vec2vec operations.

//Each vector in X has length Lx (Ni), and each vector in Y has length Ly (No).
//If colmajor, then W has size Ni x No, and y = W'*x for each col vec x.
//If rowmajor, then W has size No x Ni, and y = W*x for each col vec x.
//Or:
//If colmajor, then W has size Ni x No, and y = x*W for each row vec x.
//If rowmajor, then W has size No x Ni, and y = x*W' for each row vec x.
//That is:
//For performance reasons, this assumes that W has leading dimension Ni!

#include <stdio.h>
#include <math.h>
#include <cblas.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int linear_s (float *Y, const float *X, const float *W, const size_t R, const size_t C, const size_t S, const size_t H, const size_t Ly, const char iscolmajor, const size_t dim);
int linear_d (double *Y, const double *X, const double *W, const size_t R, const size_t C, const size_t S, const size_t H, const size_t Ly, const char iscolmajor, const size_t dim);
int linear_c (float *Y, const float *X, const float *W, const size_t R, const size_t C, const size_t S, const size_t H, const size_t Ly, const char iscolmajor, const size_t dim);
int linear_z (double *Y, const double *X, const double *W, const size_t R, const size_t C, const size_t S, const size_t H, const size_t Ly, const char iscolmajor, const size_t dim);


int linear_s (float *Y, const float *X, const float *W, const size_t R, const size_t C, const size_t S, const size_t H, const size_t Ly, const char iscolmajor, const size_t dim)
{
    const size_t N = R*C*S*H;
    const size_t Lx = (dim==0u) ? R : (dim==1u) ? C : (dim==2u) ? S : H;
    const size_t Nw = Lx*Ly;
    float sm2;

    if (N==0u) {}
    else if (Lx==N)
    {
        if (Nw<30000)
        {
            for (size_t ly=0; ly<Ly; ++ly, X-=Lx, ++Y)
            {
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
            }
        }
    }

    return 0;
}


int linear_d (double *Y, const double *X, const double *W, const size_t R, const size_t C, const size_t S, const size_t H, const size_t Ly, const char iscolmajor, const size_t dim)
{
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
            }
        }
    }

    return 0;
}


int linear_c (float *Y, const float *X, const float *W, const size_t R, const size_t C, const size_t S, const size_t H, const size_t Ly, const char iscolmajor, const size_t dim)
{
    const size_t N = R*C*S*H;
    const size_t Lx = (dim==0u) ? R : (dim==1u) ? C : (dim==2u) ? S : H;
    const size_t Nw = Lx*Ly;
    float sm2r, sm2i, xr, xi, ar, ai;

    if (N==0u) {}
    else if (Lx==N)
    {
        if (Nw<30000)
        {
            for (size_t ly=0; ly<Ly; ++ly, X-=2*Lx, ++Y)
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
            }
        }
    }

    return 0;
}


int linear_z (double *Y, const double *X, const double *W, const size_t R, const size_t C, const size_t S, const size_t H, const size_t Ly, const char iscolmajor, const size_t dim)
{
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
        {
            if (Nw<4500)
            {
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
            }
        }
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
