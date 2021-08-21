//This gets maxout layer-wise activation function for X.
//This is used on the output of M separate affine transforms,
//each of which has output vector length of No (num output neurons).
//These are stacked into a single long vector of length M*No.
//For each of the No positions, the max over the M elements is taken.
//Thus, M is the number of inputs per output neuron,
//and each output neuron just takes a max.

//This is programmed as a vec2vec operation, with the vecs along dim.
//The output vecs have length Ly = No, and the input vecs have length Lx = M*Ly.

#include <stdio.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int maxout_s (float *Y, const float *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim, const size_t M);
int maxout_d (double *Y, const double *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim, const size_t M);


int maxout_s (float *Y, const float *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim, const size_t M)
{
    if (dim>3) { fprintf(stderr,"error in maxout_s: dim must be in [0 3]\n"); return 1; }

    const size_t N = R*C*S*H;
    const size_t Lx = (dim==0u) ? R : (dim==1u) ? C : (dim==2u) ? S : H;
    if (Lx%M) { fprintf(stderr,"error in maxout_s: each vector in X must have length that is multiple of M\n"); return 1; }
    const size_t Ly = Lx / M;
    float mx;

    if (N==0u) {}
    else if (Lx==1)
    {
        for (size_t n=0u; n<N; ++n, ++X, ++Y) { *Y = *X; }
    }
    else if (Lx==N)
    {
        for (size_t l=0u; l<Ly; ++l, X-=Lx-1, ++Y)
        {
            mx = *X; X += Ly;
            for (size_t m=1; m<M; ++m, X+=Ly) { if (*X>mx) { mx = *X; } }
            *Y = mx;
        }
    }
    else
    {
        const size_t K = (iscolmajor) ? ((dim==0u) ? 1u : (dim==1u) ? R : (dim==2u) ? R*C : R*C*S) : ((dim==0u) ? C*S*H : (dim==1u) ? S*H : (dim==2u) ? H : 1u);
        const size_t B = (iscolmajor && dim==0u) ? C*S*H : K;
        const size_t V = N/Lx, G = V/B;

        if (K==1u && (G==1u || B==1u))
        {
            for (size_t v=0; v<V; ++v, X-=Ly-1)
            {
                for (size_t l=0u; l<Ly; ++l, ++Y)
                {
                    mx = *X; X += Ly;
                    for (size_t m=1; m<M; ++m, X+=Ly) { if (*X>mx) { mx = *X; } }
                    *Y = mx;
                    if (l<Ly-1) { X -= Lx - 1; }
                }
            }
        }
        else
        {
            for (size_t g=G; g>0u; --g, X+=B*(Lx-1), Y+=B*(Ly-1))
            {
                for (size_t b=B; b>0u; --b, X-=K*Ly-1, Y-=K*Ly-1)
                {
                    for (size_t l=0u; l<Ly; ++l, X-=K*(Lx-1), Y+=K)
                    {
                        mx = *X; X += K*Ly;
                        for (size_t m=1; m<M; ++m, X+=K*Ly) { if (*X>mx) { mx = *X; } }
                        *Y = mx;
                    }
                }
            }
        }
    }

    return 0;
}


int maxout_d (double *Y, const double *X, const size_t R, const size_t C, const size_t S, const size_t H, const char iscolmajor, const size_t dim, const size_t M)
{
    if (dim>3) { fprintf(stderr,"error in maxout_d: dim must be in [0 3]\n"); return 1; }

    const size_t N = R*C*S*H;
    const size_t Lx = (dim==0u) ? R : (dim==1u) ? C : (dim==2u) ? S : H;
    if (Lx%M) { fprintf(stderr,"error in maxout_d: each vector in X must have length that is multiple of M\n"); return 1; }
    const size_t Ly = Lx / M;
    double mx;

    if (N==0u) {}
    else if (Lx==1)
    {
        for (size_t n=0u; n<N; ++n, ++X, ++Y) { *Y = *X; }
    }
    else if (Lx==N)
    {
        for (size_t l=0u; l<Ly; ++l, X-=Lx-1, ++Y)
        {
            mx = *X; X += Ly;
            for (size_t m=1; m<M; ++m, X+=Ly) { if (*X>mx) { mx = *X; } }
            *Y = mx;
        }
    }
    else
    {
        const size_t K = (iscolmajor) ? ((dim==0u) ? 1u : (dim==1u) ? R : (dim==2u) ? R*C : R*C*S) : ((dim==0u) ? C*S*H : (dim==1u) ? S*H : (dim==2u) ? H : 1u);
        const size_t B = (iscolmajor && dim==0u) ? C*S*H : K;
        const size_t V = N/Lx, G = V/B;

        if (K==1u && (G==1u || B==1u))
        {
            for (size_t v=0; v<V; ++v, X-=Ly-1)
            {
                for (size_t l=0u; l<Ly; ++l, ++Y)
                {
                    mx = *X; X += Ly;
                    for (size_t m=1; m<M; ++m, X+=Ly) { if (*X>mx) { mx = *X; } }
                    *Y = mx;
                    if (l<Ly-1) { X -= Lx - 1; }
                }
            }
        }
        else
        {
            for (size_t g=G; g>0u; --g, X+=B*(Lx-1), Y+=B*(Ly-1))
            {
                for (size_t b=B; b>0u; --b, X-=K*Ly-1, Y-=K*Ly-1)
                {
                    for (size_t l=0u; l<Ly; ++l, X-=K*(Lx-1), Y+=K)
                    {
                        mx = *X; X += K*Ly;
                        for (size_t m=1; m<M; ++m, X+=K*Ly) { if (*X>mx) { mx = *X; } }
                        *Y = mx;
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
