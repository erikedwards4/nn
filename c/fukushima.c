//This does CELL (~soma) stage of Fukushima model.
//This requires each neuron to have 2 input time series, X1 and X2,
//where X1 is the excitatory and X2 the inhibitory input.
//For each neuron and time-point: y = (1+x1)/(1+x2) - 1.
//X1 and X2 result from applying weights (wx).
//The input X here is [X1; X2].

//For dim==0, T>N and not in_place, row-major is ~10% faster here and with C++.
//For dim==0, T>N and in_place, row-major makes much more sense and is ~10% faster here and ~1.5x faster with C++.
//For dim==1, T>N and not in_pace, col-major is ~10% faster here and with C++.
//For dim==1, T>N and in_place, col-major makes much more sense and is ~10% faster here and ~1.5x faster with C++.

//The in_place version is definitely faster for dim==0 and row-major or dim==1 and col-major,
//but actually a bit slower for dim==1 and row-major dim==0 and col-major.

#include <stdio.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int fukushima_s (float *Y, const float *X, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int fukushima_d (double *Y, const double *X, const size_t N, const size_t T, const char iscolmajor, const size_t dim);

int fukushima_inplace_s (float *X, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int fukushima_inplace_d (double *X, const size_t N, const size_t T, const char iscolmajor, const size_t dim);


int fukushima_s (float *Y, const float *X, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const size_t NT = N*T;
    size_t nT, tN;

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t t=0; t<T; ++t)
            {
                tN = t*N;
                for (size_t n=0; n<N; ++n) { Y[tN+n] = (1.0f+X[2*tN+n])/(1.0f+X[2*tN+N+n]) - 1.0f; }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                for (size_t t=0; t<T; ++t) { Y[nT+t] = (1.0f+X[nT+t])/(1.0f+X[nT+NT+t]) - 1.0f; }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                for (size_t t=0; t<T; ++t) { Y[nT+t] = (1.0f+X[nT+t])/(1.0f+X[nT+NT+t]) - 1.0f; }
            }
        }
        else
        {
            for (size_t t=0; t<T; ++t)
            {
                tN = t*N;
                for (size_t n=0; n<N; ++n) { Y[tN+n] = (1.0f+X[2*tN+n])/(1.0f+X[2*tN+N+n]) - 1.0f; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in fukushima_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int fukushima_d (double *Y, const double *X, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const size_t NT = N*T;
    size_t nT, tN;

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t t=0; t<T; ++t)
            {
                tN = t*N;
                for (size_t n=0; n<N; ++n) { Y[tN+n] = (1.0+X[2*tN+n])/(1.0+X[2*tN+N+n]) - 1.0; }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                for (size_t t=0; t<T; ++t) { Y[nT+t] = (1.0+X[nT+t])/(1.0+X[nT+NT+t]) - 1.0; }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                for (size_t t=0; t<T; ++t) { Y[nT+t] = (1.0+X[nT+t])/(1.0+X[nT+NT+t]) - 1.0; }
            }
        }
        else
        {
            for (size_t t=0; t<T; ++t)
            {
                tN = t*N;
                for (size_t n=0; n<N; ++n) { Y[tN+n] = (1.0+X[2*tN+n])/(1.0+X[2*tN+N+n]) - 1.0; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in fukushima_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int fukushima_inplace_s (float *X, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const size_t NT = N*T;
    size_t nT, tN2;

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t t=0; t<T; ++t)
            {
                tN2 = 2*t*N;
                for (size_t n=0; n<N; ++n) { X[tN2+n] = (1.0f+X[tN2+n])/(1.0f+X[tN2+N+n]) - 1.0f; }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                for (size_t t=0; t<T; ++t) { X[nT+t] = (1.0f+X[nT+t])/(1.0f+X[nT+NT+t]) - 1.0f; }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                for (size_t t=0; t<T; ++t) { X[nT+t] = (1.0f+X[nT+t])/(1.0f+X[nT+NT+t]) - 1.0f; }
            }
        }
        else
        {
            for (size_t t=0; t<T; ++t)
            {
                tN2 = 2*t*N;
                for (size_t n=0; n<N; ++n) { X[tN2+n] = (1.0f+X[tN2+n])/(1.0f+X[tN2+N+n]) - 1.0f; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in fukushima_inplace_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int fukushima_inplace_d (double *X, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    const size_t NT = N*T;
    size_t nT, tN2;

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t t=0; t<T; ++t)
            {
                tN2 = 2*t*N;
                for (size_t n=0; n<N; ++n) { X[tN2+n] = (1.0+X[tN2+n])/(1.0+X[tN2+N+n]) - 1.0; }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                for (size_t t=0; t<T; ++t) { X[nT+t] = (1.0+X[nT+t])/(1.0+X[nT+NT+t]) - 1.0; }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                for (size_t t=0; t<T; ++t) { X[nT+t] = (1.0+X[nT+t])/(1.0+X[nT+NT+t]) - 1.0; }
            }
        }
        else
        {
            for (size_t t=0; t<T; ++t)
            {
                tN2 = 2*t*N;
                for (size_t n=0; n<N; ++n) { X[tN2+n] = (1.0+X[tN2+n])/(1.0+X[tN2+N+n]) - 1.0; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in fukushima_inplace_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
