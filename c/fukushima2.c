//This does CELL (~soma) stage of Fukushima model.
//This requires each neuron to have 2 input time series, Xe and Xi,
//where Xe is the excitatory and Xi the inhibitory input.
//For each neuron and time-point: y = (1+xe)/(1+xi) - 1.
//Xe and Xi result from applying weights (linear0) We and Wi, respectively.

#include <stdio.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int fukushima2_s (float *Y, const float *Xe, const float *Xi, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int fukushima2_d (double *Y, const double *Xe, const double *Xi, const size_t N, const size_t T, const char iscolmajor, const size_t dim);

int fukushima2_inplace_s (float *Xe, const float *Xi, const size_t N, const size_t T, const char iscolmajor, const size_t dim);
int fukushima2_inplace_d (double *Xe, const double *Xi, const size_t N, const size_t T, const char iscolmajor, const size_t dim);


int fukushima2_s (float *Y, const float *Xe, const float *Xi, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    size_t nT, tN;

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t t=0; t<T; ++t)
            {
                tN = t*N;
                for (size_t n=0; n<N; ++n) { Y[tN+n] = (1.0f+Xe[tN+n])/(1.0f+Xi[tN+n]) - 1.0f; }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                for (size_t t=0; t<T; ++t) { Y[nT+t] = (1.0f+Xe[nT+t])/(1.0f+Xi[nT+t]) - 1.0f; }
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
                for (size_t t=0; t<T; ++t) { Y[nT+t] = (1.0f+Xe[nT+t])/(1.0f+Xi[nT+t]) - 1.0f; }
            }
        }
        else
        {
            for (size_t t=0; t<T; ++t)
            {
                tN = t*N;
                for (size_t n=0; n<N; ++n) { Y[tN+n] = (1.0f+Xe[tN+n])/(1.0f+Xi[tN+n]) - 1.0f; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in fukushima2_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int fukushima2_d (double *Y, const double *Xe, const double *Xi, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    size_t nT, tN;

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t t=0; t<T; ++t)
            {
                tN = t*N;
                for (size_t n=0; n<N; ++n) { Y[tN+n] = (1.0+Xe[tN+n])/(1.0+Xi[tN+n]) - 1.0; }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                for (size_t t=0; t<T; ++t) { Y[nT+t] = (1.0+Xe[nT+t])/(1.0+Xi[nT+t]) - 1.0; }
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
                for (size_t t=0; t<T; ++t) { Y[nT+t] = (1.0+Xe[nT+t])/(1.0+Xi[nT+t]) - 1.0; }
            }
        }
        else
        {
            for (size_t t=0; t<T; ++t)
            {
                tN = t*N;
                for (size_t n=0; n<N; ++n) { Y[tN+n] = (1.0+Xe[tN+n])/(1.0+Xi[tN+n]) - 1.0; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in fukushima2_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int fukushima2_inplace_s (float *Xe, const float *Xi, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    size_t nT, tN;

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t t=0; t<T; ++t)
            {
                tN = t*N;
                for (size_t n=0; n<N; ++n) { Xe[tN+n] = (1.0f+Xe[tN+n])/(1.0f+Xi[tN+n]) - 1.0f; }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                for (size_t t=0; t<T; ++t) { Xe[nT+t] = (1.0f+Xe[nT+t])/(1.0f+Xi[nT+t]) - 1.0f; }
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
                for (size_t t=0; t<T; ++t) { Xe[nT+t] = (1.0f+Xe[nT+t])/(1.0f+Xi[nT+t]) - 1.0f; }
            }
        }
        else
        {
            for (size_t t=0; t<T; ++t)
            {
                tN = t*N;
                for (size_t n=0; n<N; ++n) { Xe[tN+n] = (1.0f+Xe[tN+n])/(1.0f+Xi[tN+n]) - 1.0f; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in fukushima2_inplace_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int fukushima2_inplace_d (double *Xe, const double *Xi, const size_t N, const size_t T, const char iscolmajor, const size_t dim)
{
    size_t nT, tN;

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t t=0; t<T; ++t)
            {
                tN = t*N;
                for (size_t n=0; n<N; ++n) { Xe[tN+n] = (1.0+Xe[tN+n])/(1.0+Xi[tN+n]) - 1.0; }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                for (size_t t=0; t<T; ++t) { Xe[nT+t] = (1.0+Xe[nT+t])/(1.0+Xi[nT+t]) - 1.0; }
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
                for (size_t t=0; t<T; ++t) { Xe[nT+t] = (1.0+Xe[nT+t])/(1.0+Xi[nT+t]) - 1.0; }
            }
        }
        else
        {
            for (size_t t=0; t<T; ++t)
            {
                tN = t*N;
                for (size_t n=0; n<N; ++n) { Xe[tN+n] = (1.0+Xe[tN+n])/(1.0+Xi[tN+n]) - 1.0; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in fukushima2_inplace_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
