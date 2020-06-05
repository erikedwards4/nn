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
//#include <time.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int fukushima_s (float *Y, const float *X, const int N, const int T, const int dim, const char iscolmajor);
int fukushima_d (double *Y, const double *X, const int N, const int T, const int dim, const char iscolmajor);

int fukushima_inplace_s (float *X, const int N, const int T, const int dim, const char iscolmajor);
int fukushima_inplace_d (double *X, const int N, const int T, const int dim, const char iscolmajor);


int fukushima_s (float *Y, const float *X, const int N, const int T, const int dim, const char iscolmajor)
{
    const int NT = N*T;
    int n, t, nT, tN;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in fukushima_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in fukushima_s: T (num time points) must be positive\n"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (dim==0)
    {
        if (iscolmajor)
        {
            for (t=0; t<T; t++)
            {
                tN = t*N;
                for (n=0; n<N; n++) { Y[tN+n] = (1.0f+X[2*tN+n])/(1.0f+X[2*tN+N+n]) - 1.0f; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                for (t=0; t<T; t++) { Y[nT+t] = (1.0f+X[nT+t])/(1.0f+X[nT+NT+t]) - 1.0f; }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                for (t=0; t<T; t++) { Y[nT+t] = (1.0f+X[nT+t])/(1.0f+X[nT+NT+t]) - 1.0f; }
            }
        }
        else
        {
            for (t=0; t<T; t++)
            {
                tN = t*N;
                for (n=0; n<N; n++) { Y[tN+n] = (1.0f+X[2*tN+n])/(1.0f+X[2*tN+N+n]) - 1.0f; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in fukushima_s: dim must be 0 or 1.\n"); return 1;
    }

    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);
    return 0;
}


int fukushima_d (double *Y, const double *X, const int N, const int T, const int dim, const char iscolmajor)
{
    const int NT = N*T;
    int n, t, nT, tN;

    //Checks
    if (N<1) { fprintf(stderr,"error in fukushima_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in fukushima_d: T (num time points) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (t=0; t<T; t++)
            {
                tN = t*N;
                for (n=0; n<N; n++) { Y[tN+n] = (1.0+X[2*tN+n])/(1.0+X[2*tN+N+n]) - 1.0; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                for (t=0; t<T; t++) { Y[nT+t] = (1.0+X[nT+t])/(1.0+X[nT+NT+t]) - 1.0; }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                for (t=0; t<T; t++) { Y[nT+t] = (1.0+X[nT+t])/(1.0+X[nT+NT+t]) - 1.0; }
            }
        }
        else
        {
            for (t=0; t<T; t++)
            {
                tN = t*N;
                for (n=0; n<N; n++) { Y[tN+n] = (1.0+X[2*tN+n])/(1.0+X[2*tN+N+n]) - 1.0; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in fukushima_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int fukushima_inplace_s (float *X, const int N, const int T, const int dim, const char iscolmajor)
{
    const int NT = N*T;
    int n, t, nT, tN2;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in fukushima_inplace_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in fukushima_inplace_s: T (num time points) must be positive\n"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (dim==0)
    {
        if (iscolmajor)
        {
            for (t=0; t<T; t++)
            {
                tN2 = 2*t*N;
                for (n=0; n<N; n++) { X[tN2+n] = (1.0f+X[tN2+n])/(1.0f+X[tN2+N+n]) - 1.0f; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                for (t=0; t<T; t++) { X[nT+t] = (1.0f+X[nT+t])/(1.0f+X[nT+NT+t]) - 1.0f; }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                for (t=0; t<T; t++) { X[nT+t] = (1.0f+X[nT+t])/(1.0f+X[nT+NT+t]) - 1.0f; }
            }
        }
        else
        {
            for (t=0; t<T; t++)
            {
                tN2 = 2*t*N;
                for (n=0; n<N; n++) { X[tN2+n] = (1.0f+X[tN2+n])/(1.0f+X[tN2+N+n]) - 1.0f; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in fukushima_inplace_s: dim must be 0 or 1.\n"); return 1;
    }

    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);
    return 0;
}


int fukushima_inplace_d (double *X, const int N, const int T, const int dim, const char iscolmajor)
{
    const int NT = N*T;
    int n, t, nT, tN2;

    //Checks
    if (N<1) { fprintf(stderr,"error in fukushima_inplace_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in fukushima_inplace_d: T (num time points) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (t=0; t<T; t++)
            {
                tN2 = 2*t*N;
                for (n=0; n<N; n++) { X[tN2+n] = (1.0+X[tN2+n])/(1.0+X[tN2+N+n]) - 1.0; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                for (t=0; t<T; t++) { X[nT+t] = (1.0+X[nT+t])/(1.0+X[nT+NT+t]) - 1.0; }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                for (t=0; t<T; t++) { X[nT+t] = (1.0+X[nT+t])/(1.0+X[nT+NT+t]) - 1.0; }
            }
        }
        else
        {
            for (t=0; t<T; t++)
            {
                tN2 = 2*t*N;
                for (n=0; n<N; n++) { X[tN2+n] = (1.0+X[tN2+n])/(1.0+X[tN2+N+n]) - 1.0; }
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
