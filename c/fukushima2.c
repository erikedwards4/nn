//This does CELL (~soma) stage of Fukushima model.
//This requires each neuron to have 2 input time series, Xe and Xi,
//where Xe is the excitatory and Xi the inhibitory input.
//For each neuron and time-point: y = (1+xe)/(1+xi) - 1.
//Xe and Xi result from applying weights (linear0) We and Wi, respectively.

#include <stdio.h>
//#include <time.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int fukushima2_s (float *Y, const float *Xe, const float *Xi, const int N, const int T, const int dim, const char iscolmajor);
int fukushima2_d (double *Y, const double *Xe, const double *Xi, const int N, const int T, const int dim, const char iscolmajor);

int fukushima2_inplace_s (float *Xe, const float *Xi, const int N, const int T, const int dim, const char iscolmajor);
int fukushima2_inplace_d (double *Xe, const double *Xi, const int N, const int T, const int dim, const char iscolmajor);


int fukushima2_s (float *Y, const float *Xe, const float *Xi, const int N, const int T, const int dim, const char iscolmajor)
{
    int n, t, nT, tN;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in fukushima2_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in fukushima2_s: T (num time points) must be positive\n"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (dim==0)
    {
        if (iscolmajor)
        {
            for (t=0; t<T; t++)
            {
                tN = t*N;
                for (n=0; n<N; n++) { Y[tN+n] = (1.0f+Xe[tN+n])/(1.0f+Xi[tN+n]) - 1.0f; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                for (t=0; t<T; t++) { Y[nT+t] = (1.0f+Xe[nT+t])/(1.0f+Xi[nT+t]) - 1.0f; }
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
                for (t=0; t<T; t++) { Y[nT+t] = (1.0f+Xe[nT+t])/(1.0f+Xi[nT+t]) - 1.0f; }
            }
        }
        else
        {
            for (t=0; t<T; t++)
            {
                tN = t*N;
                for (n=0; n<N; n++) { Y[tN+n] = (1.0f+Xe[tN+n])/(1.0f+Xi[tN+n]) - 1.0f; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in fukushima2_s: dim must be 0 or 1.\n"); return 1;
    }

    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);
    return 0;
}


int fukushima2_d (double *Y, const double *Xe, const double *Xi, const int N, const int T, const int dim, const char iscolmajor)
{
    int n, t, nT, tN;

    //Checks
    if (N<1) { fprintf(stderr,"error in fukushima2_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in fukushima2_d: T (num time points) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (t=0; t<T; t++)
            {
                tN = t*N;
                for (n=0; n<N; n++) { Y[tN+n] = (1.0+Xe[tN+n])/(1.0+Xi[tN+n]) - 1.0; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                for (t=0; t<T; t++) { Y[nT+t] = (1.0+Xe[nT+t])/(1.0+Xi[nT+t]) - 1.0; }
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
                for (t=0; t<T; t++) { Y[nT+t] = (1.0+Xe[nT+t])/(1.0+Xi[nT+t]) - 1.0; }
            }
        }
        else
        {
            for (t=0; t<T; t++)
            {
                tN = t*N;
                for (n=0; n<N; n++) { Y[tN+n] = (1.0+Xe[tN+n])/(1.0+Xi[tN+n]) - 1.0; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in fukushima2_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int fukushima2_inplace_s (float *Xe, const float *Xi, const int N, const int T, const int dim, const char iscolmajor)
{
    int n, t, nT, tN;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in fukushima2_inplace_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in fukushima2_inplace_s: T (num time points) must be positive\n"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (dim==0)
    {
        if (iscolmajor)
        {
            for (t=0; t<T; t++)
            {
                tN = t*N;
                for (n=0; n<N; n++) { Xe[tN+n] = (1.0f+Xe[tN+n])/(1.0f+Xi[tN+n]) - 1.0f; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                for (t=0; t<T; t++) { Xe[nT+t] = (1.0f+Xe[nT+t])/(1.0f+Xi[nT+t]) - 1.0f; }
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
                for (t=0; t<T; t++) { Xe[nT+t] = (1.0f+Xe[nT+t])/(1.0f+Xi[nT+t]) - 1.0f; }
            }
        }
        else
        {
            for (t=0; t<T; t++)
            {
                tN = t*N;
                for (n=0; n<N; n++) { Xe[tN+n] = (1.0f+Xe[tN+n])/(1.0f+Xi[tN+n]) - 1.0f; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in fukushima2_inplace_s: dim must be 0 or 1.\n"); return 1;
    }

    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);
    return 0;
}


int fukushima2_inplace_d (double *Xe, const double *Xi, const int N, const int T, const int dim, const char iscolmajor)
{
    int n, t, nT, tN;

    //Checks
    if (N<1) { fprintf(stderr,"error in fukushima2_inplace_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in fukushima2_inplace_d: T (num time points) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (t=0; t<T; t++)
            {
                tN = t*N;
                for (n=0; n<N; n++) { Xe[tN+n] = (1.0+Xe[tN+n])/(1.0+Xi[tN+n]) - 1.0; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                for (t=0; t<T; t++) { Xe[nT+t] = (1.0+Xe[nT+t])/(1.0+Xi[nT+t]) - 1.0; }
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
                for (t=0; t<T; t++) { Xe[nT+t] = (1.0+Xe[nT+t])/(1.0+Xi[nT+t]) - 1.0; }
            }
        }
        else
        {
            for (t=0; t<T; t++)
            {
                tN = t*N;
                for (n=0; n<N; n++) { Xe[tN+n] = (1.0+Xe[tN+n])/(1.0+Xi[tN+n]) - 1.0; }
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
