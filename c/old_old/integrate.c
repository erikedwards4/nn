//CELL (~soma) stage consisting of simple 1st-order integrator with time-constant tau (in seconds).
//The sample rate (fs) must be given in Hz.
//Note that one often specifies the cutoff frequency (fc) in Hz, instead of tau, where fc = 1/(2*pi*tau).
//Sometimes one specifies the cutoff frequency in radians/s, which is just 1/tau.

//For dim=0: Y[n,t] = a[n]*Y[n,t-1] + b[n]*X[n,t];
//For dim=1: Y[t,n] = a[n]*Y[t-1,n] + b[n]*X[t,n];
//where a[n] = exp(-1/(fs*tau[n])) and b[n] = 1 - a[n].
//Note that tau is a vector of length N, so that each neuron has its own tau.

//For complex inputs, real and imag parts are filtered separately, so tau is still real.


#include <stdio.h>
#include <math.h>
//#include <time.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int integrate_s (float *Y, const float *X, const float *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int integrate_d (double *Y, const double *X, const double *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);
int integrate_c (float *Y, const float *X, const float *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int integrate_z (double *Y, const double *X, const double *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);

int integrate_inplace_s (float *X, const float *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int integrate_inplace_d (double *X, const double *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);
int integrate_inplace_c (float *X, const float *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int integrate_inplace_z (double *X, const double *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);

int integrate_s (float *Y, const float *X, const float *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs)
{
    int nT;
    float a, b;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in integrate_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in integrate_s: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0f) { fprintf(stderr,"error in integrate_s: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0f) { fprintf(stderr,"error in integrate_s: taus must be positive\n"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (N==1)
    {
        a = expf(-1.0f/(fs*tau[0])); b = 1.0f - a;
        Y[0] = b*X[0];
        for (size_t t=1; t<T; ++t) { Y[t] = a*Y[t-1] + b*X[t]; }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a;
                Y[n] = b*X[n];
                for (size_t t=1; t<T; ++t) { Y[n+t*N] = a*Y[n+(t-1)*N] + b*X[n+t*N]; }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a;
                Y[nT] = b*X[nT];
                for (size_t t=1; t<T; ++t) { Y[nT+t] = a*Y[nT+t-1] + b*X[nT+t]; }
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
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a;
                Y[nT] = b*X[nT];
                for (size_t t=1; t<T; ++t) { Y[nT+t] = a*Y[nT+t-1] + b*X[nT+t]; }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a;
                Y[n] = b*X[n];
                for (size_t t=1; t<T; ++t) { Y[n+t*N] = a*Y[n+(t-1)*N] + b*X[n+t*N]; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in integrate_s: dim must be 0 or 1.\n"); return 1;
    }
    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);

    return 0;
}


int integrate_d (double *Y, const double *X, const double *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs)
{
    int nT;
    double a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in integrate_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in integrate_d: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0) { fprintf(stderr,"error in integrate_d: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0) { fprintf(stderr,"error in integrate_d: taus must be positive\n"); return 1; }

    if (N==1)
    {
        a = exp(-1.0/(fs*tau[0])); b = 1.0 - a;
        Y[0] = b*X[0];
        for (size_t t=1; t<T; ++t) { Y[t] = a*Y[t-1] + b*X[t]; }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a;
                Y[n] = b*X[n];
                for (size_t t=1; t<T; ++t) { Y[n+t*N] = a*Y[n+(t-1)*N] + b*X[n+t*N]; }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a;
                Y[nT] = b*X[nT];
                for (size_t t=1; t<T; ++t) { Y[nT+t] = a*Y[nT+t-1] + b*X[nT+t]; }
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
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a;
                Y[nT] = b*X[nT];
                for (size_t t=1; t<T; ++t) { Y[nT+t] = a*Y[nT+t-1] + b*X[nT+t]; }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a;
                Y[n] = b*X[n];
                for (size_t t=1; t<T; ++t) { Y[n+t*N] = a*Y[n+(t-1)*N] + b*X[n+t*N]; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in integrate_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int integrate_c (float *Y, const float *X, const float *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs)
{
    int nT;
    float a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in integrate_c: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in integrate_c: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0f) { fprintf(stderr,"error in integrate_c: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0f) { fprintf(stderr,"error in integrate_c: taus must be positive\n"); return 1; }

    if (N==1)
    {
        a = expf(-1.0f/(fs*tau[0])); b = 1.0f - a;
        Y[0] = b*X[0]; Y[1] = b*X[1];
        for (size_t t=1; t<T; ++t)
        {
            Y[2*t] = a*Y[2*t-2] + b*X[2*t];
            Y[2*t+1] = a*Y[2*t-1] + b*X[2*t+1];
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a;
                Y[2*n] = b*X[2*n]; Y[2*n+1] = b*X[2*n+1];
                for (size_t t=1; t<T; ++t)
                {
                    Y[2*(n+t*N)] = a*Y[2*(n+(t-1)*N)] + b*X[2*(n+t*N)];
                    Y[2*(n+t*N)+1] = a*Y[2*(n+(t-1)*N)+1] + b*X[2*(n+t*N)+1];
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a;
                Y[2*nT] = b*X[2*nT]; Y[2*nT+1] = b*X[2*nT+1];
                for (size_t t=1; t<T; ++t)
                {
                    Y[2*(nT+t)] = a*Y[2*(nT+t-1)] + b*X[2*(nT+t)];
                    Y[2*(nT+t)+1] = a*Y[2*(nT+t-1)+1] + b*X[2*(nT+t)+1];
                }
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
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a;
                Y[2*nT] = b*X[2*nT]; Y[2*nT+1] = b*X[2*nT+1];
                for (size_t t=1; t<T; ++t)
                {
                    Y[2*(nT+t)] = a*Y[2*(nT+t-1)] + b*X[2*(nT+t)];
                    Y[2*(nT+t)+1] = a*Y[2*(nT+t-1)+1] + b*X[2*(nT+t)+1];
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a;
                Y[2*n] = b*X[2*n]; Y[2*n+1] = b*X[2*n+1];
                for (size_t t=1; t<T; ++t)
                {
                    Y[2*(n+t*N)] = a*Y[2*(n+(t-1)*N)] + b*X[2*(n+t*N)];
                    Y[2*(n+t*N)+1] = a*Y[2*(n+(t-1)*N)+1] + b*X[2*(n+t*N)+1];
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in integrate_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int integrate_z (double *Y, const double *X, const double *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs)
{
    int nT;
    double a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in integrate_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in integrate_d: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0) { fprintf(stderr,"error in integrate_d: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0) { fprintf(stderr,"error in integrate_d: taus must be positive\n"); return 1; }

    if (N==1)
    {
        a = exp(-1.0/(fs*tau[0])); b = 1.0 - a;
        Y[0] = b*X[0]; Y[1] = b*X[1];
        for (size_t t=1; t<T; ++t)
        {
            Y[2*t] = a*Y[2*t-2] + b*X[2*t];
            Y[2*t+1] = a*Y[2*t-1] + b*X[2*t+1];
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a;
                Y[2*n] = b*X[2*n]; Y[2*n+1] = b*X[2*n+1];
                for (size_t t=1; t<T; ++t)
                {
                    Y[2*(n+t*N)] = a*Y[2*(n+(t-1)*N)] + b*X[2*(n+t*N)];
                    Y[2*(n+t*N)+1] = a*Y[2*(n+(t-1)*N)+1] + b*X[2*(n+t*N)+1];
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a;
                Y[2*nT] = b*X[2*nT]; Y[2*nT+1] = b*X[2*nT+1];
                for (size_t t=1; t<T; ++t)
                {
                    Y[2*(nT+t)] = a*Y[2*(nT+t-1)] + b*X[2*(nT+t)];
                    Y[2*(nT+t)+1] = a*Y[2*(nT+t-1)+1] + b*X[2*(nT+t)+1];
                }
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
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a;
                Y[2*nT] = b*X[2*nT]; Y[2*nT+1] = b*X[2*nT+1];
                for (size_t t=1; t<T; ++t)
                {
                    Y[2*(nT+t)] = a*Y[2*(nT+t-1)] + b*X[2*(nT+t)];
                    Y[2*(nT+t)+1] = a*Y[2*(nT+t-1)+1] + b*X[2*(nT+t)+1];
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a;
                Y[2*n] = b*X[2*n]; Y[2*n+1] = b*X[2*n+1];
                for (size_t t=1; t<T; ++t)
                {
                    Y[2*(n+t*N)] = a*Y[2*(n+(t-1)*N)] + b*X[2*(n+t*N)];
                    Y[2*(n+t*N)+1] = a*Y[2*(n+(t-1)*N)+1] + b*X[2*(n+t*N)+1];
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in integrate_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int integrate_inplace_s (float *X, const float *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs)
{
    int nT;
    float a, b;
    //struct timespec tic, toc;

    //Checks
    if (N<1) { fprintf(stderr,"error in integrate_inplace_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in integrate_inplace_s: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0f) { fprintf(stderr,"error in integrate_inplace_s: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0f) { fprintf(stderr,"error in integrate_inplace_s: taus must be positive\n"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    if (N==1)
    {
        a = expf(-1.0f/(fs*tau[0])); b = 1.0f - a;
        X[0] *= b;
        for (size_t t=1; t<T; ++t) { X[t] = a*X[t-1] + b*X[t]; }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a;
                X[n] *= b;
                for (size_t t=1; t<T; ++t) { X[n+t*N] = a*X[n+(t-1)*N] + b*X[n+t*N]; }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a;
                X[nT] *= b;
                for (size_t t=1; t<T; ++t) { X[nT+t] = a*X[nT+t-1] + b*X[nT+t]; }
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
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a;
                X[nT] *= b;
                for (size_t t=1; t<T; ++t) { X[nT+t] = a*X[nT+t-1] + b*X[nT+t]; }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a;
                X[n] *= b;
                for (size_t t=1; t<T; ++t) { X[n+t*N] = a*X[n+(t-1)*N] + b*X[n+t*N]; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in integrate_inplace_s: dim must be 0 or 1.\n"); return 1;
    }
    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);

    return 0;
}


int integrate_inplace_d (double *X, const double *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs)
{
    int nT;
    double a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in integrate_inplace_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in integrate_inplace_d: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0) { fprintf(stderr,"error in integrate_inplace_d: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0) { fprintf(stderr,"error in integrate_inplace_d: taus must be positive\n"); return 1; }

    if (N==1)
    {
        a = exp(-1.0/(fs*tau[0])); b = 1.0 - a;
        X[0] *= b;
        for (size_t t=1; t<T; ++t) { X[t] = a*X[t-1] + b*X[t]; }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a;
                X[n] *= b;
                for (size_t t=1; t<T; ++t) { X[n+t*N] = a*X[n+(t-1)*N] + b*X[n+t*N]; }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a;
                X[nT] *= b;
                for (size_t t=1; t<T; ++t) { X[nT+t] = a*X[nT+t-1] + b*X[nT+t]; }
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
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a;
                X[nT] *= b;
                for (size_t t=1; t<T; ++t) { X[nT+t] = a*X[nT+t-1] + b*X[nT+t]; }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a;
                X[n] *= b;
                for (size_t t=1; t<T; ++t) { X[n+t*N] = a*X[n+(t-1)*N] + b*X[n+t*N]; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in integrate_inplace_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int integrate_inplace_c (float *X, const float *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs)
{
    int nT;
    float a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in integrate_inplace_c: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in integrate_inplace_c: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0f) { fprintf(stderr,"error in integrate_inplace_c: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0f) { fprintf(stderr,"error in integrate_inplace_c: taus must be positive\n"); return 1; }

    if (N==1)
    {
        a = expf(-1.0f/(fs*tau[0])); b = 1.0f - a;
        X[0] *= b; X[1] *= b;
        for (size_t t=1; t<T; ++t)
        {
            X[2*t] = a*X[2*t-2] + b*X[2*t];
            X[2*t+1] = a*X[2*t-1] + b*X[2*t+1];
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a;
                X[2*n] *= b; X[2*n+1] *= b;
                for (size_t t=1; t<T; ++t)
                {
                    X[2*(n+t*N)] = a*X[2*(n+(t-1)*N)] + b*X[2*(n+t*N)];
                    X[2*(n+t*N)+1] = a*X[2*(n+(t-1)*N)+1] + b*X[2*(n+t*N)+1];
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a;
                X[2*nT] *= b; X[2*nT+1] *= b;
                for (size_t t=1; t<T; ++t)
                {
                    X[2*(nT+t)] = a*X[2*(nT+t-1)] + b*X[2*(nT+t)];
                    X[2*(nT+t)+1] = a*X[2*(nT+t-1)+1] + b*X[2*(nT+t)+1];
                }
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
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a;
                X[2*nT] *= b; X[2*nT+1] *= b;
                for (size_t t=1; t<T; ++t)
                {
                    X[2*(nT+t)] = a*X[2*(nT+t-1)] + b*X[2*(nT+t)];
                    X[2*(nT+t)+1] = a*X[2*(nT+t-1)+1] + b*X[2*(nT+t)+1];
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a;
                X[2*n] *= b; X[2*n+1] *= b;
                for (size_t t=1; t<T; ++t)
                {
                    X[2*(n+t*N)] = a*X[2*(n+(t-1)*N)] + b*X[2*(n+t*N)];
                    X[2*(n+t*N)+1] = a*X[2*(n+(t-1)*N)+1] + b*X[2*(n+t*N)+1];
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in integrate_inplace_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int integrate_inplace_z (double *X, const double *tau, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs)
{
    int nT;
    double a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in integrate_inplace_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in integrate_inplace_d: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0) { fprintf(stderr,"error in integrate_inplace_d: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0) { fprintf(stderr,"error in integrate_inplace_d: taus must be positive\n"); return 1; }

    if (N==1)
    {
        a = exp(-1.0/(fs*tau[0])); b = 1.0 - a;
        X[0] *= b; X[1] *= b;
        for (size_t t=1; t<T; ++t)
        {
            X[2*t] = a*X[2*t-2] + b*X[2*t];
            X[2*t+1] = a*X[2*t-1] + b*X[2*t+1];
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t n=0; n<N; ++n)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a;
                X[2*n] *= b; X[2*n+1] *= b;
                for (size_t t=1; t<T; ++t)
                {
                    X[2*(n+t*N)] = a*X[2*(n+(t-1)*N)] + b*X[2*(n+t*N)];
                    X[2*(n+t*N)+1] = a*X[2*(n+(t-1)*N)+1] + b*X[2*(n+t*N)+1];
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a;
                X[2*nT] *= b; X[2*nT+1] *= b;
                for (size_t t=1; t<T; ++t)
                {
                    X[2*(nT+t)] = a*X[2*(nT+t-1)] + b*X[2*(nT+t)];
                    X[2*(nT+t)+1] = a*X[2*(nT+t-1)+1] + b*X[2*(nT+t)+1];
                }
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
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a;
                X[2*nT] *= b; X[2*nT+1] *= b;
                for (size_t t=1; t<T; ++t)
                {
                    X[2*(nT+t)] = a*X[2*(nT+t-1)] + b*X[2*(nT+t)];
                    X[2*(nT+t)+1] = a*X[2*(nT+t-1)+1] + b*X[2*(nT+t)+1];
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a;
                X[2*n] *= b; X[2*n+1] *= b;
                for (size_t t=1; t<T; ++t)
                {
                    X[2*(n+t*N)] = a*X[2*(n+(t-1)*N)] + b*X[2*(n+t*N)];
                    X[2*(n+t*N)+1] = a*X[2*(n+(t-1)*N)+1] + b*X[2*(n+t*N)+1];
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in integrate_inplace_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
