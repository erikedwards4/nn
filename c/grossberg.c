//CELL (~soma) stage consisting of a 1st-order integrator with time-constant tau (in seconds),
//along with a negative feedback from just after to just before the integrator,
//and an automatic gain control loop of gamma-beta*y(t-1) that multiplies the driving input.
//This is the discrete-time implementation of the Grossberg model [Cichocki & Unbehauen 1993, p.58].

//For dim=0: Y[n,t] = a[n]*Y[n,t-1] + b[n]*(g[n,t]*X[n,t]-alpha[n]*Y[n,t-1]);
//For dim=1: Y[t,n] = a[n]*Y[t-1,n] + b[n]*(g[t,n]*X[t,n]-alpha[n]*Y[t-1,n]);
//where a[n] = exp(-1/(fs*tau[n])) and b[n] = 1 - a[n].
//and g[n,t] = gamma[n] - beta[n]*Y[n,t-1].
//and g[t,n] = gamma[n] - beta[n]*Y[t-1,n].

//tau, alpha, beta and gamma are vectors of length N, so that each neuron has its own parameters.

//For complex inputs, real and imag parts are filtered separately, so params are real.

//There is a grossberg2 model still to be made, that uses Fukushima split of E and I inputs [C & U 1993, p.59].
//For the I inputs, the feedback beta term is added rather than subtracted.
//For now, I drop the constraint that the betas are nonnegative, so that this model could still be accomplished here.


#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int grossberg_s (float *Y, const float *X, const float *tau, const float *alpha, const float *beta, const float *gamma, const int N, const int T, const int dim, const char iscolmajor, const float fs);
int grossberg_d (double *Y, const double *X, const double *tau, const double *alpha, const double *beta, const double *gamma, const int N, const int T, const int dim, const char iscolmajor, const double fs);
int grossberg_c (float *Y, const float *X, const float *tau, const float *alpha, const float *beta, const float *gamma, const int N, const int T, const int dim, const char iscolmajor, const float fs);
int grossberg_z (double *Y, const double *X, const double *tau, const double *alpha, const double *beta, const double *gamma, const int N, const int T, const int dim, const char iscolmajor, const double fs);

int grossberg_inplace_s (float *X, const float *tau, const float *alpha, const float *beta, const float *gamma, const int N, const int T, const int dim, const char iscolmajor, const float fs);
int grossberg_inplace_d (double *X, const double *tau, const double *alpha, const double *beta, const double *gamma, const int N, const int T, const int dim, const char iscolmajor, const double fs);
int grossberg_inplace_c (float *X, const float *tau, const float *alpha, const float *beta, const float *gamma, const int N, const int T, const int dim, const char iscolmajor, const float fs);
int grossberg_inplace_z (double *X, const double *tau, const double *alpha, const double *beta, const double *gamma, const int N, const int T, const int dim, const char iscolmajor, const double fs);


int grossberg_s (float *Y, const float *X, const float *tau, const float *alpha, const float *beta, const float *gamma, const int N, const int T, const int dim, const char iscolmajor, const float fs)
{
    int n, t, nT;
    float a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in grossberg_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in grossberg_s: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0f) { fprintf(stderr,"error in grossberg_s: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0f) { fprintf(stderr,"error in grossberg_s: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0f) { fprintf(stderr,"error in grossberg_s: alphas must be nonnegative\n"); return 1; }
    //if (beta[0]<0.0f) { fprintf(stderr,"error in grossberg_s: betas must be nonnegative\n"); return 1; }

    if (N==1)
    {
        a = expf(-1.0f/(fs*tau[0])); b = 1.0f - a; a -= b*alpha[0];
        Y[0] = b*gamma[0]*X[0];
        for (t=1; t<T; t++) { Y[t] = a*Y[t-1] + b*(gamma[0]-beta[0]*Y[t-1])*X[t]; }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                Y[n] = b*gamma[n]*X[n];
                for (t=1; t<T; t++) { Y[n+t*N] = a*Y[n+(t-1)*N] + b*(gamma[n]-beta[n]*Y[n+(t-1)*N])*X[n+t*N]; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                nT = n*T; Y[nT] = b*gamma[n]*X[nT];
                for (t=1; t<T; t++) { Y[nT+t] = a*Y[nT+t-1] + b*(gamma[n]-beta[n]*Y[nT+t-1])*X[nT+t]; }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                nT = n*T; Y[nT] = b*gamma[n]*X[nT];
                for (t=1; t<T; t++) { Y[nT+t] = a*Y[nT+t-1] + b*(gamma[n]-beta[n]*Y[nT+t-1])*X[nT+t]; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                Y[n] = b*gamma[n]*X[n];
                for (t=1; t<T; t++) { Y[n+t*N] = a*Y[n+(t-1)*N] + b*(gamma[n]-beta[n]*Y[n+(t-1)*N])*X[n+t*N]; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in grossberg_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int grossberg_d (double *Y, const double *X, const double *tau, const double *alpha, const double *beta, const double *gamma, const int N, const int T, const int dim, const char iscolmajor, const double fs)
{
    int n, t, nT;
    double a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in grossberg_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in grossberg_d: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0) { fprintf(stderr,"error in grossberg_d: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0) { fprintf(stderr,"error in grossberg_d: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0) { fprintf(stderr,"error in grossberg_d: alphas must be nonnegative\n"); return 1; }
    //if (beta[0]<0.0) { fprintf(stderr,"error in grossberg_d: betas must be nonnegative\n"); return 1; }

    if (N==1)
    {
        a = exp(-1.0/(fs*tau[0])); b = 1.0 - a; a -= b*alpha[0];
        Y[0] = b*gamma[0]*X[0];
        for (t=1; t<T; t++) { Y[t] = a*Y[t-1] + b*(gamma[0]-beta[0]*Y[t-1])*X[t]; }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                Y[n] = b*gamma[n]*X[n];
                for (t=1; t<T; t++) { Y[n+t*N] = a*Y[n+(t-1)*N] + b*(gamma[n]-beta[n]*Y[n+(t-1)*N])*X[n+t*N]; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                nT = n*T; Y[nT] = b*gamma[n]*X[nT];
                for (t=1; t<T; t++) { Y[nT+t] = a*Y[nT+t-1] + b*(gamma[n]-beta[n]*Y[nT+t-1])*X[nT+t]; }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                nT = n*T; Y[nT] = b*gamma[n]*X[nT];
                for (t=1; t<T; t++) { Y[nT+t] = a*Y[nT+t-1] + b*(gamma[n]-beta[n]*Y[nT+t-1])*X[nT+t]; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                Y[n] = b*gamma[n]*X[n];
                for (t=1; t<T; t++) { Y[n+t*N] = a*Y[n+(t-1)*N] + b*(gamma[n]-beta[n]*Y[n+(t-1)*N])*X[n+t*N]; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in grossberg_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int grossberg_c (float *Y, const float *X, const float *tau, const float *alpha, const float *beta, const float *gamma, const int N, const int T, const int dim, const char iscolmajor, const float fs)
{
    int n, t, nT;
    float a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in grossberg_c: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in grossberg_c: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0f) { fprintf(stderr,"error in grossberg_c: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0f) { fprintf(stderr,"error in grossberg_c: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0f) { fprintf(stderr,"error in grossberg_c: alphas must be nonnegative\n"); return 1; }
    //if (beta[0]<0.0f) { fprintf(stderr,"error in grossberg_c: betas must be nonnegative\n"); return 1; }

    if (N==1)
    {
        a = expf(-1.0f/(fs*tau[0])); b = 1.0f - a; a -= b*alpha[0];
        Y[0] = b*gamma[0]*X[0]; Y[1] = b*gamma[0]*X[1];
        for (t=1; t<T; t++)
        {
            Y[2*t] = a*Y[2*(t-1)] + b*(gamma[0]-beta[0]*Y[2*(t-1)])*X[2*t];
            Y[2*t+1] = a*Y[2*(t-1)+1] + b*(gamma[0]-beta[0]*Y[2*(t-1)+1])*X[2*t+1];
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                Y[2*n] = b*gamma[n]*X[2*n]; Y[2*n+1] = b*gamma[n]*X[2*n+1];
                for (t=1; t<T; t++)
                {
                    Y[2*(n+t*N)] = a*Y[2*(n+(t-1)*N)] + b*(gamma[n]-beta[n]*Y[2*(n+(t-1)*N)])*X[2*(n+t*N)];
                    Y[2*(n+t*N)+1] = a*Y[2*(n+(t-1)*N)+1] + b*(gamma[n]-beta[n]*Y[2*(n+(t-1)*N)+1])*X[2*(n+t*N)+1];
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                nT = n*T; Y[2*nT] = b*gamma[n]*X[2*nT]; Y[2*nT+1] = b*gamma[n]*X[2*nT+1];
                for (t=1; t<T; t++)
                {
                    Y[2*(nT+t)] = a*Y[2*(nT+t-1)] + b*(gamma[n]-beta[n]*Y[2*(nT+t-1)])*X[2*(nT+t)];
                    Y[2*(nT+t)+1] = a*Y[2*(nT+t-1)+1] + b*(gamma[n]-beta[n]*Y[2*(nT+t-1)+1])*X[2*(nT+t)+1];
                }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                nT = n*T; Y[2*nT] = b*gamma[n]*X[2*nT]; Y[2*nT+1] = b*gamma[n]*X[2*nT+1];
                for (t=1; t<T; t++)
                {
                    Y[2*(nT+t)] = a*Y[2*(nT+t-1)] + b*(gamma[n]-beta[n]*Y[2*(nT+t-1)])*X[2*(nT+t)];
                    Y[2*(nT+t)+1] = a*Y[2*(nT+t-1)+1] + b*(gamma[n]-beta[n]*Y[2*(nT+t-1)+1])*X[2*(nT+t)+1];
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                Y[2*n] = b*gamma[n]*X[2*n]; Y[2*n+1] = b*gamma[n]*X[2*n+1];
                for (t=1; t<T; t++)
                {
                    Y[2*(n+t*N)] = a*Y[2*(n+(t-1)*N)] + b*(gamma[n]-beta[n]*Y[2*(n+(t-1)*N)])*X[2*(n+t*N)];
                    Y[2*(n+t*N)+1] = a*Y[2*(n+(t-1)*N)+1] + b*(gamma[n]-beta[n]*Y[2*(n+(t-1)*N)+1])*X[2*(n+t*N)+1];
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in grossberg_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int grossberg_z (double *Y, const double *X, const double *tau, const double *alpha, const double *beta, const double *gamma, const int N, const int T, const int dim, const char iscolmajor, const double fs)
{
    int n, t, nT;
    double a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in grossberg_z: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in grossberg_z: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0) { fprintf(stderr,"error in grossberg_z: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0) { fprintf(stderr,"error in grossberg_z: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0) { fprintf(stderr,"error in grossberg_z: alphas must be nonnegative\n"); return 1; }
    //if (beta[0]<0.0) { fprintf(stderr,"error in grossberg_z: betas must be nonnegative\n"); return 1; }

    if (N==1)
    {
        a = exp(-1.0/(fs*tau[0])); b = 1.0 - a; a -= b*alpha[0];
        Y[0] = b*gamma[0]*X[0]; Y[1] = b*gamma[0]*X[1];
        for (t=1; t<T; t++)
        {
            Y[2*t] = a*Y[2*(t-1)] + b*(gamma[0]-beta[0]*Y[2*(t-1)])*X[2*t];
            Y[2*t+1] = a*Y[2*(t-1)+1] + b*(gamma[0]-beta[0]*Y[2*(t-1)+1])*X[2*t+1];
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                Y[2*n] = b*gamma[n]*X[2*n]; Y[2*n+1] = b*gamma[n]*X[2*n+1];
                for (t=1; t<T; t++)
                {
                    Y[2*(n+t*N)] = a*Y[2*(n+(t-1)*N)] + b*(gamma[n]-beta[n]*Y[2*(n+(t-1)*N)])*X[2*(n+t*N)];
                    Y[2*(n+t*N)+1] = a*Y[2*(n+(t-1)*N)+1] + b*(gamma[n]-beta[n]*Y[2*(n+(t-1)*N)+1])*X[2*(n+t*N)+1];
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                nT = n*T; Y[2*nT] = b*gamma[n]*X[2*nT]; Y[2*nT+1] = b*gamma[n]*X[2*nT+1];
                for (t=1; t<T; t++)
                {
                    Y[2*(nT+t)] = a*Y[2*(nT+t-1)] + b*(gamma[n]-beta[n]*Y[2*(nT+t-1)])*X[2*(nT+t)];
                    Y[2*(nT+t)+1] = a*Y[2*(nT+t-1)+1] + b*(gamma[n]-beta[n]*Y[2*(nT+t-1)+1])*X[2*(nT+t)+1];
                }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                nT = n*T; Y[2*nT] = b*gamma[n]*X[2*nT]; Y[2*nT+1] = b*gamma[n]*X[2*nT+1];
                for (t=1; t<T; t++)
                {
                    Y[2*(nT+t)] = a*Y[2*(nT+t-1)] + b*(gamma[n]-beta[n]*Y[2*(nT+t-1)])*X[2*(nT+t)];
                    Y[2*(nT+t)+1] = a*Y[2*(nT+t-1)+1] + b*(gamma[n]-beta[n]*Y[2*(nT+t-1)+1])*X[2*(nT+t)+1];
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                Y[2*n] = b*gamma[n]*X[2*n]; Y[2*n+1] = b*gamma[n]*X[2*n+1];
                for (t=1; t<T; t++)
                {
                    Y[2*(n+t*N)] = a*Y[2*(n+(t-1)*N)] + b*(gamma[n]-beta[n]*Y[2*(n+(t-1)*N)])*X[2*(n+t*N)];
                    Y[2*(n+t*N)+1] = a*Y[2*(n+(t-1)*N)+1] + b*(gamma[n]-beta[n]*Y[2*(n+(t-1)*N)+1])*X[2*(n+t*N)+1];
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in grossberg_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int grossberg_inplace_s (float *X, const float *tau, const float *alpha, const float *beta, const float *gamma, const int N, const int T, const int dim, const char iscolmajor, const float fs)
{
    int n, t, nT;
    float a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in grossberg_inplace_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in grossberg_inplace_s: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0f) { fprintf(stderr,"error in grossberg_inplace_s: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0f) { fprintf(stderr,"error in grossberg_inplace_s: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0f) { fprintf(stderr,"error in grossberg_inplace_s: alphas must be nonnegative\n"); return 1; }
    //if (beta[0]<0.0f) { fprintf(stderr,"error in grossberg_inplace_s: betas must be nonnegative\n"); return 1; }

    if (N==1)
    {
        a = expf(-1.0f/(fs*tau[0])); b = 1.0f - a; a -= b*alpha[0];
        X[0] = b*gamma[0]*X[0];
        for (t=1; t<T; t++) { X[t] = a*X[t-1] + b*(gamma[0]-beta[0]*X[t-1])*X[t]; }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                X[n] = b*gamma[n]*X[n];
                for (t=1; t<T; t++) { X[n+t*N] = a*X[n+(t-1)*N] + b*(gamma[n]-beta[n]*X[n+(t-1)*N])*X[n+t*N]; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                nT = n*T; X[nT] = b*gamma[n]*X[nT];
                for (t=1; t<T; t++) { X[nT+t] = a*X[nT+t-1] + b*(gamma[n]-beta[n]*X[nT+t-1])*X[nT+t]; }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                nT = n*T; X[nT] = b*gamma[n]*X[nT];
                for (t=1; t<T; t++) { X[nT+t] = a*X[nT+t-1] + b*(gamma[n]-beta[n]*X[nT+t-1])*X[nT+t]; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                X[n] = b*gamma[n]*X[n];
                for (t=1; t<T; t++) { X[n+t*N] = a*X[n+(t-1)*N] + b*(gamma[n]-beta[n]*X[n+(t-1)*N])*X[n+t*N]; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in grossberg_inplace_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int grossberg_inplace_d (double *X, const double *tau, const double *alpha, const double *beta, const double *gamma, const int N, const int T, const int dim, const char iscolmajor, const double fs)
{
    int n, t, nT;
    double a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in grossberg_inplace_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in grossberg_inplace_d: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0) { fprintf(stderr,"error in grossberg_inplace_d: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0) { fprintf(stderr,"error in grossberg_d: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0) { fprintf(stderr,"error in grossberg_inplace_d: alphas must be nonnegative\n"); return 1; }
    //if (beta[0]<0.0) { fprintf(stderr,"error in grossberg_inplace_d: betas must be nonnegative\n"); return 1; }

    if (N==1)
    {
        a = exp(-1.0/(fs*tau[0])); b = 1.0 - a; a -= b*alpha[0];
        X[0] = b*gamma[0]*X[0];
        for (t=1; t<T; t++) { X[t] = a*X[t-1] + b*(gamma[0]-beta[0]*X[t-1])*X[t]; }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                X[n] = b*gamma[n]*X[n];
                for (t=1; t<T; t++) { X[n+t*N] = a*X[n+(t-1)*N] + b*(gamma[n]-beta[n]*X[n+(t-1)*N])*X[n+t*N]; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                nT = n*T; X[nT] = b*gamma[n]*X[nT];
                for (t=1; t<T; t++) { X[nT+t] = a*X[nT+t-1] + b*(gamma[n]-beta[n]*X[nT+t-1])*X[nT+t]; }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                nT = n*T; X[nT] = b*gamma[n]*X[nT];
                for (t=1; t<T; t++) { X[nT+t] = a*X[nT+t-1] + b*(gamma[n]-beta[n]*X[nT+t-1])*X[nT+t]; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                X[n] = b*gamma[n]*X[n];
                for (t=1; t<T; t++) { X[n+t*N] = a*X[n+(t-1)*N] + b*(gamma[n]-beta[n]*X[n+(t-1)*N])*X[n+t*N]; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in grossberg_inplace_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int grossberg_inplace_c (float *X, const float *tau, const float *alpha, const float *beta, const float *gamma, const int N, const int T, const int dim, const char iscolmajor, const float fs)
{
    int n, t, nT;
    float a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in grossberg_inplace_c: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in grossberg_inplace_c: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0f) { fprintf(stderr,"error in grossberg_inplace_c: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0f) { fprintf(stderr,"error in grossberg_inplace_c: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0f) { fprintf(stderr,"error in grossberg_inplace_c: alphas must be nonnegative\n"); return 1; }
    //if (beta[0]<0.0f) { fprintf(stderr,"error in grossberg_inplace_c: betas must be nonnegative\n"); return 1; }

    if (N==1)
    {
        a = expf(-1.0f/(fs*tau[0])); b = 1.0f - a; a -= b*alpha[0];
        X[0] = b*gamma[0]*X[0]; X[1] = b*gamma[0]*X[1];
        for (t=1; t<T; t++)
        {
            X[2*t] = a*X[2*(t-1)] + b*(gamma[0]-beta[0]*X[2*(t-1)])*X[2*t];
            X[2*t+1] = a*X[2*(t-1)+1] + b*(gamma[0]-beta[0]*X[2*(t-1)+1])*X[2*t+1];
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                X[2*n] = b*gamma[n]*X[2*n]; X[2*n+1] = b*gamma[n]*X[2*n+1];
                for (t=1; t<T; t++)
                {
                    X[2*(n+t*N)] = a*X[2*(n+(t-1)*N)] + b*(gamma[n]-beta[n]*X[2*(n+(t-1)*N)])*X[2*(n+t*N)];
                    X[2*(n+t*N)+1] = a*X[2*(n+(t-1)*N)+1] + b*(gamma[n]-beta[n]*X[2*(n+(t-1)*N)+1])*X[2*(n+t*N)+1];
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                nT = n*T; X[2*nT] = b*gamma[n]*X[2*nT]; X[2*nT+1] = b*gamma[n]*X[2*nT+1];
                for (t=1; t<T; t++)
                {
                    X[2*(nT+t)] = a*X[2*(nT+t-1)] + b*(gamma[n]-beta[n]*X[2*(nT+t-1)])*X[2*(nT+t)];
                    X[2*(nT+t)+1] = a*X[2*(nT+t-1)+1] + b*(gamma[n]-beta[n]*X[2*(nT+t-1)+1])*X[2*(nT+t)+1];
                }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                nT = n*T; X[2*nT] = b*gamma[n]*X[2*nT]; X[2*nT+1] = b*gamma[n]*X[2*nT+1];
                for (t=1; t<T; t++)
                {
                    X[2*(nT+t)] = a*X[2*(nT+t-1)] + b*(gamma[n]-beta[n]*X[2*(nT+t-1)])*X[2*(nT+t)];
                    X[2*(nT+t)+1] = a*X[2*(nT+t-1)+1] + b*(gamma[n]-beta[n]*X[2*(nT+t-1)+1])*X[2*(nT+t)+1];
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                X[2*n] = b*gamma[n]*X[2*n]; X[2*n+1] = b*gamma[n]*X[2*n+1];
                for (t=1; t<T; t++)
                {
                    X[2*(n+t*N)] = a*X[2*(n+(t-1)*N)] + b*(gamma[n]-beta[n]*X[2*(n+(t-1)*N)])*X[2*(n+t*N)];
                    X[2*(n+t*N)+1] = a*X[2*(n+(t-1)*N)+1] + b*(gamma[n]-beta[n]*X[2*(n+(t-1)*N)+1])*X[2*(n+t*N)+1];
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in grossberg_inplace_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int grossberg_inplace_z (double *X, const double *tau, const double *alpha, const double *beta, const double *gamma, const int N, const int T, const int dim, const char iscolmajor, const double fs)
{
    int n, t, nT;
    double a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in grossberg_inplace_z: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in grossberg_inplace_z: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0) { fprintf(stderr,"error in grossberg_inplace_z: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0) { fprintf(stderr,"error in grossberg_inplace_z: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0) { fprintf(stderr,"error in grossberg_inplace_z: alphas must be nonnegative\n"); return 1; }
    //if (beta[0]<0.0) { fprintf(stderr,"error in grossberg_inplace_z: betas must be nonnegative\n"); return 1; }

    if (N==1)
    {
        a = exp(-1.0/(fs*tau[0])); b = 1.0 - a; a -= b*alpha[0];
        X[0] = b*gamma[0]*X[0]; X[1] = b*gamma[0]*X[1];
        for (t=1; t<T; t++)
        {
            X[2*t] = a*X[2*(t-1)] + b*(gamma[0]-beta[0]*X[2*(t-1)])*X[2*t];
            X[2*t+1] = a*X[2*(t-1)+1] + b*(gamma[0]-beta[0]*X[2*(t-1)+1])*X[2*t+1];
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                X[2*n] = b*gamma[n]*X[2*n]; X[2*n+1] = b*gamma[n]*X[2*n+1];
                for (t=1; t<T; t++)
                {
                    X[2*(n+t*N)] = a*X[2*(n+(t-1)*N)] + b*(gamma[n]-beta[n]*X[2*(n+(t-1)*N)])*X[2*(n+t*N)];
                    X[2*(n+t*N)+1] = a*X[2*(n+(t-1)*N)+1] + b*(gamma[n]-beta[n]*X[2*(n+(t-1)*N)+1])*X[2*(n+t*N)+1];
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                nT = n*T; X[2*nT] = b*gamma[n]*X[2*nT]; X[2*nT+1] = b*gamma[n]*X[2*nT+1];
                for (t=1; t<T; t++)
                {
                    X[2*(nT+t)] = a*X[2*(nT+t-1)] + b*(gamma[n]-beta[n]*X[2*(nT+t-1)])*X[2*(nT+t)];
                    X[2*(nT+t)+1] = a*X[2*(nT+t-1)+1] + b*(gamma[n]-beta[n]*X[2*(nT+t-1)+1])*X[2*(nT+t)+1];
                }
            }
        }
    }
    else if (dim==1)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                nT = n*T; X[2*nT] = b*gamma[n]*X[2*nT]; X[2*nT+1] = b*gamma[n]*X[2*nT+1];
                for (t=1; t<T; t++)
                {
                    X[2*(nT+t)] = a*X[2*(nT+t-1)] + b*(gamma[n]-beta[n]*X[2*(nT+t-1)])*X[2*(nT+t)];
                    X[2*(nT+t)+1] = a*X[2*(nT+t-1)+1] + b*(gamma[n]-beta[n]*X[2*(nT+t-1)+1])*X[2*(nT+t)+1];
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                X[2*n] = b*gamma[n]*X[2*n]; X[2*n+1] = b*gamma[n]*X[2*n+1];
                for (t=1; t<T; t++)
                {
                    X[2*(n+t*N)] = a*X[2*(n+(t-1)*N)] + b*(gamma[n]-beta[n]*X[2*(n+(t-1)*N)])*X[2*(n+t*N)];
                    X[2*(n+t*N)+1] = a*X[2*(n+(t-1)*N)+1] + b*(gamma[n]-beta[n]*X[2*(n+(t-1)*N)+1])*X[2*(n+t*N)+1];
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in grossberg_inplace_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
