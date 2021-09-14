//CELL (~soma) stage consisting of a 1st-order integrator with time-constant tau (in seconds),
//along with a negative feedback from just after to just before the integrator,
//and an automatic gain control loop of gamma-beta*y(t-1) that multiplies the driving input.
//This is the discrete-time implementation of the Grossberg model [Cichocki & Unbehauen 1993, p.58].

//This version "2" uses separate excitatory (Xe) and inhibitory (Xi) driving inputs.
//For the Xi inputs, the feedback beta term is added rather than subtracted.

//For dim=0: Y[n,t] = a[n]*Y[n,t-1] + b[n]*(ge[n,t]*Xe[n,t]-gi[n,t]*Xi[n,t]-alpha[n]*Y[n,t-1]);
//For dim=1: Y[t,n] = a[n]*Y[t-1,n] + b[n]*(ge[t,n]*Xe[t,n]-gi[t,n]*Xi[t,n]-alpha[n]*Y[t-1,n]);
//where a[n] = exp(-1/(fs*tau[n])) and b[n] = 1 - a[n].
//and ge[n,t] = gammae[n] - betae[n]*Y[n,t-1].
//and gi[n,t] = gammai[n] - betai[n]*Y[n,t-1].
//and ge[t,n] = gammae[n] - betae[n]*Y[t-1,n].
//and gi[t,n] = gammai[n] - betai[n]*Y[t-1,n].

//tau, alpha, betae, betai, gammae and gammai are vectors of length N, so that each neuron has its own parameters.


#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int grossberg2_s (float *Y, const float *Xe, const float *Xi, const float *tau, const float *alpha, const float *betae, const float *betai, const float *gammae, const float *gammai, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int grossberg2_d (double *Y, const double *Xe, const double *Xi, const double *tau, const double *alpha, const double *betae, const double *betai, const double *gammae, const double *gammai, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);

int grossberg2_inplace_s (float *Xe, const float *Xi, const float *tau, const float *alpha, const float *betae, const float *betai, const float *gammae, const float *gammai, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs);
int grossberg2_inplace_d (double *Xe, const double *Xi, const double *tau, const double *alpha, const double *betae, const double *betai, const double *gammae, const double *gammai, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs);


int grossberg2_s (float *Y, const float *Xe, const float *Xi, const float *tau, const float *alpha, const float *betae, const float *betai, const float *gammae, const float *gammai, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs)
{
    if (fs<=0.0f) { fprintf(stderr,"error in grossberg2_s: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0f) { fprintf(stderr,"error in grossberg2_s: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0f) { fprintf(stderr,"error in grossberg2_s: alphas must be nonnegative\n"); return 1; }
    if (betae[0]<0.0f) { fprintf(stderr,"error in grossberg2_s: betas must be nonnegative\n"); return 1; }
    if (betai[0]<0.0f) { fprintf(stderr,"error in grossberg2_s: betas must be nonnegative\n"); return 1; }

    size_t nT, tN;
    float a, b;

    if (N==1u)
    {
        a = expf(-1.0f/(fs*tau[0])); b = 1.0f - a; a -= b*alpha[0];
        Y[0] = b*(gammae[0]*Xe[0]-gammai[0]*Xi[0]);
        for (size_t t=1; t<T; ++t) { Y[t] = a*Y[t-1] + b*((gammae[0]-betae[0]*Y[t-1])*Xe[t]-(gammai[0]-betai[0]*Y[t-1])*Xi[t]); }
    }
    else if (dim==0u)
    {
        if (iscolmajor)
        {
            for (size_t n=0u; n<N; ++n)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                Y[n] = b*(gammae[n]*Xe[n]-gammai[n]*Xi[n]);
                for (size_t t=1; t<T; ++t)
                {
                    tN = t*N;
                    Y[n+tN] = a*Y[n+tN-N] + b*((gammae[n]-betae[n]*Y[n+tN-N])*Xe[n+tN]-(gammai[n]-betai[n]*Y[n+tN-N])*Xi[n+tN]);
                }
            }
        }
        else
        {
            for (size_t n=0u; n<N; ++n)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                nT = n*T; Y[nT] = b*(gammae[n]*Xe[nT]-gammai[n]*Xi[nT]);
                for (size_t t=1; t<T; ++t)
                {
                    Y[nT+t] = a*Y[nT+t-1] + b*((gammae[n]-betae[n]*Y[nT+t-1])*Xe[nT+t]-(gammai[n]-betai[n]*Y[nT+t-1])*Xi[nT+t]);
                }
            }
        }
    }
    else if (dim==1u)
    {
        if (iscolmajor)
        {
            for (size_t n=0u; n<N; ++n)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                nT = n*T; Y[nT] = b*(gammae[n]*Xe[nT]-gammai[n]*Xi[nT]);
                for (size_t t=1; t<T; ++t)
                {
                    Y[nT+t] = a*Y[nT+t-1] + b*((gammae[n]-betae[n]*Y[nT+t-1])*Xe[nT+t]-(gammai[n]-betai[n]*Y[nT+t-1])*Xi[nT+t]);
                }
            }
        }
        else
        {
            for (size_t n=0u; n<N; ++n)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                Y[n] = b*(gammae[n]*Xe[n]-gammai[n]*Xi[n]);
                for (size_t t=1; t<T; ++t)
                {
                    tN = t*N;
                    Y[n+tN] = a*Y[n+tN-N] + b*((gammae[n]-betae[n]*Y[n+tN-N])*Xe[n+tN]-(gammai[n]-betai[n]*Y[n+tN-N])*Xi[n+tN]);
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in grossberg2_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int grossberg2_d (double *Y, const double *Xe, const double *Xi, const double *tau, const double *alpha, const double *betae, const double *betai, const double *gammae, const double *gammai, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs)
{
    if (fs<=0.0) { fprintf(stderr,"error in grossberg2_d: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0) { fprintf(stderr,"error in grossberg2_d: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0) { fprintf(stderr,"error in grossberg2_d: alphas must be nonnegative\n"); return 1; }
    if (betae[0]<0.0) { fprintf(stderr,"error in grossberg2_d: betas must be nonnegative\n"); return 1; }
    if (betai[0]<0.0) { fprintf(stderr,"error in grossberg2_d: betas must be nonnegative\n"); return 1; }

    size_t nT, tN;
    double a, b;

    if (N==1u)
    {
        a = exp(-1.0/(fs*tau[0])); b = 1.0 - a; a -= b*alpha[0];
        Y[0] = b*(gammae[0]*Xe[0]-gammai[0]*Xi[0]);
        for (size_t t=1; t<T; ++t) { Y[t] = a*Y[t-1] + b*((gammae[0]-betae[0]*Y[t-1])*Xe[t]-(gammai[0]-betai[0]*Y[t-1])*Xi[t]); }
    }
    else if (dim==0u)
    {
        if (iscolmajor)
        {
            for (size_t n=0u; n<N; ++n)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                Y[n] = b*(gammae[n]*Xe[n]-gammai[n]*Xi[n]);
                for (size_t t=1; t<T; ++t)
                {
                    tN = t*N;
                    Y[n+tN] = a*Y[n+tN-N] + b*((gammae[n]-betae[n]*Y[n+tN-N])*Xe[n+tN]-(gammai[n]-betai[n]*Y[n+tN-N])*Xi[n+tN]);
                }
            }
        }
        else
        {
            for (size_t n=0u; n<N; ++n)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                nT = n*T; Y[nT] = b*(gammae[n]*Xe[nT]-gammai[n]*Xi[nT]);
                for (size_t t=1; t<T; ++t)
                {
                    Y[nT+t] = a*Y[nT+t-1] + b*((gammae[n]-betae[n]*Y[nT+t-1])*Xe[nT+t]-(gammai[n]-betai[n]*Y[nT+t-1])*Xi[nT+t]);
                }
            }
        }
    }
    else if (dim==1u)
    {
        if (iscolmajor)
        {
            for (size_t n=0u; n<N; ++n)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                nT = n*T; Y[nT] = b*(gammae[n]*Xe[nT]-gammai[n]*Xi[nT]);
                for (size_t t=1; t<T; ++t)
                {
                    Y[nT+t] = a*Y[nT+t-1] + b*((gammae[n]-betae[n]*Y[nT+t-1])*Xe[nT+t]-(gammai[n]-betai[n]*Y[nT+t-1])*Xi[nT+t]);
                }
            }
        }
        else
        {
            for (size_t n=0u; n<N; ++n)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                Y[n] = b*(gammae[n]*Xe[n]-gammai[n]*Xi[n]);
                for (size_t t=1; t<T; ++t)
                {
                    tN = t*N;
                    Y[n+tN] = a*Y[n+tN-N] + b*((gammae[n]-betae[n]*Y[n+tN-N])*Xe[n+tN]-(gammai[n]-betai[n]*Y[n+tN-N])*Xi[n+tN]);
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in grossberg2_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int grossberg2_inplace_s (float *Xe, const float *Xi, const float *tau, const float *alpha, const float *betae, const float *betai, const float *gammae, const float *gammai, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const float fs)
{
    if (fs<=0.0f) { fprintf(stderr,"error in grossberg2_inplace_s: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0f) { fprintf(stderr,"error in grossberg2_inplace_s: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0f) { fprintf(stderr,"error in grossberg2_inplace_s: alphas must be nonnegative\n"); return 1; }
    if (betae[0]<0.0f) { fprintf(stderr,"error in grossberg2_inplace_s: betas must be nonnegative\n"); return 1; }
    if (betai[0]<0.0f) { fprintf(stderr,"error in grossberg2_inplace_s: betas must be nonnegative\n"); return 1; }

    size_t nT, tN;
    float a, b;

    if (N==1u)
    {
        a = expf(-1.0f/(fs*tau[0])); b = 1.0f - a; a -= b*alpha[0];
        Xe[0] = b*(gammae[0]*Xe[0]-gammai[0]*Xi[0]);
        for (size_t t=1; t<T; ++t) { Xe[t] = a*Xe[t-1] + b*((gammae[0]-betae[0]*Xe[t-1])*Xe[t]-(gammai[0]-betai[0]*Xe[t-1])*Xi[t]); }
    }
    else if (dim==0u)
    {
        if (iscolmajor)
        {
            for (size_t n=0u; n<N; ++n)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                Xe[n] = b*(gammae[n]*Xe[n]-gammai[n]*Xi[n]);
                for (size_t t=1; t<T; ++t)
                {
                    tN = t*N;
                    Xe[n+tN] = a*Xe[n+tN-N] + b*((gammae[n]-betae[n]*Xe[n+tN-N])*Xe[n+tN]-(gammai[n]-betai[n]*Xe[n+tN-N])*Xi[n+tN]);
                }
            }
        }
        else
        {
            for (size_t n=0u; n<N; ++n)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                nT = n*T; Xe[nT] = b*(gammae[n]*Xe[nT]-gammai[n]*Xi[nT]);
                for (size_t t=1; t<T; ++t)
                {
                    Xe[nT+t] = a*Xe[nT+t-1] + b*((gammae[n]-betae[n]*Xe[nT+t-1])*Xe[nT+t]-(gammai[n]-betai[n]*Xe[nT+t-1])*Xi[nT+t]);
                }
            }
        }
    }
    else if (dim==1u)
    {
        if (iscolmajor)
        {
            for (size_t n=0u; n<N; ++n)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                nT = n*T; Xe[nT] = b*(gammae[n]*Xe[nT]-gammai[n]*Xi[nT]);
                for (size_t t=1; t<T; ++t)
                {
                    Xe[nT+t] = a*Xe[nT+t-1] + b*((gammae[n]-betae[n]*Xe[nT+t-1])*Xe[nT+t]-(gammai[n]-betai[n]*Xe[nT+t-1])*Xi[nT+t]);
                }
            }
        }
        else
        {
            for (size_t n=0u; n<N; ++n)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                Xe[n] = b*(gammae[n]*Xe[n]-gammai[n]*Xi[n]);
                for (size_t t=1; t<T; ++t)
                {
                    tN = t*N;
                    Xe[n+tN] = a*Xe[n+tN-N] + b*((gammae[n]-betae[n]*Xe[n+tN-N])*Xe[n+tN]-(gammai[n]-betai[n]*Xe[n+tN-N])*Xi[n+tN]);
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in grossberg2_inplace_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int grossberg2_inplace_d (double *Xe, const double *Xi, const double *tau, const double *alpha, const double *betae, const double *betai, const double *gammae, const double *gammai, const size_t N, const size_t T, const char iscolmajor, const size_t dim, const double fs)
{
    if (fs<=0.0) { fprintf(stderr,"error in grossberg2_inplace_d: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0) { fprintf(stderr,"error in grossberg2_inplace_d: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0) { fprintf(stderr,"error in grossberg2_inplace_d: alphas must be nonnegative\n"); return 1; }
    if (betae[0]<0.0) { fprintf(stderr,"error in grossberg2_inplace_d: betas must be nonnegative\n"); return 1; }
    if (betai[0]<0.0) { fprintf(stderr,"error in grossberg2_inplace_d: betas must be nonnegative\n"); return 1; }
    
    size_t nT, tN;
    double a, b;

    if (N==1u)
    {
        a = exp(-1.0/(fs*tau[0])); b = 1.0 - a; a -= b*alpha[0];
        Xe[0] = b*(gammae[0]*Xe[0]-gammai[0]*Xi[0]);
        for (size_t t=1; t<T; ++t) { Xe[t] = a*Xe[t-1] + b*((gammae[0]-betae[0]*Xe[t-1])*Xe[t]-(gammai[0]-betai[0]*Xe[t-1])*Xi[t]); }
    }
    else if (dim==0u)
    {
        if (iscolmajor)
        {
            for (size_t n=0u; n<N; ++n)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                Xe[n] = b*(gammae[n]*Xe[n]-gammai[n]*Xi[n]);
                for (size_t t=1; t<T; ++t)
                {
                    tN = t*N;
                    Xe[n+tN] = a*Xe[n+tN-N] + b*((gammae[n]-betae[n]*Xe[n+tN-N])*Xe[n+tN]-(gammai[n]-betai[n]*Xe[n+tN-N])*Xi[n+tN]);
                }
            }
        }
        else
        {
            for (size_t n=0u; n<N; ++n)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                nT = n*T; Xe[nT] = b*(gammae[n]*Xe[nT]-gammai[n]*Xi[nT]);
                for (size_t t=1; t<T; ++t)
                {
                    Xe[nT+t] = a*Xe[nT+t-1] + b*((gammae[n]-betae[n]*Xe[nT+t-1])*Xe[nT+t]-(gammai[n]-betai[n]*Xe[nT+t-1])*Xi[nT+t]);
                }
            }
        }
    }
    else if (dim==1u)
    {
        if (iscolmajor)
        {
            for (size_t n=0u; n<N; ++n)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                nT = n*T; Xe[nT] = b*(gammae[n]*Xe[nT]-gammai[n]*Xi[nT]);
                for (size_t t=1; t<T; ++t)
                {
                    Xe[nT+t] = a*Xe[nT+t-1] + b*((gammae[n]-betae[n]*Xe[nT+t-1])*Xe[nT+t]-(gammai[n]-betai[n]*Xe[nT+t-1])*Xi[nT+t]);
                }
            }
        }
        else
        {
            for (size_t n=0u; n<N; ++n)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                Xe[n] = b*(gammae[n]*Xe[n]-gammai[n]*Xi[n]);
                for (size_t t=1; t<T; ++t)
                {
                    tN = t*N;
                    Xe[n+tN] = a*Xe[n+tN-N] + b*((gammae[n]-betae[n]*Xe[n+tN-N])*Xe[n+tN]-(gammai[n]-betai[n]*Xe[n+tN-N])*Xi[n+tN]);
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in grossberg2_inplace_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
