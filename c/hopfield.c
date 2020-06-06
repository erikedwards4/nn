//CELL (~soma) stage consisting of simple 1st-order integrator with time-constant tau (in seconds),
//along with a negative feedback from just after to just before the integrator.
//This is the discrete-time implementation of the Hopfield model [Cichocki & Unbehauen 1993].

//For dim=0: Y[n,t] = a[n]*Y[n,t-1] + b[n]*(X[n,t]-alpha[n]*Y[n,t-1]);
//For dim=1: Y[t,n] = a[n]*Y[t-1,n] + b[n]*(X[t,n]-alpha[n]*Y[t-1,n]);
//where a[n] = exp(-1/(fs*tau[n])) and b[n] = 1 - a[n].

//Thus, setting a[n] to a[n]-b[n]*alpha[n], this gives:
//For dim=0: Y[n,t] = a[n]*Y[n,t-1] + b[n]*X[n,t];
//For dim=1: Y[t,n] = a[n]*Y[t-1,n] + b[n]*X[t,n];

//tau and alpha are vectors of length N, so that each neuron has its own parameters.

//For complex inputs, real and imag parts are filtered separately, so params are real.


#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int hopfield_s (float *Y, const float *X, const float *tau, const float *alpha, const int N, const int T, const int dim, const char iscolmajor, const float fs);
int hopfield_d (double *Y, const double *X, const double *tau, const double *alpha, const int N, const int T, const int dim, const char iscolmajor, const double fs);
int hopfield_c (float *Y, const float *X, const float *tau, const float *alpha, const int N, const int T, const int dim, const char iscolmajor, const float fs);
int hopfield_z (double *Y, const double *X, const double *tau, const double *alpha, const int N, const int T, const int dim, const char iscolmajor, const double fs);

int hopfield_inplace_s (float *X, const float *tau, const float *alpha, const int N, const int T, const int dim, const char iscolmajor, const float fs);
int hopfield_inplace_d (double *X, const double *tau, const double *alpha, const int N, const int T, const int dim, const char iscolmajor, const double fs);
int hopfield_inplace_c (float *X, const float *tau, const float *alpha, const int N, const int T, const int dim, const char iscolmajor, const float fs);
int hopfield_inplace_z (double *X, const double *tau, const double *alpha, const int N, const int T, const int dim, const char iscolmajor, const double fs);


int hopfield_s (float *Y, const float *X, const float *tau, const float *alpha, const int N, const int T, const int dim, const char iscolmajor, const float fs)
{
    int n, t, nT;
    float a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in hopfield_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in hopfield_s: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0f) { fprintf(stderr,"error in hopfield_s: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0f) { fprintf(stderr,"error in hopfield_s: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0f) { fprintf(stderr,"error in hopfield_s: alphas must be nonnegative\n"); return 1; }

    if (N==1)
    {
        a = expf(-1.0f/(fs*tau[0])); b = 1.0f - a; a -= b*alpha[0];
        Y[0] = b*X[0];
        for (t=1; t<T; t++) { Y[t] = a*Y[t-1] + b*X[t]; }
        //for (t=1; t<T; t++) { Y[t] = a*Y[t-1] + b*(X[t]-alpha[0]*Y[t-1]); }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                Y[n] = b*X[n];
                for (t=1; t<T; t++) { Y[n+t*N] = a*Y[n+(t-1)*N] + b*X[n+t*N]; }
                //for (t=1; t<T; t++) { Y[n+t*N] = a*Y[n+(t-1)*N] + b*(X[n+t*N]-alpha[n]*Y[n+(t-1)*N]); }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                nT = n*T; Y[nT] = b*X[nT];
                for (t=1; t<T; t++) { Y[nT+t] = a*Y[nT+t-1] + b*X[nT+t]; }
                //for (t=1; t<T; t++) { Y[nT+t] = a*Y[nT+t-1] + b*(X[nT+t]-alpha[n]*Y[nT+t-1]); }
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
                nT = n*T; Y[nT] = b*X[nT];
                for (t=1; t<T; t++) { Y[nT+t] = a*Y[nT+t-1] + b*X[nT+t]; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                Y[n] = b*X[n];
                for (t=1; t<T; t++) { Y[n+t*N] = a*Y[n+(t-1)*N] + b*X[n+t*N]; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in hopfield_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int hopfield_d (double *Y, const double *X, const double *tau, const double *alpha, const int N, const int T, const int dim, const char iscolmajor, const double fs)
{
    int n, t, nT;
    double a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in hopfield_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in hopfield_d: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0) { fprintf(stderr,"error in hopfield_d: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0) { fprintf(stderr,"error in hopfield_d: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0) { fprintf(stderr,"error in hopfield_d: alphas must be nonnegative\n"); return 1; }

    if (N==1)
    {
        a = exp(-1.0/(fs*tau[0])); b = 1.0 - a; a -= b*alpha[0];
        Y[0] = b*X[0];
        for (t=1; t<T; t++) { Y[t] = a*Y[t-1] + b*X[t]; }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                Y[n] = b*X[n];
                for (t=1; t<T; t++) { Y[n+t*N] = a*Y[n+(t-1)*N] + b*X[n+t*N]; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                nT = n*T; Y[nT] = b*X[nT];
                for (t=1; t<T; t++) { Y[nT+t] = a*Y[nT+t-1] + b*X[nT+t]; }
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
                nT = n*T; Y[nT] = b*X[nT];
                for (t=1; t<T; t++) { Y[nT+t] = a*Y[nT+t-1] + b*X[nT+t]; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                Y[n] = b*X[n];
                for (t=1; t<T; t++) { Y[n+t*N] = a*Y[n+(t-1)*N] + b*X[n+t*N]; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in hopfield_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int hopfield_c (float *Y, const float *X, const float *tau, const float *alpha, const int N, const int T, const int dim, const char iscolmajor, const float fs)
{
    int n, t, nT;
    float a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in hopfield_c: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in hopfield_c: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0f) { fprintf(stderr,"error in hopfield_c: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0f) { fprintf(stderr,"error in hopfield_c: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0f) { fprintf(stderr,"error in hopfield_c: alphas must be nonnegative\n"); return 1; }

    if (N==1)
    {
        a = expf(-1.0f/(fs*tau[0])); b = 1.0f - a; a -= b*alpha[0];
        Y[0] = b*X[0]; Y[1] = b*X[1];
        for (t=1; t<T; t++)
        {
            Y[2*t] = a*Y[2*(t-1)] + b*X[2*t];
            Y[2*t+1] = a*Y[2*(t-1)+1] + b*X[2*t+1];
            //Y[2*t] = a*Y[2*(t-1)] + b*(X[2*t]-alpha[0]*Y[2*(t-1)]);
            //Y[2*t+1] = a*Y[2*(t-1)+1] + b*(X[2*t+1]-alpha[0]*Y[2*(t-1)+1]);
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                Y[2*n] = b*X[2*n]; Y[2*n+1] = b*X[2*n+1];
                for (t=1; t<T; t++)
                {
                    Y[2*(n+t*N)] = a*Y[2*(n+(t-1)*N)] + b*X[2*(n+t*N)];
                    Y[2*(n+t*N)+1] = a*Y[2*(n+(t-1)*N)+1] + b*X[2*(n+t*N)+1];
                    //Y[2*(n+t*N)] = a*Y[2*(n+(t-1)*N)] + b*(X[2*(n+t*N)]-alpha[n]*Y[2*(n+(t-1)*N)]);
                    //Y[2*(n+t*N)+1] = a*Y[n+(t-1)*N] + b*(X[n+t*N]-alpha[n]*Y[2*(n+(t-1)*N)+1]);
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                nT = n*T; Y[2*nT] = b*X[2*nT]; Y[2*nT+1] = b*X[2*nT+1];
                for (t=1; t<T; t++)
                {
                    Y[2*(nT+t)] = a*Y[2*(nT+t-1)] + b*X[2*(nT+t)];
                    Y[2*(nT+t)+1] = a*Y[2*(nT+t-1)+1] + b*X[2*(nT+t)+1];
                    //Y[2*(nT+t)] = a*Y[2*(nT+t-1)] + b*(X[2*(nT+t)]-alpha[n]*Y[2*(nT+t-1)]);
                    //Y[2*(nT+t)+1] = a*Y[2*(nT+t-1)+1] + b*(X[2*(nT+t)+1]-alpha[n]*Y[2*(nT+t-1)+1]);
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
                nT = n*T; Y[2*nT] = b*X[2*nT]; Y[2*nT+1] = b*X[2*nT+1];
                for (t=1; t<T; t++)
                {
                    Y[2*(nT+t)] = a*Y[2*(nT+t-1)] + b*X[2*(nT+t)];
                    Y[2*(nT+t)+1] = a*Y[2*(nT+t-1)+1] + b*X[2*(nT+t)+1];
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                Y[2*n] = b*X[2*n]; Y[2*n+1] = b*X[2*n+1];
                for (t=1; t<T; t++)
                {
                    Y[2*(n+t*N)] = a*Y[2*(n+(t-1)*N)] + b*X[2*(n+t*N)];
                    Y[2*(n+t*N)+1] = a*Y[2*(n+(t-1)*N)+1] + b*X[2*(n+t*N)+1];
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in hopfield_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int hopfield_z (double *Y, const double *X, const double *tau, const double *alpha, const int N, const int T, const int dim, const char iscolmajor, const double fs)
{
    int n, t, nT;
    double a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in hopfield_z: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in hopfield_z: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0) { fprintf(stderr,"error in hopfield_z: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0) { fprintf(stderr,"error in hopfield_z: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0) { fprintf(stderr,"error in hopfield_z: alphas must be nonnegative\n"); return 1; }

    if (N==1)
    {
        a = exp(-1.0/(fs*tau[0])); b = 1.0 - a; a -= b*alpha[0];
        Y[0] = b*X[0]; Y[1] = b*X[1];
        for (t=1; t<T; t++)
        {
            Y[2*t] = a*Y[2*(t-1)] + b*X[2*t];
            Y[2*t+1] = a*Y[2*(t-1)+1] + b*X[2*t+1];
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                Y[2*n] = b*X[2*n]; Y[2*n+1] = b*X[2*n+1];
                for (t=1; t<T; t++)
                {
                    Y[2*(n+t*N)] = a*Y[2*(n+(t-1)*N)] + b*X[2*(n+t*N)];
                    Y[2*(n+t*N)+1] = a*Y[2*(n+(t-1)*N)+1] + b*X[2*(n+t*N)+1];
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                nT = n*T; Y[2*nT] = b*X[2*nT]; Y[2*nT+1] = b*X[2*nT+1];
                for (t=1; t<T; t++)
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
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                nT = n*T; Y[2*nT] = b*X[2*nT]; Y[2*nT+1] = b*X[2*nT+1];
                for (t=1; t<T; t++)
                {
                    Y[2*(nT+t)] = a*Y[2*(nT+t-1)] + b*X[2*(nT+t)];
                    Y[2*(nT+t)+1] = a*Y[2*(nT+t-1)+1] + b*X[2*(nT+t)+1];
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                Y[2*n] = b*X[2*n]; Y[2*n+1] = b*X[2*n+1];
                for (t=1; t<T; t++)
                {
                    Y[2*(n+t*N)] = a*Y[2*(n+(t-1)*N)] + b*X[2*(n+t*N)];
                    Y[2*(n+t*N)+1] = a*Y[2*(n+(t-1)*N)+1] + b*X[2*(n+t*N)+1];
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in hopfield_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int hopfield_inplace_s (float *X, const float *tau, const float *alpha, const int N, const int T, const int dim, const char iscolmajor, const float fs)
{
    int n, t, nT;
    float a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in hopfield_inplace_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in hopfield_inplace_s: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0f) { fprintf(stderr,"error in hopfield_inplace_s: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0f) { fprintf(stderr,"error in hopfield_inplace_s: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0f) { fprintf(stderr,"error in hopfield_inplace_s: alphas must be nonnegative\n"); return 1; }

    if (N==1)
    {
        a = expf(-1.0f/(fs*tau[0])); b = 1.0f - a; a -= b*alpha[0];
        X[0] *= b;
        for (t=1; t<T; t++) { X[t] = a*X[t-1] + b*X[t]; }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                X[n] *= b;
                for (t=1; t<T; t++) { X[n+t*N] = a*X[n+(t-1)*N] + b*X[n+t*N]; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                nT = n*T; X[nT] *= b;
                for (t=1; t<T; t++) { X[nT+t] = a*X[nT+t-1] + b*X[nT+t]; }
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
                nT = n*T; X[nT] *= b;
                for (t=1; t<T; t++) { X[nT+t] = a*X[nT+t-1] + b*X[nT+t]; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                X[n] *= b;
                for (t=1; t<T; t++) { X[n+t*N] = a*X[n+(t-1)*N] + b*X[n+t*N]; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in hopfield_inplace_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int hopfield_inplace_d (double *X, const double *tau, const double *alpha, const int N, const int T, const int dim, const char iscolmajor, const double fs)
{
    int n, t, nT;
    double a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in hopfield_inplace_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in hopfield_inplace_d: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0) { fprintf(stderr,"error in hopfield_inplace_d: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0) { fprintf(stderr,"error in hopfield_d: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0) { fprintf(stderr,"error in hopfield_inplace_d: alphas must be nonnegative\n"); return 1; }

    if (N==1)
    {
        a = exp(-1.0/(fs*tau[0])); b = 1.0 - a; a -= b*alpha[0];
        X[0] *= b;
        for (t=1; t<T; t++) { X[t] = a*X[t-1] + b*X[t]; }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                X[n] *= b;
                for (t=1; t<T; t++) { X[n+t*N] = a*X[n+(t-1)*N] + b*X[n+t*N]; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                nT = n*T; X[nT] *= b;
                for (t=1; t<T; t++) { X[nT+t] = a*X[nT+t-1] + b*X[nT+t]; }
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
                nT = n*T; X[nT] *= b;
                for (t=1; t<T; t++) { X[nT+t] = a*X[nT+t-1] + b*X[nT+t]; }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                X[n] *= b;
                for (t=1; t<T; t++) { X[n+t*N] = a*X[n+(t-1)*N] + b*X[n+t*N]; }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in hopfield_inplace_d: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int hopfield_inplace_c (float *X, const float *tau, const float *alpha, const int N, const int T, const int dim, const char iscolmajor, const float fs)
{
    int n, t, nT;
    float a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in hopfield_inplace_c: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in hopfield_inplace_c: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0f) { fprintf(stderr,"error in hopfield_inplace_c: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0f) { fprintf(stderr,"error in hopfield_inplace_c: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0f) { fprintf(stderr,"error in hopfield_inplace_c: alphas must be nonnegative\n"); return 1; }

    if (N==1)
    {
        a = expf(-1.0f/(fs*tau[0])); b = 1.0f - a; a -= b*alpha[0];
        X[0] *= b; X[1] *= b;
        for (t=1; t<T; t++)
        {
            X[2*t] = a*X[2*t-2] + b*X[2*t];
            X[2*t+1] = a*X[2*t-1] + b*X[2*t+1];
            //X[2*t] = a*X[2*t-2] + b*(X[2*t]-alpha[0]*X[2*(t-1)]);
            //X[2*t+1] = a*X[2*t-1] + b*(X[2*t+1]-alpha[0]*Y[2*(t-1)+1]);
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                X[2*n] *= b; X[2*n+1] *= b;
                for (t=1; t<T; t++)
                {
                    X[2*(n+t*N)] = a*X[2*(n+(t-1)*N)] + b*X[2*(n+t*N)];
                    X[2*(n+t*N)+1] = a*X[2*(n+(t-1)*N)+1] + b*X[2*(n+t*N)+1];
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                nT = n*T; X[2*nT] *= b; X[2*nT+1] *= b;
                for (t=1; t<T; t++)
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
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                nT = n*T; X[2*nT] *= b; X[2*nT+1] *= b;
                for (t=1; t<T; t++)
                {
                    X[2*(nT+t)] = a*X[2*(nT+t-1)] + b*X[2*(nT+t)];
                    X[2*(nT+t)+1] = a*X[2*(nT+t-1)+1] + b*X[2*(nT+t)+1];
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = expf(-1.0f/(fs*tau[n])); b = 1.0f - a; a -= b*alpha[n];
                X[2*n] *= b; X[2*n+1] *= b;
                for (t=1; t<T; t++)
                {
                    X[2*(n+t*N)] = a*X[2*(n+(t-1)*N)] + b*X[2*(n+t*N)];
                    X[2*(n+t*N)+1] = a*X[2*(n+(t-1)*N)+1] + b*X[2*(n+t*N)+1];
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in hopfield_inplace_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int hopfield_inplace_z (double *X, const double *tau, const double *alpha, const int N, const int T, const int dim, const char iscolmajor, const double fs)
{
    int n, t, nT;
    double a, b;

    //Checks
    if (N<1) { fprintf(stderr,"error in hopfield_inplace_z: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in hopfield_inplace_z: T (num time points) must be positive\n"); return 1; }
    if (fs<=0.0) { fprintf(stderr,"error in hopfield_inplace_z: fs (sample rate) must be positive\n"); return 1; }
    if (tau[0]<=0.0) { fprintf(stderr,"error in hopfield_inplace_z: taus must be positive\n"); return 1; }
    if (alpha[0]<0.0) { fprintf(stderr,"error in hopfield_inplace_z: alphas must be nonnegative\n"); return 1; }

    if (N==1)
    {
        a = exp(-1.0/(fs*tau[0])); b = 1.0 - a; a -= b*alpha[0];
        X[0] *= b; X[1] *= b;
        for (t=1; t<T; t++)
        {
            X[2*t] = a*X[2*(t-1)] + b*X[2*t];
            X[2*t+1] = a*X[2*(t-1)+1] + b*X[2*t+1];
        }
    }
    else if (dim==0)
    {
        if (iscolmajor)
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                X[2*n] *= b; X[2*n+1] *= b;
                for (t=1; t<T; t++)
                {
                    X[2*(n+t*N)] = a*X[2*(n+(t-1)*N)] + b*X[2*(n+t*N)];
                    X[2*(n+t*N)+1] = a*X[2*(n+(t-1)*N)+1] + b*X[2*(n+t*N)+1];
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                nT = n*T; X[2*nT] *= b; X[2*nT+1] *= b;
                for (t=1; t<T; t++)
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
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                nT = n*T; X[2*nT] *= b; X[2*nT+1] *= b;
                for (t=1; t<T; t++)
                {
                    X[2*(nT+t)] = a*X[2*(nT+t-1)] + b*X[2*(nT+t)];
                    X[2*(nT+t)+1] = a*X[2*(nT+t-1)+1] + b*X[2*(nT+t)+1];
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                a = exp(-1.0/(fs*tau[n])); b = 1.0 - a; a -= b*alpha[n];
                X[2*n] *= b; X[2*n+1] *= b;
                for (t=1; t<T; t++)
                {
                    X[2*(n+t*N)] = a*X[2*(n+(t-1)*N)] + b*X[2*(n+t*N)];
                    X[2*(n+t*N)+1] = a*X[2*(n+(t-1)*N)+1] + b*X[2*(n+t*N)+1];
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in hopfield_inplace_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
