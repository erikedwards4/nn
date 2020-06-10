//This gets maxout layer-wise activation function for X.
//This is just the max along rows or cols of X according to dim.

//For complex input X, output Y is complex with elements having max absolute values.


#include <stdio.h>
#include <cblas.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int maxout_s (float *Y, const float *X, const int N, const int T, const int M, const int dim, const char iscolmajor);
int maxout_d (double *Y, const double *X, const int N, const int T, const int M, const int dim, const char iscolmajor);
int maxout_c (float *Y, const float *X, const int N, const int T, const int M, const int dim, const char iscolmajor);
int maxout_z (double *Y, const double *X, const int N, const int T, const int M, const int dim, const char iscolmajor);


int maxout_s (float *Y, const float *X, const int N, const int T, const int M, const int dim, const char iscolmajor)
{
    const int MN = M*N, TN = T*N;
    int m, n, t, tN, tMN, nT;
    float mx;

    //Checks
    if (M<1) { fprintf(stderr,"error in maxout_s: M (num inputs per neuron) must be positive\n"); return 1; }
    if (N<1) { fprintf(stderr,"error in maxout_s: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in maxout_s: T (num time points) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (t=0; t<T; t++)
            {
                tN = t*N; tMN = t*MN;
                for (n=0; n<N; n++)
                {
                    mx = X[tMN+n];
                    for (m=0; m<M; m++) { if (X[tMN+m*N+n]>mx) { mx = X[tMN+m*N+n]; } }
                    Y[tN+n] = mx;
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                for (t=0; t<T; t++)
                {
                    mx = X[nT+t];
                    for (m=0; m<M; m++) { if (X[m*TN+nT+t]>mx) { mx = X[m*TN+nT+t]; } }
                    Y[nT+t] = mx;
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
                nT = n*T;
                for (t=0; t<T; t++)
                {
                    mx = X[nT+t];
                    for (m=0; m<M; m++) { if (X[m*TN+nT+t]>mx) { mx = X[m*TN+nT+t]; } }
                    Y[nT+t] = mx;
                }
            }
        }
        else
        {
            for (t=0; t<T; t++)
            {
                tN = t*N; tMN = t*MN;
                for (n=0; n<N; n++)
                {
                    mx = X[tMN+n];
                    for (m=0; m<M; m++) { if (X[tMN+m*N+n]>mx) { mx = X[tMN+m*N+n]; } }
                    Y[tN+n] = mx;
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in maxout_s: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int maxout_d (double *Y, const double *X, const int N, const int T, const int M, const int dim, const char iscolmajor)
{
    const int MN = M*N, TN = T*N;
    int m, n, t, tN, tMN, nT;
    double mx;

    //Checks
    if (M<1) { fprintf(stderr,"error in maxout_d: M (num inputs per neuron) must be positive\n"); return 1; }
    if (N<1) { fprintf(stderr,"error in maxout_d: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in maxout_d: T (num time points) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (t=0; t<T; t++)
            {
                tN = t*N; tMN = t*MN;
                for (n=0; n<N; n++)
                {
                    mx = X[tMN+n];
                    for (m=0; m<M; m++) { if (X[tMN+m*N+n]>mx) { mx = X[tMN+m*N+n]; } }
                    Y[tN+n] = mx;
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                for (t=0; t<T; t++)
                {
                    mx = X[nT+t];
                    for (m=0; m<M; m++) { if (X[m*TN+nT+t]>mx) { mx = X[m*TN+nT+t]; } }
                    Y[nT+t] = mx;
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
                nT = n*T;
                for (t=0; t<T; t++)
                {
                    mx = X[nT+t];
                    for (m=0; m<M; m++) { if (X[m*TN+nT+t]>mx) { mx = X[m*TN+nT+t]; } }
                    Y[nT+t] = mx;
                }
            }
        }
        else
        {
            for (t=0; t<T; t++)
            {
                tN = t*N; tMN = t*MN;
                for (n=0; n<N; n++)
                {
                    mx = X[tMN+n];
                    for (m=0; m<M; m++) { if (X[tMN+m*N+n]>mx) { mx = X[tMN+m*N+n]; } }
                    Y[tN+n] = mx;
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in maxout_d: dim must be 0 or 1.\n"); return 1;
    }
    
    return 0;
}


int maxout_c (float *Y, const float *X, const int N, const int T, const int M, const int dim, const char iscolmajor)
{
    const int MN = M*N, TN = T*N;
    int m, n, t, tN, tMN, nT;

    //Checks
    if (M<1) { fprintf(stderr,"error in maxout_c: M (num inputs per neuron) must be positive\n"); return 1; }
    if (N<1) { fprintf(stderr,"error in maxout_c: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in maxout_c: T (num time points) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (t=0; t<T; t++)
            {
                tN = t*N; tMN = t*MN;
                for (n=0; n<N; n++)
                {
                    m = (int)cblas_icamax(M,&X[2*(tMN+n)],N);
                    Y[2*(tN+n)] = X[2*(tMN+m*N+n)]; Y[2*(tN+n)+1] = X[2*(tMN+m*N+n)+1];
                    //Y[tN+n] = sqrtf(X[2*(tMN+m*N+n)]*X[2*(tMN+m*N+n)]+X[2*(tMN+m*N+n)+1]*X[2*(tMN+m*N+n)+1]);
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                for (t=0; t<T; t++)
                {
                    m = (int)cblas_icamax(M,&X[2*(nT+t)],TN);
                    Y[2*(nT+t)] = X[2*(m*TN+nT+t)]; Y[2*(nT+t)+1] = X[2*(m*TN+nT+t)+1];
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
                nT = n*T;
                for (t=0; t<T; t++)
                {
                    m = (int)cblas_icamax(M,&X[2*(nT+t)],TN);
                    Y[2*(nT+t)] = X[2*(m*TN+nT+t)]; Y[2*(nT+t)+1] = X[2*(m*TN+nT+t)+1];
                }
            }
        }
        else
        {
            for (t=0; t<T; t++)
            {
                tN = t*N; tMN = t*MN;
                for (n=0; n<N; n++)
                {
                    m = (int)cblas_icamax(M,&X[2*(tMN+n)],N);
                    Y[2*(tN+n)] = X[2*(tMN+m*N+n)]; Y[2*(tN+n)+1] = X[2*(tMN+m*N+n)+1];
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in maxout_c: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


int maxout_z (double *Y, const double *X, const int N, const int T, const int M, const int dim, const char iscolmajor)
{
    const int MN = M*N, TN = T*N;
    int m, n, t, tN, tMN, nT;

    //Checks
    if (M<1) { fprintf(stderr,"error in maxout_z: M (num inputs per neuron) must be positive\n"); return 1; }
    if (N<1) { fprintf(stderr,"error in maxout_z: N (num neurons) must be positive\n"); return 1; }
    if (T<1) { fprintf(stderr,"error in maxout_z: T (num time points) must be positive\n"); return 1; }

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (t=0; t<T; t++)
            {
                tN = t*N; tMN = t*MN;
                for (n=0; n<N; n++)
                {
                    m = (int)cblas_izamax(M,&X[2*(tMN+n)],N);
                    Y[2*(tN+n)] = X[2*(tMN+m*N+n)]; Y[2*(tN+n)+1] = X[2*(tMN+m*N+n)+1];
                }
            }
        }
        else
        {
            for (n=0; n<N; n++)
            {
                nT = n*T;
                for (t=0; t<T; t++)
                {
                    m = (int)cblas_izamax(M,&X[2*(nT+t)],TN);
                    Y[2*(nT+t)] = X[2*(m*TN+nT+t)]; Y[2*(nT+t)+1] = X[2*(m*TN+nT+t)+1];
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
                nT = n*T;
                for (t=0; t<T; t++)
                {
                    m = (int)cblas_izamax(M,&X[2*(nT+t)],TN);
                    Y[2*(nT+t)] = X[2*(m*TN+nT+t)]; Y[2*(nT+t)+1] = X[2*(m*TN+nT+t)+1];
                }
            }
        }
        else
        {
            for (t=0; t<T; t++)
            {
                tN = t*N; tMN = t*MN;
                for (n=0; n<N; n++)
                {
                    m = (int)cblas_izamax(M,&X[2*(tMN+n)],N);
                    Y[2*(tN+n)] = X[2*(tMN+m*N+n)]; Y[2*(tN+n)+1] = X[2*(tMN+m*N+n)+1];
                }
            }
        }
    }
    else
    {
        fprintf(stderr,"error in maxout_z: dim must be 0 or 1.\n"); return 1;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
