//This gets maxout layer-wise activation function for X.
//This is just the max along rows or cols of X according to dim.
//M is the number of inputs per neuron

//For complex input X, output Y is complex with elements having max absolute values.


#include <stdio.h>
#include <cblas.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int maxout_s (float *Y, const float *X, const size_t N, const size_t T, const size_t M, const char iscolmajor, const size_t dim);
int maxout_d (double *Y, const double *X, const size_t N, const size_t T, const size_t M, const char iscolmajor, const size_t dim);
int maxout_c (float *Y, const float *X, const size_t N, const size_t T, const size_t M, const char iscolmajor, const size_t dim);
int maxout_z (double *Y, const double *X, const size_t N, const size_t T, const size_t M, const char iscolmajor, const size_t dim);


int maxout_s (float *Y, const float *X, const size_t N, const size_t T, const size_t M, const char iscolmajor, const size_t dim)
{
    const size_t MN = M*N, TN = T*N;
    size_t tN, tMN, nT;
    float mx;

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t t=0; t<T; ++t)
            {
                tMN = t*MN;
                for (size_t n=0; n<N; ++n, ++Y)
                {
                    mx = X[tMN+n];
                    for (size_t m=0; m<M; ++m) { if (X[tMN+m*N+n]>mx) { mx = X[tMN+m*N+n]; } }
                    *Y = mx;
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                for (size_t t=0; t<T; ++t)
                {
                    mx = X[nT+t];
                    for (size_t m=0; m<M; ++m) { if (X[m*TN+nT+t]>mx) { mx = X[m*TN+nT+t]; } }
                    Y[nT+t] = mx;
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
                for (size_t t=0; t<T; ++t)
                {
                    mx = X[nT+t];
                    for (size_t m=0; m<M; ++m) { if (X[m*TN+nT+t]>mx) { mx = X[m*TN+nT+t]; } }
                    Y[nT+t] = mx;
                }
            }
        }
        else
        {
            for (size_t t=0; t<T; ++t)
            {
                tMN = t*MN;
                for (size_t n=0; n<N; ++n, ++Y)
                {
                    mx = X[tMN+n];
                    for (size_t m=0; m<M; ++m) { if (X[tMN+m*N+n]>mx) { mx = X[tMN+m*N+n]; } }
                    *Y = mx;
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


int maxout_d (double *Y, const double *X, const size_t N, const size_t T, const size_t M, const char iscolmajor, const size_t dim)
{
    const size_t MN = M*N, TN = T*N;
    size_t tN, tMN, nT;
    double mx;

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t t=0; t<T; ++t)
            {
                tN = t*N; tMN = t*MN;
                for (size_t n=0; n<N; ++n)
                {
                    mx = X[tMN+n];
                    for (size_t m=0; m<M; ++m) { if (X[tMN+m*N+n]>mx) { mx = X[tMN+m*N+n]; } }
                    Y[tN+n] = mx;
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                for (size_t t=0; t<T; ++t)
                {
                    mx = X[nT+t];
                    for (size_t m=0; m<M; ++m) { if (X[m*TN+nT+t]>mx) { mx = X[m*TN+nT+t]; } }
                    Y[nT+t] = mx;
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
                for (size_t t=0; t<T; ++t)
                {
                    mx = X[nT+t];
                    for (size_t m=0; m<M; ++m) { if (X[m*TN+nT+t]>mx) { mx = X[m*TN+nT+t]; } }
                    Y[nT+t] = mx;
                }
            }
        }
        else
        {
            for (size_t t=0; t<T; ++t)
            {
                tN = t*N; tMN = t*MN;
                for (size_t n=0; n<N; ++n)
                {
                    mx = X[tMN+n];
                    for (size_t m=0; m<M; ++m) { if (X[tMN+m*N+n]>mx) { mx = X[tMN+m*N+n]; } }
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


int maxout_c (float *Y, const float *X, const size_t N, const size_t T, const size_t M, const char iscolmajor, const size_t dim)
{
    const size_t MN = M*N, TN = T*N;
    size_t tN, tMN, nT;

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t t=0; t<T; ++t)
            {
                tN = t*N; tMN = t*MN;
                for (size_t n=0; n<N; ++n)
                {
                    m = (int)cblas_icamax(M,&X[2*(tMN+n)],(int)N);
                    Y[2*(tN+n)] = X[2*(tMN+m*N+n)]; Y[2*(tN+n)+1] = X[2*(tMN+m*N+n)+1];
                    //Y[tN+n] = sqrtf(X[2*(tMN+m*N+n)]*X[2*(tMN+m*N+n)]+X[2*(tMN+m*N+n)+1]*X[2*(tMN+m*N+n)+1]);
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                for (size_t t=0; t<T; ++t)
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
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                for (size_t t=0; t<T; ++t)
                {
                    m = (int)cblas_icamax(M,&X[2*(nT+t)],TN);
                    Y[2*(nT+t)] = X[2*(m*TN+nT+t)]; Y[2*(nT+t)+1] = X[2*(m*TN+nT+t)+1];
                }
            }
        }
        else
        {
            for (size_t t=0; t<T; ++t)
            {
                tN = t*N; tMN = t*MN;
                for (size_t n=0; n<N; ++n)
                {
                    m = (int)cblas_icamax(M,&X[2*(tMN+n)],(int)N);
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


int maxout_z (double *Y, const double *X, const size_t N, const size_t T, const size_t M, const char iscolmajor, const size_t dim)
{
    const size_t MN = M*N, TN = T*N;
    size_t tN, tMN, nT;

    if (dim==0)
    {
        if (iscolmajor)
        {
            for (size_t t=0; t<T; ++t)
            {
                tN = t*N; tMN = t*MN;
                for (size_t n=0; n<N; ++n)
                {
                    m = (int)cblas_izamax(M,&X[2*(tMN+n)],(int)N);
                    Y[2*(tN+n)] = X[2*(tMN+m*N+n)]; Y[2*(tN+n)+1] = X[2*(tMN+m*N+n)+1];
                }
            }
        }
        else
        {
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                for (size_t t=0; t<T; ++t)
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
            for (size_t n=0; n<N; ++n)
            {
                nT = n*T;
                for (size_t t=0; t<T; ++t)
                {
                    m = (int)cblas_izamax(M,&X[2*(nT+t)],TN);
                    Y[2*(nT+t)] = X[2*(m*TN+nT+t)]; Y[2*(nT+t)+1] = X[2*(m*TN+nT+t)+1];
                }
            }
        }
        else
        {
            for (size_t t=0; t<T; ++t)
            {
                tN = t*N; tMN = t*MN;
                for (size_t n=0; n<N; ++n)
                {
                    m = (int)cblas_izamax(M,&X[2*(tMN+n)],(int)N);
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
