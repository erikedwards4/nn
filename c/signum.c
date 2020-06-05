//This gets signum (sign) activation function for each element of X.
//I also include a thresh (which is 0 by usual definition) for more general use.

//This is a.k.a. the hard limiter.
//This is used in the Hopfield network, for example.

#include <stdio.h>
//#include <time.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int signum_s (float *Y, const float *X, const int N, const float thresh);
int signum_d (double *Y, const double *X, const int N, const double thresh);

int signum_inplace_s (float *X, const int N, const float thresh);
int signum_inplace_d (double *X, const int N, const double thresh);


int signum_s (float *Y, const float *X, const int N, const float thresh)
{
    int n;
    //struct timespec tic, toc;

    //Checks
    if (N<0) { fprintf(stderr,"error in signum_s: N (num elements X) must be nonnegative\n"); return 1; }

    //clock_gettime(CLOCK_REALTIME,&tic);
    for (n=0; n<N; n++) { Y[n] = (X[n]>thresh) - (X[n]<thresh); }
    // for (n=0; n<N; n++)
    // {
    //     if (X[n]<thresh) { Y[n] = -1.0f; }
    //     else if (X[n]>thresh) { Y[n] = 1.0f; }
    //     else { Y[n] = 0.0f; }
    // }
    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);

    return 0;
}


int signum_d (double *Y, const double *X, const int N, const double thresh)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in signum_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = (X[n]>thresh) - (X[n]<thresh); }
    
    return 0;
}


int signum_inplace_s (float *X, const int N, const float thresh)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in signum_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++)
    {
        if (X[n]<thresh) { X[n] = -1.0f; }
        else if (X[n]>thresh) { X[n] = 1.0f; }
    }

    return 0;
}


int signum_inplace_d (double *X, const int N, const double thresh)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in signum_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++)
    {
        if (X[n]<thresh) { X[n] = -1.0; }
        else if (X[n]>thresh) { X[n] = 1.0; }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
