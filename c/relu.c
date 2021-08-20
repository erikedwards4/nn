//This gets ReLU activation function for each element of X.

#include <stdio.h>
#include <time.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int relu_s (float *Y, const float *X, const size_t N);
int relu_d (double *Y, const double *X, const size_t N);

int relu_inplace_s (float *X, const size_t N);
int relu_inplace_d (double *X, const size_t N);


int relu_s (float *Y, const float *X, const size_t N)
{
    for (size_t n=0; n<N; ++n, ++X, ++Y) { *Y = (*X<0.0f) ? 0.0f : *X; }
    //for (size_t n=0; n<N; ++n, ++X, ++Y) { *Y = (float)(*X>0.0f) * *X; }

    return 0;
}


int relu_d (double *Y, const double *X, const size_t N)
{
    for (size_t n=0; n<N; ++n, ++X, ++Y) { *Y = (*X<0.0) ? 0.0 : *X; }
    
    return 0;
}


int relu_inplace_s (float *X, const size_t N)
{
    //struct timespec tic, toc; clock_gettime(CLOCK_REALTIME,&tic);
    for (size_t n=0; n<N; ++n, ++X) { if (*X<0.0f) { *X = 0.0f; } }
    //for (size_t n=0; n<N; ++n, ++X) { *X *= (float)(*X>0.0f); }
    //clock_gettime(CLOCK_REALTIME,&toc); fprintf(stderr,"elapsed time = %.6f ms\n",(toc.tv_sec-tic.tv_sec)*1e3+(toc.tv_nsec-tic.tv_nsec)/1e6);

    return 0;
}


int relu_inplace_d (double *X, const size_t N)
{
    for (size_t n=0; n<N; ++n, ++X) { if (*X<0.0) { *X = 0.0; } }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
