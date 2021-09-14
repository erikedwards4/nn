//Gets output side of perceptron layer.
//I also include a thresh (which is 0 by usual definition) for more general use.

#include <stdio.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int perceptron_s (float *Y, const float *X, const size_t N, const float thresh);
int perceptron_d (double *Y, const double *X, const size_t N, const double thresh);

int perceptron_inplace_s (float *X, const size_t N, const float thresh);
int perceptron_inplace_d (double *X, const size_t N, const double thresh);


int perceptron_s (float *Y, const float *X, const size_t N, const float thresh)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in perceptron_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (size_t n=0u; n<N; ++n) { Y[n] = (X[n]<thresh) ? 0.0f : 1.0f; }

    return 0;
}


int perceptron_d (double *Y, const double *X, const size_t N, const double thresh)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in perceptron_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (size_t n=0u; n<N; ++n) { Y[n] = (X[n]<thresh) ? 0.0 : 1.0; }
    
    return 0;
}


int perceptron_inplace_s (float *X, const size_t N, const float thresh)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in perceptron_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (size_t n=0u; n<N; ++n) { X[n] = (X[n]<thresh) ? 0.0f : 1.0f; }

    return 0;
}


int perceptron_inplace_d (double *X, const size_t N, const double thresh)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in perceptron_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (size_t n=0u; n<N; ++n) { X[n] = (X[n]<thresh) ? 0.0 : 1.0; }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
