//Gets the softplus function (derivative of logistic) of input X element-wise.
//This has in-place and not-in-place versions.

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int softplus_s (float *Y, const float *X, const size_t N);
int softplus_d (double *Y, const double *X, const size_t N);

int softplus_inplace_s (float *X, const size_t N);
int softplus_inplace_d (double *X, const size_t N);


int softplus_s (float *Y, const float *X, const size_t N)
{
    for (size_t n=0; n<N; ++n, ++X, ++Y) { *Y = logf(1.0f+expf(*X)); }

    return 0;
}


int softplus_d (double *Y, const double *X, const size_t N)
{
    for (size_t n=0; n<N; ++n, ++X, ++Y) { *Y = log(1.0+exp(*X)); }
    
    return 0;
}


int softplus_inplace_s (float *X, const size_t N)
{
    for (size_t n=0; n<N; ++n, ++X) { *X = logf(1.0f+expf(*X)); }

    return 0;
}


int softplus_inplace_d (double *X, const size_t N)
{
    for (size_t n=0; n<N; ++n, ++X) { *X = log(1.0+exp(*X)); }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
