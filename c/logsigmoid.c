//This gets logsigmoid activation function for each element of X.

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int logsigmoid_s (float *Y, const float *X, const size_t N);
int logsigmoid_d (double *Y, const double *X, const size_t N);

int logsigmoid_inplace_s (float *X, const size_t N);
int logsigmoid_inplace_d (double *X, const size_t N);


int logsigmoid_s (float *Y, const float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        //*Y = logf(1.0f/(1.0f+expf(-*X)));
        *Y = -logf(1.0f+expf(-*X));
    }

    return 0;
}


int logsigmoid_d (double *Y, const double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X, ++Y)
    {
        *Y = -log(1.0+exp(-*X));
    }

    return 0;
}


int logsigmoid_inplace_s (float *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X)
    {
        *X = -logf(1.0f+expf(-*X));
    }

    return 0;
}


int logsigmoid_inplace_d (double *X, const size_t N)
{
    for (size_t n=N; n>0u; --n, ++X)
    {
        *X = -log(1.0+exp(-*X));
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
