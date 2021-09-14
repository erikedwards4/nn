//Gets the "new" Gaussian Error Linear Unit (GELU) of input X element-wise.
//This has in-place and not-in-place versions.

//The gelu_new_new is described here:
//https://github.com/huggingface/transformers/blob/master/src/transformers/activations.py

//powf(*X,3.0f) tested at same speed (perhaps a tiny bit slower?) as *X**X**X
//powf(*X,3.0f) has more assembly instructions at -O1 and -O2, but fewer at -O3.

#include <stdio.h>
#include <math.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int gelu_new_s (float *Y, const float *X, const size_t N);
int gelu_new_d (double *Y, const double *X, const size_t N);

int gelu_new_inplace_s (float *X, const size_t N);
int gelu_new_inplace_d (double *X, const size_t N);


int gelu_new_s (float *Y, const float *X, const size_t N)
{
    const float sc = (float)sqrt(2.0/M_PI);

    for (size_t n=0; n<N; ++n, ++X, ++Y)
    {
        *Y = 0.5f * *X * (1.0f + tanhf(sc * (*X + 0.044715f*powf(*X,3.0f))));
    }

    return 0;
}


int gelu_new_d (double *Y, const double *X, const size_t N)
{
    const double sc = sqrt(2.0/M_PI);

    for (size_t n=0; n<N; ++n, ++X, ++Y)
    {
        *Y = 0.5 * *X * (1.0 + tanh(sc * (*X + 0.044715*pow(*X,3.0))));
    }
    
    return 0;
}


int gelu_new_inplace_s (float *X, const size_t N)
{
    const float sc = (float)sqrt(2.0/M_PI);

    for (size_t n=0; n<N; ++n, ++X)
    {
        *X = 0.5f * *X * (1.0f + tanhf(sc * (*X + 0.044715f*powf(*X,3.0f))));
    }

    return 0;
}


int gelu_new_inplace_d (double *X, const size_t N)
{
    const double sc = sqrt(2.0/M_PI);

    for (size_t n=0; n<N; ++n, ++X)
    {
        *X = 0.5 * *X * (1.0 + tanh(sc * (*X + 0.044715*pow(*X,3.0))));
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
