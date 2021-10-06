//This gets parametric ReLU activation function for each element of X.
//This is also called the PReLU (where default alpha is 0.25).
//For compatibility to PyTorch, this function is identical to PReLU.

#include <stdio.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int leaky_relu_s (float *Y, const float *X, const size_t N, const float alpha);
int leaky_relu_d (double *Y, const double *X, const size_t N, const double alpha);

int leaky_relu_inplace_s (float *X, const size_t N, const float alpha);
int leaky_relu_inplace_d (double *X, const size_t N, const double alpha);


int leaky_relu_s (float *Y, const float *X, const size_t N, const float alpha)
{
    if (alpha==0.0f)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0f) ? 0.0f : *X; }
    }
    else if (alpha==1.0f)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X; }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0f) ? alpha**X : *X; }
    }

    return 0;
}


int leaky_relu_d (double *Y, const double *X, const size_t N, const double alpha)
{
    if (alpha==0.0)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0) ? 0.0 : *X; }
    }
    else if (alpha==1.0)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X; }
    }
    else
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = (*X<0.0) ? alpha**X : *X; }
    }
    
    return 0;
}


int leaky_relu_inplace_s (float *X, const size_t N, const float alpha)
{
    if (alpha==0.0f)
    {
        for (size_t n=N; n>0u; --n, ++X) { if (*X<0.0f) { *X = 0.0f; } }
    }
    else if (alpha==1.0f) {}
    else
    {
        for (size_t n=N; n>0u; --n, ++X) { if(*X<0.0f) { *X *= alpha; } }
    }

    return 0;
}


int leaky_relu_inplace_d (double *X, const size_t N, const double alpha)
{
    if (alpha==0.0)
    {
        for (size_t n=N; n>0u; --n, ++X) { if (*X<0.0) { *X = 0.0; } }
    }
    else if (alpha==1.0) {}
    else
    {
        for (size_t n=N; n>0u; --n, ++X) { if(*X<0.0) { *X *= alpha; } }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
