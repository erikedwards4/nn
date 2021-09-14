//Gets the SQNL function [Wuraola & Patel 2018] of input X element-wise.
//This has in-place and not-in-place versions.

#include <stdio.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int sqnl_s (float *Y, const float *X, const size_t N);
int sqnl_d (double *Y, const double *X, const size_t N);

int sqnl_inplace_s (float *X, const size_t N);
int sqnl_inplace_d (double *X, const size_t N);


int sqnl_s (float *Y, const float *X, const size_t N)
{


    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X>0.0f)
        {
            if (*X<2.0f) { *Y = *X - 0.25f**X**X; }
            else { *Y = 1.0f; }
        }
        else
        {
            if (*X>-2.0f) { *Y = *X + 0.25f**X**X; }
            else { *Y = -1.0f; }
        }
    }

    return 0;
}


int sqnl_d (double *Y, const double *X, const size_t N)
{


    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X>0.0)
        {
            if (*X<2.0) { *Y = *X - 0.25**X**X; }
            else { *Y = 1.0; }
        }
        else
        {
            if (*X>-2.0) { *Y = *X + 0.25**X**X; }
            else { *Y = -1.0; }
        }
    }
    
    return 0;
}


int sqnl_inplace_s (float *X, const size_t N)
{


    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X>0.0f)
        {
            if (*X<2.0f) { *X -= 0.25f**X**X; }
            else { *X = 1.0f; }
        }
        else
        {
            if (*X>-2.0f) { *X += 0.25f**X**X; }
            else { *X = -1.0f; }
        }
    }

    return 0;
}


int sqnl_inplace_d (double *X, const size_t N)
{


    for (size_t n=N; n>0u; --n, ++X)
    {
        if (*X>0.0)
        {
            if (*X<2.0) { *X -= 0.25**X**X; }
            else { *X = 1.0; }
        }
        else
        {
            if (*X>-2.0) { *X += 0.25**X**X; }
            else { *X = -1.0; }
        }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
