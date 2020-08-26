//Gets the softsign function (aka ElliotSig) of input X element-wise.
//This has in-place and not-in-place versions.

#include <stdio.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int softsign_s (float *Y, const float *X, const size_t N);
int softsign_d (double *Y, const double *X, const size_t N);

int softsign_inplace_s (float *X, const size_t N);
int softsign_inplace_d (double *X, const size_t N);


int softsign_s (float *Y, const float *X, const size_t N)
{
    for (size_t n=0; n<N; ++n, ++X, ++Y) { *Y = (*X>0.0f) ? *X/(1.0f+*X) : *X/(1.0f-*X); }

    return 0;
}


int softsign_d (double *Y, const double *X, const size_t N)
{
    for (size_t n=0; n<N; ++n, ++X, ++Y) { *Y = (*X>0.0) ? *X/(1.0+*X) : *X/(1.0-*X); }
    
    return 0;
}


int softsign_inplace_s (float *X, const size_t N)
{
    for (size_t n=0; n<N; ++n, ++X) { *X = (*X>0.0f) ? *X/(1.0f+*X) : *X/(1.0f-*X); }

    return 0;
}


int softsign_inplace_d (double *X, const size_t N)
{
    for (size_t n=0; n<N; ++n, ++X) { *X = (*X>0.0) ? *X/(1.0+*X) : *X/(1.0-*X); }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
