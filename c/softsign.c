//Gets the softsign function (aka ElliotSig) of input X element-wise.
//This has in-place and not-in-place versions.

#include <stdio.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int softsign_s (float *Y, const float *X, const int N);
int softsign_d (double *Y, const double *X, const int N);

int softsign_inplace_s (float *X, const int N);
int softsign_inplace_d (double *X, const int N);


int softsign_s (float *Y, const float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in softsign_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = (X[n]>0.0f) ? X[n]/(1.0f+X[n]) : X[n]/(1.0f-X[n]); }

    return 0;
}


int softsign_d (double *Y, const double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in softsign_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = (X[n]>0.0) ? X[n]/(1.0+X[n]) : X[n]/(1.0-X[n]); }
    
    return 0;
}


int softsign_inplace_s (float *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in softsign_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] = (X[n]>0.0f) ? X[n]/(1.0f+X[n]) : X[n]/(1.0f-X[n]); }

    return 0;
}


int softsign_inplace_d (double *X, const int N)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in softsign_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] = (X[n]>0.0) ? X[n]/(1.0+X[n]) : X[n]/(1.0-X[n]); }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
