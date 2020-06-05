//This gets hard limiter with deadzone for each element of X.
//The width of the deadzone is from -delta to delta.

#include <stdio.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int deadzone_s (float *Y, const float *X, const int N, const float delta);
int deadzone_d (double *Y, const double *X, const int N, const double delta);

int deadzone_inplace_s (float *X, const int N, const float delta);
int deadzone_inplace_d (double *X, const int N, const double delta);


int deadzone_s (float *Y, const float *X, const int N, const float delta)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in deadzone_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = (X[n]>delta) - (X[n]<-delta); }

    return 0;
}


int deadzone_d (double *Y, const double *X, const int N, const double delta)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in deadzone_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { Y[n] = (X[n]>delta) - (X[n]<-delta); }
    
    return 0;
}


int deadzone_inplace_s (float *X, const int N, const float delta)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in deadzone_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] = (X[n]>delta) - (X[n]<-delta); }

    return 0;
}


int deadzone_inplace_d (double *X, const int N, const double delta)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in deadzone_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }

    for (n=0; n<N; n++) { X[n] = (X[n]>delta) - (X[n]<-delta); }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
