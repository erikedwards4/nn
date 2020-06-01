//This gets step activation function for each element of X.
//I also include a thresh (which is 0 by usual definition) for more general use.

//I also include an option m, such that the step goes from -1 to 1 instead of 0 to 1.
//The -1 to 1 step function is used in the Hopfield network.
//For the usual case (m=0), this is the binary step function.

#include <stdio.h>

#ifdef __cplusplus
namespace openn {
extern "C" {
#endif

int step_s (float *Y, const float *X, const int N, const float thresh, const char m);
int step_d (double *Y, const double *X, const int N, const double thresh, const char m);

int step_inplace_s (float *X, const int N, const float thresh, const char m);
int step_inplace_d (double *X, const int N, const double thresh, const char m);


int step_s (float *Y, const float *X, const int N, const float thresh, const char m)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in step_s: N (num elements X) must be nonnegative\n"); return 1; }

    if (m)
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<thresh) ? -1.0f : 1.0f; }
    }
    else
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<thresh) ? 0.0f : 1.0f; }
    }

    return 0;
}


int step_d (double *Y, const double *X, const int N, const double thresh, const char m)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in step_d: N (num elements X) must be nonnegative\n"); return 1; }

    if (m)
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<thresh) ? -1.0 : 1.0; }
    }
    else
    {
        for (n=0; n<N; n++) { Y[n] = (X[n]<thresh) ? 0.0 : 1.0; }
    }
    
    return 0;
}


int step_inplace_s (float *X, const int N, const float thresh, const char m)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in step_inplace_s: N (num elements X) must be nonnegative\n"); return 1; }

    if (m)
    {
        for (n=0; n<N; n++) { X[n] = (X[n]<thresh) ? -1.0f : 1.0f; }
    }
    else
    {
        for (n=0; n<N; n++) { X[n] = (X[n]<thresh) ? 0.0f : 1.0f; }
    }

    return 0;
}


int step_inplace_d (double *X, const int N, const double thresh, const char m)
{
    int n;

    //Checks
    if (N<0) { fprintf(stderr,"error in step_inplace_d: N (num elements X) must be nonnegative\n"); return 1; }

    if (m)
    {
        for (n=0; n<N; n++) { X[n] = (X[n]<thresh) ? -1.0 : 1.0; }
    }
    else
    {
        for (n=0; n<N; n++) { X[n] = (X[n]<thresh) ? 0.0 : 1.0; }
    }
    
    return 0;
}


#ifdef __cplusplus
}
}
#endif
