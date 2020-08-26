//Compare function for ascending sort.

#include <stdio.h>
#include <string.h>
#include <math.h>
//#include <complex.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int cmp_ascend_s (const void *a, const void *b);
int cmp_ascend_d (const void *a, const void *b);
int cmp_ascend_c (const void *a, const void *b);
int cmp_ascend_z (const void *a, const void *b);


int cmp_ascend_s (const void *a, const void *b)
{
	float x1 = *(const float*)a, x2 = *(const float*)b;
	if (x1!=x1) { return 1; }
    else if (x2!=x2) { return -1; }
    else if (x1>x2) { return 1; }
    else if (x2>x1) { return -1; }
    else { return 0; }
}


int cmp_ascend_d (const void *a, const void *b)
{
	double x1 = *(const double*)a, x2 = *(const double*)b;
	if (x1!=x1) { return 1; }
    else if (x2!=x2) { return -1; }
    else if (x1>x2) { return 1; }
    else if (x2>x1) { return -1; }
    else { return 0; }
}


int cmp_ascend_c (const void *a, const void *b)
{
    //float complex x1 = *(const float complex*)a, x2 = *(const float complex*)b;
    float x1[2], x2[2], abs1, abs2;
    memcpy(x1,a,2*sizeof(float));
    memcpy(x2,b,2*sizeof(float));
    abs1 = sqrtf(x1[0]*x1[0] + x1[1]*x1[1]);
    abs2 = sqrtf(x2[0]*x2[0] + x2[1]*x2[1]);
	if (abs1!=abs1) { return 1; }
    else if (abs2!=abs2) { return -1; }
    else if (abs1>abs2) { return 1; }
    else if (abs2>abs1) { return -1; }
	else if (atan2f(x1[1],x1[0])>atan2f(x2[1],x2[0])) { return 1; }
    else if (atan2f(x2[1],x2[0])>atan2f(x1[1],x1[0])) { return -1; }
    else { return 0; }
}


int cmp_ascend_z (const void *a, const void *b)
{
    //double complex x1 = *(const double complex*)a, x2 = *(const double complex*)b;
    double x1[2], x2[2], abs1, abs2;
    memcpy(x1,a,2*sizeof(double));
    memcpy(x2,b,2*sizeof(double));
    abs1 = sqrt(x1[0]*x1[0] + x1[1]*x1[1]);
    abs2 = sqrt(x2[0]*x2[0] + x2[1]*x2[1]);
	if (abs1!=abs1) { return 1; }
    else if (abs2!=abs2) { return -1; }
    else if (abs1>abs2) { return 1; }
    else if (abs2>abs1) { return -1; }
	else if (atan2(x1[1],x1[0])>atan2(x2[1],x2[0])) { return 1; }
    else if (atan2(x2[1],x2[0])>atan2(x1[1],x1[0])) { return -1; }
    else { return 0; }
}


#ifdef __cplusplus
}
}
#endif
