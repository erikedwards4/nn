//This does 1D max pooling, like PyTorch MaxPool1d,
//but uses the full set of input params/opts available for conv1d.

//Considered as a component of the present framework,
//this is an IN (input) method for N==N neurons.
//In PyTorch, N and N are called C, i.e. num chans.

//The parameters are kernel_size == Lk (i.e., kernel width);
//and stride, dilation and padding for the maxpooling itself.
//Note that no kernel is input; it is implicitly just ones.

//If using dil==1, use maxpool1.

//X is the input of size N x Li,
//where Li==L_in is usually thought of as the number of time points.
//X can be row- or col-major, but N==C_in must always be the leading dim, i.e.:
//X has size N x Li for col-major.
//X has size Li x N for row-major.

//Each vector in Y has length Lo==L_out, set by:
//Lo =  ceil[1 + (Li + 2*pad - dil*(Lk-1) - 1)/stride], if ceil_mode is true
//Lo = floor[1 + (Li + 2*pad - dil*(Lk-1) - 1)/stride], if ceil_mode is false [default]
//Y has size N x Lo for col-major.
//Y has size Lo x N for row-major.

//The following params/opts are included:
//N:            size_t  num input neurons (leading dim of X)
//Li:           size_t  length of input time-series (other dim of X)
//Lk:           size_t  length of kernel in time (num cols of K)
//pad:          int     padding length (pad>0 means add extra samps)
//str:          size_t  stride length (step-size) in samps
//dil:          size_t  dilation factor in samps
//ceil_mode:    bool    calculate Lo with ceil (see above)

#include <stdio.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int maxpool1d_s (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode);
int maxpool1d_d (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode);
int maxpool1d_c (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode);
int maxpool1d_z (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode);


int maxpool1d_s (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode)
{
    if (str<1u) { fprintf(stderr,"error in maxpool1d_s: str (stride) must be positive\n"); return 1; }
    if (dil<1u) { fprintf(stderr,"error in maxpool1d_s: dil (dilation) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in maxpool1d_s: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in maxpool1d_s: Lk (kernel_size) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in maxpool1d_s: N (num input neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in maxpool1d_s: pad length must be > -Li\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in maxpool1d_s: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in maxpool1d_s: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-Tk)/str + (size_t)(ceil_mode && (size_t)(Ti-Tk)%str);
    
    const size_t jump = N*(dil-1u);         //fixed jump due to dilation for X below
    size_t w=0u;                            //current window (frame)
    int ss=-pad, es=ss+Tk-1;                //current start-samp, end-samp
    int t = 0, prev_t = 0;                  //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        //Init Y
        for (size_t n=N; n>0u; --n, ++Y) { *Y = HUGE_VALF; }
        Y -= N;

        //Non-negative samps
        if (es>=0)
        {
            for (int s=es%(int)dil; s<=es; s+=dil, Y-=N)
            {
                X += (s-prev_t) * (int)N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
                prev_t = s + 1;
            }
        }
        X -= prev_t * (int)N;
        prev_t = 0;

        Y+=N; ss+=str; es+=str; ++w;
    }
    X += ss*(int)N;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X; }
        X += jump; Y -= N;
        for (size_t l=Lk-1u; l>0u; --l, X+=jump, Y-=N)
        {
            for (size_t n=N; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
        }
        X -= (int)N*((int)(Lk*dil)-(int)str); Y += N;
        es+=str; ++w;
    }
    prev_t = ss = es - Tk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Init Y
        for (size_t n=N; n>0u; --n, ++Y) { *Y = HUGE_VALF; }
        Y -= N;

        //Get num valid samps
        int V = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

        //Valid samps
        for (int s=ss; s<(int)Li; s+=dil, Y-=N)
        {
            X += (s-prev_t) * (int)N;
            for (size_t n=N; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
            prev_t = s + 1;
        }

        Y+=N; ss+=str; es+=str; ++w;
    }

    return 0;
}


int maxpool1d_d (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode)
{
    if (str<1u) { fprintf(stderr,"error in maxpool1d_d: str (stride) must be positive\n"); return 1; }
    if (dil<1u) { fprintf(stderr,"error in maxpool1d_d: dil (dilation) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in maxpool1d_d: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in maxpool1d_d: Lk (kernel_size) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in maxpool1d_d: N (num input neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in maxpool1d_d: pad length must be > -Li\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in maxpool1d_d: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in maxpool1d_d: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-Tk)/str + (size_t)(ceil_mode && (size_t)(Ti-Tk)%str);
    
    const size_t jump = N*(dil-1u);         //fixed jump due to dilation for X below
    size_t w=0u;                            //current window (frame)
    int ss=-pad, es=ss+Tk-1;                //current start-samp, end-samp
    int t = 0, prev_t = 0;                  //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        //Init Y
        for (size_t n=N; n>0u; --n, ++Y) { *Y = HUGE_VAL; }
        Y -= N;

        //Non-negative samps
        if (es>=0)
        {
            for (int s=es%(int)dil; s<=es; s+=dil, Y-=N)
            {
                X += (s-prev_t) * (int)N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
                prev_t = s + 1;
            }
        }
        X -= prev_t * (int)N;
        prev_t = 0;

        Y+=N; ss+=str; es+=str; ++w;
    }
    X += ss*(int)N;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X; }
        X += jump; Y -= N;
        for (size_t l=Lk-1u; l>0u; --l, X+=jump, Y-=N)
        {
            for (size_t n=N; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
        }
        X -= (int)N*((int)(Lk*dil)-(int)str); Y += N;
        es+=str; ++w;
    }
    prev_t = ss = es - Tk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Init Y
        for (size_t n=N; n>0u; --n, ++Y) { *Y = HUGE_VAL; }
        Y -= N;

        //Get num valid samps
        int V = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

        //Valid samps
        for (int s=ss; s<(int)Li; s+=dil, Y-=N)
        {
            X += (s-prev_t) * (int)N;
            for (size_t n=N; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
            prev_t = s + 1;
        }

        Y+=N; ss+=str; es+=str; ++w;
    }

    return 0;
}


int maxpool1d_c (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode)
{
    if (str<1u) { fprintf(stderr,"error in maxpool1d_c: str (stride) must be positive\n"); return 1; }
    if (dil<1u) { fprintf(stderr,"error in maxpool1d_c: dil (dilation) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in maxpool1d_c: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in maxpool1d_c: Lk (kernel_size) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in maxpool1d_c: N (num input neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in maxpool1d_c: pad length must be > -Li\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in maxpool1d_c: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in maxpool1d_c: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-Tk)/str + (size_t)(ceil_mode && (size_t)(Ti-Tk)%str);
    
    const size_t N2 = 2u*N;                 //fixed jump due to dilation for X below
    const size_t jump = N2*(dil-1u);        //fixed jump due to dilation for X below
    size_t w=0u;                            //current window (frame)
    int ss=-pad, es=ss+Tk-1;                //current start-samp, end-samp
    int t = 0, prev_t = 0;                  //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        //Init Y
        for (size_t n=N2; n>0u; --n, ++Y) { *Y = HUGE_VALF; }
        Y -= N2;

        //Non-negative samps
        if (es>=0)
        {
            for (int s=es%(int)dil; s<=es; s+=dil, Y-=N2)
            {
                X += (s-prev_t) * (int)N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
                prev_t = s + 1;
            }
        }
        X -= prev_t * (int)N2;
        prev_t = 0;

        Y+=N2; ss+=str; es+=str; ++w;
    }
    X += ss*(int)N2;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y = *X; }
        X += jump; Y -= N2;
        for (size_t l=Lk-1u; l>0u; --l, X+=jump, Y-=N2)
        {
            for (size_t n=N2; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
        }
        X -= (int)N2*((int)(Lk*dil)-(int)str); Y += N2;
        es+=str; ++w;
    }
    prev_t = ss = es - Tk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Init Y
        for (size_t n=N2; n>0u; --n, ++Y) { *Y = HUGE_VALF; }
        Y -= N2;

        //Get num valid samps
        int V = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

        //Valid samps
        for (int s=ss; s<(int)Li; s+=dil, Y-=N2)
        {
            X += (s-prev_t) * (int)N2;
            for (size_t n=N2; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
            prev_t = s + 1;
        }

        Y+=N2; ss+=str; es+=str; ++w;
    }

    return 0;
}


int maxpool1d_z (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode)
{
    if (str<1u) { fprintf(stderr,"error in maxpool1d_z: str (stride) must be positive\n"); return 1; }
    if (dil<1u) { fprintf(stderr,"error in maxpool1d_z: dil (dilation) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in maxpool1d_z: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in maxpool1d_z: Lk (kernel_size) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in maxpool1d_z: N (num input neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in maxpool1d_z: pad length must be > -Li\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in maxpool1d_z: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in maxpool1d_z: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-Tk)/str + (size_t)(ceil_mode && (size_t)(Ti-Tk)%str);
    
    const size_t N2 = 2u*N;                 //fixed jump due to dilation for X below
    const size_t jump = N2*(dil-1u);        //fixed jump due to dilation for X below
    size_t w=0u;                            //current window (frame)
    int ss=-pad, es=ss+Tk-1;                //current start-samp, end-samp
    int t = 0, prev_t = 0;                  //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        //Init Y
        for (size_t n=N2; n>0u; --n, ++Y) { *Y = HUGE_VAL; }
        Y -= N2;

        //Non-negative samps
        if (es>=0)
        {
            for (int s=es%(int)dil; s<=es; s+=dil, Y-=N2)
            {
                X += (s-prev_t) * (int)N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
                prev_t = s + 1;
            }
        }
        X -= prev_t * (int)N2;
        prev_t = 0;

        Y+=N2; ss+=str; es+=str; ++w;
    }
    X += ss*(int)N2;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y = *X; }
        X += jump; Y -= N2;
        for (size_t l=Lk-1u; l>0u; --l, X+=jump, Y-=N2)
        {
            for (size_t n=N2; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
        }
        X -= (int)N2*((int)(Lk*dil)-(int)str); Y += N2;
        es+=str; ++w;
    }
    prev_t = ss = es - Tk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Init Y
        for (size_t n=N2; n>0u; --n, ++Y) { *Y = HUGE_VAL; }
        Y -= N2;

        //Get num valid samps
        int V = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

        //Valid samps
        for (int s=ss; s<(int)Li; s+=dil, Y-=N2)
        {
            X += (s-prev_t) * (int)N2;
            for (size_t n=N2; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
            prev_t = s + 1;
        }

        Y+=N2; ss+=str; es+=str; ++w;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
