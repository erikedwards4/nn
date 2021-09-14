//This is just like maxpool1d, but doesn't allow dilation.
//This allows shorter and more efficient code.

//X is the input of size N x Li,
//where Li==L_in is usually thought of as the number of time points.
//X can be row- or col-major, but N==C_in must always be the leading dim, i.e.:
//X has size N x Li for col-major.
//X has size Li x N for row-major.

//Each vector in Y has length Lo==L_out, set by:
//Lo =  ceil[1 + (Li + 2*pad - Lk)/stride], if ceil_mode is true
//Lo = floor[1 + (Li + 2*pad - Lk)/stride], if ceil_mode is false [default]
//Y has size N x Lo for col-major.
//Y has size Lo x N for row-major.

//The following params/opts are included:
//N:            size_t  num input neurons (leading dim of X)
//Li:           size_t  length of input time-series (other dim of X)
//Lk:           size_t  length of kernel in time (num cols of K)
//pad:          int     padding length (pad>0 means add extra samps)
//str:          size_t  stride length (step-size) in samps
//ceil_mode:    bool    calculate Lo with ceil (see above)

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int maxpool1_s (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode);
int maxpool1_d (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode);
int maxpool1_c (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode);
int maxpool1_z (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode);


int maxpool1_s (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode)
{
    if (str<1u) { fprintf(stderr,"error in maxpool1_s: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in maxpool1_s: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in maxpool1_s: Lk (kernel_size) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in maxpool1_s: N (num input neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in maxpool1_s: pad length must be > -Li\n"); return 1; }

    const int Ti = (int)Li + 2*pad;             //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in maxpool1_s: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in maxpool1_s: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);
    
    const int inc = (int)N*((int)str-(int)Lk);  //fixed increment for X below
    size_t w=0u;                                //current window (frame)
    int ss=-pad, es=ss+(int)Lk-1;               //current start-samp, end-samp

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        //Init Y
        for (size_t n=N; n>0u; --n, ++Y) { *Y = -HUGE_VALF; }
        Y -= N;

        //Non-negative samps
        if (es>=0)
        {
            for (int s=0; s<=es; ++s, Y-=N)
            {
                for (size_t n=N; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
            }
            X -= (int)N*(es+1);
        }

        Y+=N; ss+=str; es+=str; ++w;
    }
    X += ss*(int)N;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X; }
        Y -= N;
        for (size_t l=Lk-1u; l>0u; --l, Y-=N)
        {
            for (size_t n=N; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
        }
        X += inc; Y += N;
        es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Init Y
        for (size_t n=N; n>0u; --n, ++Y) { *Y = -HUGE_VALF; }
        Y -= N;

        //Get num valid samps
        int V = (ss<(int)Li) ? (int)Li-ss : 0;

        //Valid samps
        if (V>0)
        {
            for (int v=V; v>0; --v, Y-=N)
            {
                for (size_t n=N; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
            }
        }
        else { X += ((int)Li-ss) * (int)N; }

        X += N*(str+(size_t)ss-Li); Y += N;
        ss+=str; es+=str; ++w;
    }

    return 0;
}


int maxpool1_d (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode)
{
    if (str<1u) { fprintf(stderr,"error in maxpool1_d: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in maxpool1_d: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in maxpool1_d: Lk (kernel_size) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in maxpool1_d: N (num input neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in maxpool1_d: pad length must be > -Li\n"); return 1; }

    const int Ti = (int)Li + 2*pad;             //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in maxpool1_d: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in maxpool1_d: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);
    
    const int inc = (int)N*((int)str-(int)Lk);  //fixed increment for X below
    size_t w=0u;                                //current window (frame)
    int ss=-pad, es=ss+(int)Lk-1;               //current start-samp, end-samp

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        //Init Y
        for (size_t n=N; n>0u; --n, ++Y) { *Y = -HUGE_VAL; }
        Y -= N;

        //Non-negative samps
        if (es>=0)
        {
            for (int s=0; s<=es; ++s, Y-=N)
            {
                for (size_t n=N; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
            }
            X -= (int)N*(es+1);
        }

        Y+=N; ss+=str; es+=str; ++w;
    }
    X += ss*(int)N;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X; }
        Y -= N;
        for (size_t l=Lk-1u; l>0u; --l, Y-=N)
        {
            for (size_t n=N; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
        }
        X += inc; Y += N;
        es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Init Y
        for (size_t n=N; n>0u; --n, ++Y) { *Y = -HUGE_VAL; }
        Y -= N;

        //Get num valid samps
        int V = (ss<(int)Li) ? (int)Li-ss : 0;

        //Valid samps
        if (V>0)
        {
            for (int v=V; v>0; --v, Y-=N)
            {
                for (size_t n=N; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
            }
        }
        else { X += ((int)Li-ss) * (int)N; }

        X += N*(str+(size_t)ss-Li); Y += N;
        ss+=str; es+=str; ++w;
    }

    return 0;
}


int maxpool1_c (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode)
{
    if (str<1u) { fprintf(stderr,"error in maxpool1_c: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in maxpool1_c: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in maxpool1_c: Lk (kernel_size) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in maxpool1_c: N (num input neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in maxpool1_c: pad length must be > -Li\n"); return 1; }

    const int Ti = (int)Li + 2*pad;             //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in maxpool1_c: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in maxpool1_c: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);
    
    const size_t N2 = 2u*N;                     //to save multiplies
    const int inc = N2*((int)str-(int)Lk);      //fixed increment for X below
    size_t w=0u;                                //current window (frame)
    int ss=-pad, es=ss+(int)Lk-1;               //current start-samp, end-samp

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        //Init Y
        for (size_t n=N2; n>0u; --n, ++Y) { *Y = -HUGE_VALF; }
        Y -= N2;

        //Non-negative samps
        if (es>=0)
        {
            for (int s=0; s<=es; ++s, Y-=N2)
            {
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
            }
            X -= (int)N2*(es+1);
        }

        Y+=N; ss+=str; es+=str; ++w;
    }
    X += ss*(int)N2;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y = *X; }
        Y -= N2;
        for (size_t l=Lk-1u; l>0u; --l, Y-=N2)
        {
            for (size_t n=N2; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
        }
        X += inc; Y += N2;
        es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Init Y
        for (size_t n=N2; n>0u; --n, ++Y) { *Y = -HUGE_VALF; }
        Y -= N2;

        //Get num valid samps
        int V = (ss<(int)Li) ? (int)Li-ss : 0;

        //Valid samps
        if (V>0)
        {
            for (int v=V; v>0; --v, Y-=N2)
            {
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
            }
        }
        else { X += ((int)Li-ss) * (int)N2; }

        X += N2*(str+(size_t)ss-Li); Y += N2;
        ss+=str; es+=str; ++w;
    }

    return 0;
}


int maxpool1_z (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode)
{
    if (str<1u) { fprintf(stderr,"error in maxpool1_z: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in maxpool1_z: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in maxpool1_z: Lk (kernel_size) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in maxpool1_z: N (num input neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in maxpool1_z: pad length must be > -Li\n"); return 1; }

    const int Ti = (int)Li + 2*pad;             //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in maxpool1_z: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in maxpool1_z: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);
    
    const size_t N2 = 2u*N;                     //to save multiplies
    const int inc = N2*((int)str-(int)Lk);      //fixed increment for X below
    size_t w=0u;                                //current window (frame)
    int ss=-pad, es=ss+(int)Lk-1;               //current start-samp, end-samp

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        //Init Y
        for (size_t n=N2; n>0u; --n, ++Y) { *Y = -HUGE_VAL; }
        Y -= N2;

        //Non-negative samps
        if (es>=0)
        {
            for (int s=0; s<=es; ++s, Y-=N2)
            {
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
            }
            X -= (int)N2*(es+1);
        }

        Y+=N; ss+=str; es+=str; ++w;
    }
    X += ss*(int)N2;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y = *X; }
        Y -= N2;
        for (size_t l=Lk-1u; l>0u; --l, Y-=N2)
        {
            for (size_t n=N2; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
        }
        for (size_t n=N2; n>0u; --n, ++Y) { *Y /= (double)Lk; }
        X += inc; Y += N2;
        es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Init Y
        for (size_t n=N2; n>0u; --n, ++Y) { *Y = -HUGE_VAL; }
        Y -= N2;

        //Get num valid samps
        int V = (ss<(int)Li) ? (int)Li-ss : 0;

        //Valid samps
        if (V>0)
        {
            for (int v=V; v>0; --v, Y-=N2)
            {
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { if (*X>*Y) { *Y = *X; } }
            }
        }
        else { X += ((int)Li-ss) * (int)N2; }

        X += N2*(str+(size_t)ss-Li); Y += N2;
        ss+=str; es+=str; ++w;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
