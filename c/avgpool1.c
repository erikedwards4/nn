//This is just like avgpool1d, but doesn't allow dilation.
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
//pad_mode:     int     padding mode (0=zeros, 1=replicate, 2=reflect, 3=circular, 4=no_count_pad)

#include <stdio.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int avgpool1_s (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in avgpool1_s: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in avgpool1_s: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in avgpool1_s: Lk (kernel_size) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in avgpool1_s: N (num input neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in avgpool1_s: pad length must be > -Li\n"); return 1; }
    if (pad_mode && pad>(int)Li) { fprintf(stderr,"error in avgpool1_s: Li (length of input vecs) must be >= pad length\n"); return 1; }
    if (pad_mode<0 || pad_mode>4) { fprintf(stderr,"error in avgpool1_s: pad_mode must be an int in {0,1,2,3,4}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;             //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in avgpool1_s: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in avgpool1_s: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);
    
    const int inc = (int)N*((int)str-(int)Lk);  //fixed increment for X below
    size_t w=0u;                                //current window (frame)
    int ss=-pad, es=ss+(int)Lk-1;               //current start-samp, end-samp
    int t = 0, prev_t = 0;                      //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        //Init Y
        for (size_t n=N; n>0u; --n, ++Y) { *Y = 0.0f; }
        Y -= N;

        //Negative samps
        if (pad_mode==1)        //repeat
        {
            for (int s=ss; s<0 && s<=es; ++s, X-=N, Y-=N)
            {
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
        }
        else if (pad_mode==2)   //reflect
        {
            for (int s=ss; s<0 && s<=es; ++s, Y-=N)
            {
                t = -s;         //PyTorch-style
                //t = -s - 1;   //Kaldi-style
                X += (t-prev_t) * (int)N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X -= prev_t * (int)N;
            prev_t = 0;
        }
        else if (pad_mode==3)   //circular
        {
            for (int s=ss; s<0 && s<=es; ++s, Y-=N)
            {
                t = (int)Li + s;
                X += (t-prev_t) * (int)N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X -= prev_t * (int)N;
            prev_t = 0;
        }

        //Non-negative samps
        if (es>=0)
        {
            for (int s=0; s<=es; ++s, Y-=N)
            {
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
            X -= (int)N*(es+1);
        }

        //Denominator
        if (pad_mode==4 && es>=0)
        {
            for (size_t n=N; n>0u; --n, ++Y) { *Y /= (float)(es+1); }
        }
        else
        {
            for (size_t n=N; n>0u; --n, ++Y) { *Y /= (float)Lk; }
        }

        ss+=str; es+=str; ++w;
    }
    X += ss*(int)N;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X; }
        Y -= N;
        for (size_t l=Lk-1u; l>0u; --l, Y-=N)
        {
            for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
        }
        for (size_t n=N; n>0u; --n, ++Y) { *Y /= (float)Lk; }
        X += inc; es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Init Y
        for (size_t n=N; n>0u; --n, ++Y) { *Y = 0.0f; }
        Y -= N;

        //Get num valid samps
        int V = (ss<(int)Li) ? (int)Li-ss : 0;

        //Valid samps
        if (V>0)
        {
            for (int v=V; v>0; --v, Y-=N)
            {
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
        }
        else { X += ((int)Li-ss) * (int)N; }

        //Non-valid samps
        if (pad_mode==1)        //repeat
        {
            for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s, Y-=N)
            {
                X -= N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
        }
        else if (pad_mode==2)   //reflect
        {
            prev_t = (int)Li;
            for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s, Y-=N)
            {
                t = 2*(int)Li - 2 - s;      //PyTorch-style
                //t = 2*(int)Li - 1 - s;    //Kaldi-style
                X += (t-prev_t) * (int)N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X += ((int)Li-prev_t) * (int)N;
        }
        else if (pad_mode==3)   //circular
        {
            prev_t = (int)Li;
            for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s, Y-=N)
            {
                t = s - (int)Li;
                X += (t-prev_t) * (int)N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X += ((int)Li-prev_t) * (int)N;
        }

        //Denominator
        if (pad_mode==4 && ss<(int)Li)
        {
            for (size_t n=N; n>0u; --n, ++Y) { *Y /= (float)((int)Li-ss); }
        }
        else
        {
            for (size_t n=N; n>0u; --n, ++Y) { *Y /= (float)Lk; }
        }

        X += N*(str+(size_t)ss-Li);
        ss+=str; es+=str; ++w;
    }

    return 0;
}


int avgpool1_d (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in avgpool1_d: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in avgpool1_d: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in avgpool1_d: Lk (kernel_size) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in avgpool1_d: N (num input neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in avgpool1_d: pad length must be > -Li\n"); return 1; }
    if (pad_mode && pad>(int)Li) { fprintf(stderr,"error in avgpool1_d: Li (length of input vecs) must be >= pad length\n"); return 1; }
    if (pad_mode<0 || pad_mode>4) { fprintf(stderr,"error in avgpool1_d: pad_mode must be an int in {0,1,2,3,4}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;             //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in avgpool1_d: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in avgpool1_d: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);
    
    const int inc = (int)N*((int)str-(int)Lk);  //fixed increment for X below
    size_t w=0u;                                //current window (frame)
    int ss=-pad, es=ss+(int)Lk-1;               //current start-samp, end-samp
    int t = 0, prev_t = 0;                      //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        //Init Y
        for (size_t n=N; n>0u; --n, ++Y) { *Y = 0.0; }
        Y -= N;

        //Negative samps
        if (pad_mode==1)        //repeat
        {
            for (int s=ss; s<0 && s<=es; ++s, X-=N, Y-=N)
            {
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
        }
        else if (pad_mode==2)   //reflect
        {
            for (int s=ss; s<0 && s<=es; ++s, Y-=N)
            {
                t = -s;         //PyTorch-style
                //t = -s - 1;   //Kaldi-style
                X += (t-prev_t) * (int)N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X -= prev_t * (int)N;
            prev_t = 0;
        }
        else if (pad_mode==3)   //circular
        {
            for (int s=ss; s<0 && s<=es; ++s, Y-=N)
            {
                t = (int)Li + s;
                X += (t-prev_t) * (int)N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X -= prev_t * (int)N;
            prev_t = 0;
        }

        //Non-negative samps
        if (es>=0)
        {
            for (int s=0; s<=es; ++s, Y-=N)
            {
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
            X -= (int)N*(es+1);
        }

        //Denominator
        if (pad_mode==4 && es>=0)
        {
            for (size_t n=N; n>0u; --n, ++Y) { *Y /= (double)(es+1); }
        }
        else
        {
            for (size_t n=N; n>0u; --n, ++Y) { *Y /= (double)Lk; }
        }

        ss+=str; es+=str; ++w;
    }
    X += ss*(int)N;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y = *X; }
        Y -= N;
        for (size_t l=Lk-1u; l>0u; --l, Y-=N)
        {
            for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
        }
        for (size_t n=N; n>0u; --n, ++Y) { *Y /= (double)Lk; }
        X += inc; es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Init Y
        for (size_t n=N; n>0u; --n, ++Y) { *Y = 0.0; }
        Y -= N;

        //Get num valid samps
        int V = (ss<(int)Li) ? (int)Li-ss : 0;

        //Valid samps
        if (V>0)
        {
            for (int v=V; v>0; --v, Y-=N)
            {
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
        }
        else { X += ((int)Li-ss) * (int)N; }

        //Non-valid samps
        if (pad_mode==1)        //repeat
        {
            for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s, Y-=N)
            {
                X -= N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
        }
        else if (pad_mode==2)   //reflect
        {
            prev_t = (int)Li;
            for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s, Y-=N)
            {
                t = 2*(int)Li - 2 - s;      //PyTorch-style
                //t = 2*(int)Li - 1 - s;    //Kaldi-style
                X += (t-prev_t) * (int)N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X += ((int)Li-prev_t) * (int)N;
        }
        else if (pad_mode==3)   //circular
        {
            prev_t = (int)Li;
            for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s, Y-=N)
            {
                t = s - (int)Li;
                X += (t-prev_t) * (int)N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X += ((int)Li-prev_t) * (int)N;
        }

        //Denominator
        if (pad_mode==4 && ss<(int)Li)
        {
            for (size_t n=N; n>0u; --n, ++Y) { *Y /= (double)((int)Li-ss); }
        }
        else
        {
            for (size_t n=N; n>0u; --n, ++Y) { *Y /= (double)Lk; }
        }

        X += N*(str+(size_t)ss-Li);
        ss+=str; es+=str; ++w;
    }

    return 0;
}


int avgpool1_c (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in avgpool1_c: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in avgpool1_c: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in avgpool1_c: Lk (kernel_size) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in avgpool1_c: N (num input neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in avgpool1_c: pad length must be > -Li\n"); return 1; }
    if (pad_mode && pad>(int)Li) { fprintf(stderr,"error in avgpool1_c: Li (length of input vecs) must be >= pad length\n"); return 1; }
    if (pad_mode<0 || pad_mode>4) { fprintf(stderr,"error in avgpool1_c: pad_mode must be an int in {0,1,2,3,4}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;             //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in avgpool1_c: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in avgpool1_c: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);
    
    const size_t N2 = 2u*N;                     //to save multiplies
    const int inc = N2*((int)str-(int)Lk);      //fixed increment for X below
    size_t w=0u;                                //current window (frame)
    int ss=-pad, es=ss+(int)Lk-1;               //current start-samp, end-samp
    int t = 0, prev_t = 0;                      //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        //Init Y
        for (size_t n=N2; n>0u; --n, ++Y) { *Y = 0.0f; }
        Y -= N2;

        //Negative samps
        if (pad_mode==1)        //repeat
        {
            for (int s=ss; s<0 && s<=es; ++s, X-=N2, Y-=N2)
            {
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
        }
        else if (pad_mode==2)   //reflect
        {
            for (int s=ss; s<0 && s<=es; ++s, Y-=N2)
            {
                t = -s;         //PyTorch-style
                //t = -s - 1;   //Kaldi-style
                X += (t-prev_t) * N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X -= prev_t*(int)N2;
            prev_t = 0;
        }
        else if (pad_mode==3)   //circular
        {
            for (int s=ss; s<0 && s<=es; ++s, Y-=N2)
            {
                t = (int)Li + s;
                X += (t-prev_t) * N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X -= prev_t*(int)N2;
            prev_t = 0;
        }

        //Non-negative samps
        if (es>=0)
        {
            for (int s=0; s<=es; ++s, Y-=N2)
            {
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
            X -= (int)N2*(es+1);
        }

        //Denominator
        if (pad_mode==4 && es>=0)
        {
            for (size_t n=N2; n>0u; --n, ++Y) { *Y /= (float)(es+1); }
        }
        else
        {
            for (size_t n=N2; n>0u; --n, ++Y) { *Y /= (float)Lk; }
        }

        ss+=str; es+=str; ++w;
    }
    X += ss*(int)N2;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y = *X; }
        Y -= N2;
        for (size_t l=Lk-1u; l>0u; --l, Y-=N2)
        {
            for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
        }
        for (size_t n=N2; n>0u; --n, ++Y) { *Y /= (float)Lk; }
        X += inc; es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Init Y
        for (size_t n=N2; n>0u; --n, ++Y) { *Y = 0.0f; }
        Y -= N2;

        //Get num valid samps
        int V = (ss<(int)Li) ? (int)Li-ss : 0;

        //Valid samps
        if (V>0)
        {
            for (int v=V; v>0; --v, Y-=N2)
            {
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
        }
        else { X += ((int)Li-ss) * (int)N2; }

        //Non-valid samps
        if (pad_mode==1)        //repeat
        {
            for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s, Y-=N2)
            {
                X -= N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
        }
        else if (pad_mode==2)   //reflect
        {
            prev_t = (int)Li;
            for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s, Y-=N2)
            {
                t = 2*(int)Li - 2 - s;      //PyTorch-style
                //t = 2*(int)Li - 1 - s;    //Kaldi-style
                X += (t-prev_t) * (int)N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X += ((int)Li-prev_t) * (int)N2;
        }
        else if (pad_mode==3)   //circular
        {
            prev_t = (int)Li;
            for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s, Y-=N2)
            {
                t = s - (int)Li;
                X += (t-prev_t) * (int)N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X += ((int)Li-prev_t) * (int)N2;
        }

        //Denominator
        if (pad_mode==4 && ss<(int)Li)
        {
            for (size_t n=N2; n>0u; --n, ++Y) { *Y /= (float)((int)Li-ss); }
        }
        else
        {
            for (size_t n=N2; n>0u; --n, ++Y) { *Y /= (float)Lk; }
        }

        X += N2*(str+(size_t)ss-Li);
        ss+=str; es+=str; ++w;
    }

    return 0;
}


int avgpool1_z (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in avgpool1_z: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in avgpool1_z: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in avgpool1_z: Lk (kernel_size) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in avgpool1_z: N (num input neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in avgpool1_z: pad length must be > -Li\n"); return 1; }
    if (pad_mode && pad>(int)Li) { fprintf(stderr,"error in avgpool1_z: Li (length of input vecs) must be >= pad length\n"); return 1; }
    if (pad_mode<0 || pad_mode>4) { fprintf(stderr,"error in avgpool1_z: pad_mode must be an int in {0,1,2,3,4}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;             //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in avgpool1_z: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in avgpool1_z: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);
    
    const size_t N2 = 2u*N;                     //to save multiplies
    const int inc = N2*((int)str-(int)Lk);      //fixed increment for X below
    size_t w=0u;                                //current window (frame)
    int ss=-pad, es=ss+(int)Lk-1;               //current start-samp, end-samp
    int t = 0, prev_t = 0;                      //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        //Init Y
        for (size_t n=N2; n>0u; --n, ++Y) { *Y = 0.0; }
        Y -= N2;

        //Negative samps
        if (pad_mode==1)        //repeat
        {
            for (int s=ss; s<0 && s<=es; ++s, X-=N2, Y-=N2)
            {
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
        }
        else if (pad_mode==2)   //reflect
        {
            for (int s=ss; s<0 && s<=es; ++s, Y-=N2)
            {
                t = -s;         //PyTorch-style
                //t = -s - 1;   //Kaldi-style
                X += (t-prev_t) * N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X -= prev_t*(int)N2;
            prev_t = 0;
        }
        else if (pad_mode==3)   //circular
        {
            for (int s=ss; s<0 && s<=es; ++s, Y-=N2)
            {
                t = (int)Li + s;
                X += (t-prev_t) * N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X -= prev_t*(int)N2;
            prev_t = 0;
        }

        //Non-negative samps
        if (es>=0)
        {
            for (int s=0; s<=es; ++s, Y-=N2)
            {
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
            X -= (int)N2*(es+1);
        }

        //Denominator
        if (pad_mode==4 && es>=0)
        {
            for (size_t n=N2; n>0u; --n, ++Y) { *Y /= (double)(es+1); }
        }
        else
        {
            for (size_t n=N2; n>0u; --n, ++Y) { *Y /= (double)Lk; }
        }

        ss+=str; es+=str; ++w;
    }
    X += ss*(int)N2;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y = *X; }
        Y -= N2;
        for (size_t l=Lk-1u; l>0u; --l, Y-=N2)
        {
            for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
        }
        for (size_t n=N2; n>0u; --n, ++Y) { *Y /= (double)Lk; }
        X += inc; es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Init Y
        for (size_t n=N2; n>0u; --n, ++Y) { *Y = 0.0; }
        Y -= N2;

        //Get num valid samps
        int V = (ss<(int)Li) ? (int)Li-ss : 0;

        //Valid samps
        if (V>0)
        {
            for (int v=V; v>0; --v, Y-=N2)
            {
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
        }
        else { X += ((int)Li-ss) * (int)N2; }

        //Non-valid samps
        if (pad_mode==1)        //repeat
        {
            for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s, Y-=N2)
            {
                X -= N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
        }
        else if (pad_mode==2)   //reflect
        {
            prev_t = (int)Li;
            for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s, Y-=N2)
            {
                t = 2*(int)Li - 2 - s;      //PyTorch-style
                //t = 2*(int)Li - 1 - s;    //Kaldi-style
                X += (t-prev_t) * (int)N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X += ((int)Li-prev_t) * (int)N2;
        }
        else if (pad_mode==3)   //circular
        {
            prev_t = (int)Li;
            for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s, Y-=N2)
            {
                t = s - (int)Li;
                X += (t-prev_t) * (int)N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X += ((int)Li-prev_t) * (int)N2;
        }

        //Denominator
        if (pad_mode==4 && ss<(int)Li)
        {
            for (size_t n=N2; n>0u; --n, ++Y) { *Y /= (double)((int)Li-ss); }
        }
        else
        {
            for (size_t n=N2; n>0u; --n, ++Y) { *Y /= (double)Lk; }
        }

        X += N2*(str+(size_t)ss-Li);
        ss+=str; es+=str; ++w;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
