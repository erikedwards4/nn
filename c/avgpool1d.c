//This does 1D average pooling, like PyTorch AvgPool1d,
//but uses the full set of input params/opts available for conv1d.

//Considered as a component of the present framework,
//this is an IN (input) method for N==N neurons.
//In PyTorch, N and N are called C, i.e. num chans.

//The parameters are kernel_size == Lk (i.e., kernel width);
//and stride, dilation and padding for the avgpooling itself.
//Note that no kernel is input; it is implicitly just ones.

//If using dil==1, use avgpool1.

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
//pad_mode:     int     padding mode (0=zeros, 1=replicate, 2=reflect, 3=circular, 4=no_count_pad)

#include <stdio.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int avgpool1d_s (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in avgpool1d_s: str (stride) must be positive\n"); return 1; }
    if (dil<1u) { fprintf(stderr,"error in avgpool1d_s: dil (dilation) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in avgpool1d_s: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in avgpool1d_s: Lk (kernel_size) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in avgpool1d_s: N (num input neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in avgpool1d_s: pad length must be > -Li\n"); return 1; }
    if (pad_mode && pad>(int)Li) { fprintf(stderr,"error in avgpool1d_s: Li (length of input vecs) must be >= pad length\n"); return 1; }
    if (pad_mode<0 || pad_mode>4) { fprintf(stderr,"error in avgpool1d_s: pad_mode must be an int in {0,1,2,3,4}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in avgpool1d_s: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in avgpool1d_s: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

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
        for (size_t n=N; n>0u; --n, ++Y) { *Y = 0.0f; }
        Y -= N;

        //Negative samps
        if (pad_mode==1)        //repeat
        {
            for (int s=ss; s<0 && s<=es; s+=dil, X-=N, Y-=N)
            {
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
        }
        else if (pad_mode==2)   //reflect
        {
            for (int s=ss; s<0 && s<=es; s+=dil, Y-=N)
            {
                t = -s;         //PyTorch-style
                //t = -s - 1;   //Kaldi-style
                //this ensures reflected extrapolation to any length
                while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                X += (t-prev_t) * (int)N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X -= prev_t * (int)N;
            prev_t = 0;
        }
        else if (pad_mode==3)   //circular
        {
            for (int s=ss; s<0 && s<=es; s+=dil, Y-=N)
            {
                t = (int)Li + s;    //this ensures circular extrapolation to any length
                while (t<0) { t += (int)Li; }
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
            for (int s=es%(int)dil; s<=es; s+=dil, Y-=N)
            {
                X += (s-prev_t) * (int)N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = s + 1;
            }
        }
        X -= prev_t * (int)N;
        prev_t = 0;

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
        X += jump; Y -= N;
        for (size_t l=Lk-1u; l>0u; --l, X+=jump, Y-=N)
        {
            for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
        }
        for (size_t n=N; n>0u; --n, ++Y) { *Y /= (float)Lk; }
        X -= (int)N*((int)(Lk*dil)-(int)str);
        es+=str; ++w;
    }
    prev_t = ss = es - Tk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Init Y
        for (size_t n=N; n>0u; --n, ++Y) { *Y = 0.0f; }
        Y -= N;

        //Get num valid samps
        int V = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

        //Valid samps
        for (int s=ss; s<(int)Li; s+=dil, Y-=N)
        {
            X += (s-prev_t) * (int)N;
            for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
            prev_t = s + 1;
        }

        //Non-valid samps
        if (pad_mode==1)        //repeat
        {
            X += ((int)Li-1-prev_t) * (int)N;
            for (int s=ss+V*(int)dil; s<=es; s+=dil, X-=N, Y-=N)
            {
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
            prev_t = (int)Li - 1;
        }
        else if (pad_mode==2)   //reflect
        {
            for (int s=ss+V*(int)dil; s<=es; s+=dil, Y-=N)
            {
                t = 2*(int)Li - 2 - s;      //PyTorch-style
                //t = 2*(int)Li - 1 - s;    //Kaldi-style
                //this ensures reflected extrapolation to any length
                while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                X += (t-prev_t) * (int)N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
        }
        else if (pad_mode==3)   //circular
        {
            for (int s=ss+V*(int)dil; s<=es; s+=dil, Y-=N)
            {
                t = s - (int)Li;    //this ensures circular extrapolation to any length
                while (t>=(int)Li) { t -= (int)Li; }
                X += (t-prev_t) * (int)N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
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

        ss+=str; es+=str; ++w;
    }

    return 0;
}


int avgpool1d_d (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in avgpool1d_d: str (stride) must be positive\n"); return 1; }
    if (dil<1u) { fprintf(stderr,"error in avgpool1d_d: dil (dilation) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in avgpool1d_d: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in avgpool1d_d: Lk (kernel_size) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in avgpool1d_d: N (num input neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in avgpool1d_d: pad length must be > -Li\n"); return 1; }
    if (pad_mode && pad>(int)Li) { fprintf(stderr,"error in avgpool1d_d: Li (length of input vecs) must be >= pad length\n"); return 1; }
    if (pad_mode<0 || pad_mode>4) { fprintf(stderr,"error in avgpool1d_d: pad_mode must be an int in {0,1,2,3,4}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in avgpool1d_d: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in avgpool1d_d: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

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
        for (size_t n=N; n>0u; --n, ++Y) { *Y = 0.0; }
        Y -= N;

        //Negative samps
        if (pad_mode==1)        //repeat
        {
            for (int s=ss; s<0 && s<=es; s+=dil, X-=N, Y-=N)
            {
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
        }
        else if (pad_mode==2)   //reflect
        {
            for (int s=ss; s<0 && s<=es; s+=dil, Y-=N)
            {
                t = -s;         //PyTorch-style
                //t = -s - 1;   //Kaldi-style
                //this ensures reflected extrapolation to any length
                while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                X += (t-prev_t) * (int)N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X -= prev_t * (int)N;
            prev_t = 0;
        }
        else if (pad_mode==3)   //circular
        {
            for (int s=ss; s<0 && s<=es; s+=dil, Y-=N)
            {
                t = (int)Li + s;    //this ensures circular extrapolation to any length
                while (t<0) { t += (int)Li; }
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
            for (int s=es%(int)dil; s<=es; s+=dil, Y-=N)
            {
                X += (s-prev_t) * (int)N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = s + 1;
            }
        }
        X -= prev_t * (int)N;
        prev_t = 0;

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
        X += jump; Y -= N;
        for (size_t l=Lk-1u; l>0u; --l, X+=jump, Y-=N)
        {
            for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
        }
        for (size_t n=N; n>0u; --n, ++Y) { *Y /= (double)Lk; }
        X -= (int)N*((int)(Lk*dil)-(int)str);
        es+=str; ++w;
    }
    prev_t = ss = es - Tk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Init Y
        for (size_t n=N; n>0u; --n, ++Y) { *Y = 0.0; }
        Y -= N;

        //Get num valid samps
        int V = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

        //Valid samps
        for (int s=ss; s<(int)Li; s+=dil, Y-=N)
        {
            X += (s-prev_t) * (int)N;
            for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
            prev_t = s + 1;
        }

        //Non-valid samps
        if (pad_mode==1)        //repeat
        {
            X += ((int)Li-1-prev_t) * (int)N;
            for (int s=ss+V*(int)dil; s<=es; s+=dil, X-=N, Y-=N)
            {
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
            prev_t = (int)Li - 1;
        }
        else if (pad_mode==2)   //reflect
        {
            for (int s=ss+V*(int)dil; s<=es; s+=dil, Y-=N)
            {
                t = 2*(int)Li - 2 - s;      //PyTorch-style
                //t = 2*(int)Li - 1 - s;    //Kaldi-style
                //this ensures reflected extrapolation to any length
                while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                X += (t-prev_t) * (int)N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
        }
        else if (pad_mode==3)   //circular
        {
            for (int s=ss+V*(int)dil; s<=es; s+=dil, Y-=N)
            {
                t = s - (int)Li;    //this ensures circular extrapolation to any length
                while (t>=(int)Li) { t -= (int)Li; }
                X += (t-prev_t) * (int)N;
                for (size_t n=N; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
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

        ss+=str; es+=str; ++w;
    }

    return 0;
}


int avgpool1d_c (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in avgpool1d_c: str (stride) must be positive\n"); return 1; }
    if (dil<1u) { fprintf(stderr,"error in avgpool1d_c: dil (dilation) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in avgpool1d_c: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in avgpool1d_c: Lk (kernel_size) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in avgpool1d_c: N (num input neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in avgpool1d_c: pad length must be > -Li\n"); return 1; }
    if (pad_mode && pad>(int)Li) { fprintf(stderr,"error in avgpool1d_c: Li (length of input vecs) must be >= pad length\n"); return 1; }
    if (pad_mode<0 || pad_mode>4) { fprintf(stderr,"error in avgpool1d_c: pad_mode must be an int in {0,1,2,3,4}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in avgpool1d_c: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in avgpool1d_c: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

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
        for (size_t n=N2; n>0u; --n, ++Y) { *Y = 0.0f; }
        Y -= N2;

        //Negative samps
        if (pad_mode==1)        //repeat
        {
            for (int s=ss; s<0 && s<=es; s+=dil, X-=N2, Y-=N2)
            {
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
        }
        else if (pad_mode==2)   //reflect
        {
            for (int s=ss; s<0 && s<=es; s+=dil, Y-=N2)
            {
                t = -s;         //PyTorch-style
                //t = -s - 1;   //Kaldi-style
                //this ensures reflected extrapolation to any length
                while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                X += (t-prev_t) * (int)N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X -= prev_t * (int)N2;
            prev_t = 0;
        }
        else if (pad_mode==3)   //circular
        {
            for (int s=ss; s<0 && s<=es; s+=dil, Y-=N2)
            {
                t = (int)Li + s;    //this ensures circular extrapolation to any length
                while (t<0) { t += (int)Li; }
                X += (t-prev_t) * (int)N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X -= prev_t * (int)N2;
            prev_t = 0;
        }

        //Non-negative samps
        if (es>=0)
        {
            for (int s=es%(int)dil; s<=es; s+=dil, Y-=N2)
            {
                X += (s-prev_t) * (int)N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = s + 1;
            }
        }
        X -= prev_t * (int)N2;
        prev_t = 0;

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
        X += jump; Y -= N2;
        for (size_t l=Lk-1u; l>0u; --l, X+=jump, Y-=N2)
        {
            for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
        }
        for (size_t n=N2; n>0u; --n, ++Y) { *Y /= (float)Lk; }
        X -= (int)N2*((int)(Lk*dil)-(int)str);
        es+=str; ++w;
    }
    prev_t = ss = es - Tk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Init Y
        for (size_t n=N2; n>0u; --n, ++Y) { *Y = 0.0f; }
        Y -= N2;

        //Get num valid samps
        int V = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

        //Valid samps
        for (int s=ss; s<(int)Li; s+=dil, Y-=N2)
        {
            X += (s-prev_t) * (int)N2;
            for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
            prev_t = s + 1;
        }

        //Non-valid samps
        if (pad_mode==1)        //repeat
        {
            X += ((int)Li-1-prev_t) * (int)N2;
            for (int s=ss+V*(int)dil; s<=es; s+=dil, X-=N2, Y-=N2)
            {
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
            prev_t = (int)Li - 1;
        }
        else if (pad_mode==2)   //reflect
        {
            for (int s=ss+V*(int)dil; s<=es; s+=dil, Y-=N2)
            {
                t = 2*(int)Li - 2 - s;      //PyTorch-style
                //t = 2*(int)Li - 1 - s;    //Kaldi-style
                //this ensures reflected extrapolation to any length
                while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                X += (t-prev_t) * (int)N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
        }
        else if (pad_mode==3)   //circular
        {
            for (int s=ss+V*(int)dil; s<=es; s+=dil, Y-=N2)
            {
                t = s - (int)Li;    //this ensures circular extrapolation to any length
                while (t>=(int)Li) { t -= (int)Li; }
                X += (t-prev_t) * (int)N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
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

        ss+=str; es+=str; ++w;
    }

    return 0;
}


int avgpool1d_z (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in avgpool1d_z: str (stride) must be positive\n"); return 1; }
    if (dil<1u) { fprintf(stderr,"error in avgpool1d_z: dil (dilation) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in avgpool1d_z: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in avgpool1d_z: Lk (kernel_size) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in avgpool1d_z: N (num input neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in avgpool1d_z: pad length must be > -Li\n"); return 1; }
    if (pad_mode && pad>(int)Li) { fprintf(stderr,"error in avgpool1d_z: Li (length of input vecs) must be >= pad length\n"); return 1; }
    if (pad_mode<0 || pad_mode>4) { fprintf(stderr,"error in avgpool1d_z: pad_mode must be an int in {0,1,2,3,4}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in avgpool1d_z: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in avgpool1d_z: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

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
        for (size_t n=N2; n>0u; --n, ++Y) { *Y = 0.0; }
        Y -= N2;

        //Negative samps
        if (pad_mode==1)        //repeat
        {
            for (int s=ss; s<0 && s<=es; s+=dil, X-=N2, Y-=N2)
            {
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
        }
        else if (pad_mode==2)   //reflect
        {
            for (int s=ss; s<0 && s<=es; s+=dil, Y-=N2)
            {
                t = -s;         //PyTorch-style
                //t = -s - 1;   //Kaldi-style
                //this ensures reflected extrapolation to any length
                while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                X += (t-prev_t) * (int)N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X -= prev_t * (int)N2;
            prev_t = 0;
        }
        else if (pad_mode==3)   //circular
        {
            for (int s=ss; s<0 && s<=es; s+=dil, Y-=N2)
            {
                t = (int)Li + s;    //this ensures circular extrapolation to any length
                while (t<0) { t += (int)Li; }
                X += (t-prev_t) * (int)N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
            X -= prev_t * (int)N2;
            prev_t = 0;
        }

        //Non-negative samps
        if (es>=0)
        {
            for (int s=es%(int)dil; s<=es; s+=dil, Y-=N2)
            {
                X += (s-prev_t) * (int)N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = s + 1;
            }
        }
        X -= prev_t * (int)N2;
        prev_t = 0;

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
        X += jump; Y -= N2;
        for (size_t l=Lk-1u; l>0u; --l, X+=jump, Y-=N2)
        {
            for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
        }
        for (size_t n=N2; n>0u; --n, ++Y) { *Y /= (double)Lk; }
        X -= (int)N2*((int)(Lk*dil)-(int)str);
        es+=str; ++w;
    }
    prev_t = ss = es - Tk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Init Y
        for (size_t n=N2; n>0u; --n, ++Y) { *Y = 0.0; }
        Y -= N2;

        //Get num valid samps
        int V = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

        //Valid samps
        for (int s=ss; s<(int)Li; s+=dil, Y-=N2)
        {
            X += (s-prev_t) * (int)N2;
            for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
            prev_t = s + 1;
        }

        //Non-valid samps
        if (pad_mode==1)        //repeat
        {
            X += ((int)Li-1-prev_t) * (int)N2;
            for (int s=ss+V*(int)dil; s<=es; s+=dil, X-=N2, Y-=N2)
            {
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
            }
            prev_t = (int)Li - 1;
        }
        else if (pad_mode==2)   //reflect
        {
            for (int s=ss+V*(int)dil; s<=es; s+=dil, Y-=N2)
            {
                t = 2*(int)Li - 2 - s;      //PyTorch-style
                //t = 2*(int)Li - 1 - s;    //Kaldi-style
                //this ensures reflected extrapolation to any length
                while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                X += (t-prev_t) * (int)N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
        }
        else if (pad_mode==3)   //circular
        {
            for (int s=ss+V*(int)dil; s<=es; s+=dil, Y-=N2)
            {
                t = s - (int)Li;    //this ensures circular extrapolation to any length
                while (t>=(int)Li) { t -= (int)Li; }
                X += (t-prev_t) * (int)N2;
                for (size_t n=N2; n>0u; --n, ++X, ++Y) { *Y += *X; }
                prev_t = t + 1;
            }
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

        ss+=str; es+=str; ++w;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
