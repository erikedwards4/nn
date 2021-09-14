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
#include <string.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int avgpool1_s (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int avgpool1_d (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int avgpool1_c (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int avgpool1_z (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);


int avgpool1_s (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in avgpool1_s: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in avgpool1_s: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in avgpool1_s: Lk (kernel_size) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in avgpool1_s: N (num input neurons) must be positive\n"); return 1; }
    if (pad_mode<0 || pad_mode>4) { fprintf(stderr,"error in avgpool1_s: pad_mode must be an int in {0,1,2,3,4}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;     //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in avgpool1_s: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in avgpool1_s: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);
    
    size_t w=0u;                        //current window (frame)
    float sm;                           //intermediate sum
    int ss=-pad, es=ss+(int)Lk-1;       //current start-samp, end-samp
    int t = 0, prev_t = 0;              //non-negative samps during extrapolation (padding)

    //struct timespec tic, toc; clock_gettime(CLOCK_REALTIME,&tic);

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        for (size_t o=N; o>0u; --o, ++Y)
        {
            sm = 0.0f;

            //Negative samps
            if (pad_mode==1)   //repeat
            {
                for (int s=ss; s<0 && s<=es; ++s, X-=N)
                {
                    for (size_t i=N; i>0u; --i, ++X) { sm += *X; }
                }
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    t = -s - 1;         //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { t = (n<0) ? -n-1 : (n<(int)Li) ? t : 2*(int)Li-1-n; }
                    X += (n-prev_t) * (int)N;
                    for (size_t i=N; i>0u; --i, ++X) { sm += *X; }
                    prev_t = t + 1;
                }
                X -= prev_t * (int)N;
                prev_t = 0;
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    t = (int)Li + s;    //this ensures circular extrapolation to any length
                    while (n<0) { t += (int)Li; }
                    X += (n-prev_t) * (int)N;
                    for (size_t i=N; i>0u; --i, ++X) { sm += *X; }
                    prev_t = t + 1;
                }
                X -= prev_t * (int)N;
                prev_t = 0;
            }

            //Nn-negative samps
            if (es>=0)
            {
                for (int i=(es+1)*(int)N; i>0; --i, ++X) { sm += *X; }
                X -= (es+1)*(int)N;
            }

            if (pad_mode==4) { *Y = sm / den; }
            if (pad_mode==4) { *Y = sm / (float)N; }
        }
        K -= Nk;
        ss+=str; es+=str; ++w;
    }
    X += ss*(int)N;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t o=N; o>0u; --o, ++Y)
        {
            sm = 0.0f;
            for (size_t i=N*Lk; i>0u; --i, ++X) { sm += *X; }
            *Y = sm;
            X -= N*Lk;
        }
        K -= Nk; X += N*str;
        es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int v = (int)Li - ss;

        for (size_t o=N; o>0u; --o, ++Y)
        {
            sm = 0.0f;

            //Valid samps
            if (v>0)
            {
                for (int i=v*(int)N; i>0; --i, ++X) { sm += *X; }
            }
            else { X += ((int)Li-ss) * (int)N; }

            //Nn-valid samps
            if (pad_mode==0)        //zeros
            {
                K += (v>0) ? N*(Lk-(size_t)v) : N*Lk;
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    X -= N;
                    for (size_t i=N; i>0u; --i, ++X) { sm += *X; }
                }
            }
            else if (pad_mode==2)   //reflect
            {
                prev_t = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    t = 2*(int)Li-1-s;  //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { t = (n<0) ? -n-1 : (n<(int)Li) ? t : 2*(int)Li-1-n; }
                    X += (n-prev_t) * (int)N;
                    for (size_t i=N; i>0u; --i, ++X) { sm += *X; }
                    prev_t = t + 1;
                }
                X += ((int)Li-prev_t) * (int)N;
            }
            else                    //circular
            {
                prev_t = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    t = s - (int)Li;    //this ensures circular extrapolation to any length
                    while (n>=(int)Li) { t -= (int)Li; }
                    X += (n-prev_t) * (int)N;
                    for (size_t i=N; i>0u; --i, ++X) { sm += *X; }
                    prev_t = t + 1;
                }
                X += ((int)Li-prev_t) * (int)N;
            }
            X += (ss-(int)Li) * (int)N;

            *Y = sm;
        }
        K -= Nk; X += str*N;
        ss+=str; es+=str; ++w;
    }

    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(double)(toc.tv_sec-tic.tv_sec)*1e3+(double)(toc.tv_nsec-tic.tv_nsec)/1e6);

    return 0;
}


int avgpool1_d (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in avgpool1_d: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in avgpool1_d: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in avgpool1_d: Lk (kernel_size) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in avgpool1_d: N (num input neurons) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in avgpool1_d: N (num output neurons) must be positive\n"); return 1; }
    if (pad_mode<0 || pad_mode>4) { fprintf(stderr,"error in avgpool1_d: pad_mode must be an int in {0,1,2,3,4}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;     //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in avgpool1_d: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in avgpool1_d: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);
    
    const size_t Nk = N*Lk*N;         //total num elements in K
    size_t w=0u;                        //current window (frame)
    double sm;                          //intermediate sum
    int ss=-pad, es=ss+(int)Lk-1;       //current start-samp, end-samp
    int t = 0, prev_t = 0;              //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        for (size_t o=N; o>0u; --o, ++Y)
        {
            sm = 0.0;

            //Negative samps
            if (pad_mode==0)        //zeros
            {
                K += (es<0) ? N*Lk : N*(size_t)(-ss);
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=ss; s<0 && s<=es; ++s, X-=N)
                {
                    for (size_t i=N; i>0u; --i, ++X) { sm += *X; }
                }
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    t = -s - 1;         //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { t = (n<0) ? -n-1 : (n<(int)Li) ? t : 2*(int)Li-1-n; }
                    X += (n-prev_t) * (int)N;
                    for (size_t i=N; i>0u; --i, ++X) { sm += *X; }
                    prev_t = t + 1;
                }
                X -= prev_t * (int)N;
                prev_t = 0;
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    t = (int)Li + s;    //this ensures circular extrapolation to any length
                    while (n<0) { t += (int)Li; }
                    X += (n-prev_t) * (int)N;
                    for (size_t i=N; i>0u; --i, ++X) { sm += *X; }
                    prev_t = t + 1;
                }
                X -= prev_t * (int)N;
                prev_t = 0;
            }

            //Nn-negative samps
            if (es>=0)
            {
                for (int i=(es+1)*(int)N; i>0; --i, ++X) { sm += *X; }
                X -= (es+1)*(int)N;
            }

            *Y = sm;
        }
        K -= Nk;
        ss+=str; es+=str; ++w;
    }
    X += ss*(int)N;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t o=N; o>0u; --o, ++Y)
        {
            sm = 0.0;
            for (size_t i=N*Lk; i>0u; --i, ++X) { sm += *X; }
            *Y = sm;
            X -= N*Lk;
        }
        K -= Nk; X += N*str;
        es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int v = (int)Li - ss;

        for (size_t o=N; o>0u; --o, ++Y)
        {
            sm = 0.0;

            //Valid samps
            if (v>0)
            {
                for (int i=v*(int)N; i>0; --i, ++X) { sm += *X; }
            }
            else { X += ((int)Li-ss) * (int)N; }

            //Nn-valid samps
            if (pad_mode==0)        //zeros
            {
                K += (v>0) ? N*(Lk-(size_t)v) : N*Lk;
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    X -= N;
                    for (size_t i=N; i>0u; --i, ++X) { sm += *X; }
                }
            }
            else if (pad_mode==2)   //reflect
            {
                prev_t = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    t = 2*(int)Li-1-s;  //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { t = (n<0) ? -n-1 : (n<(int)Li) ? t : 2*(int)Li-1-n; }
                    X += (n-prev_t) * (int)N;
                    for (size_t i=N; i>0u; --i, ++X) { sm += *X; }
                    prev_t = t + 1;
                }
                X += ((int)Li-prev_t) * (int)N;
            }
            else                    //circular
            {
                prev_t = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    t = s - (int)Li;    //this ensures circular extrapolation to any length
                    while (n>=(int)Li) { t -= (int)Li; }
                    X += (n-prev_t) * (int)N;
                    for (size_t i=N; i>0u; --i, ++X) { sm += *X; }
                    prev_t = t + 1;
                }
                X += ((int)Li-prev_t) * (int)N;
            }
            X += (ss-(int)Li) * (int)N;

            *Y = sm;
        }
        K -= Nk; X += str*N;
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
    if (N<1u) { fprintf(stderr,"error in avgpool1_c: N (num output neurons) must be positive\n"); return 1; }
    if (pad_mode<0 || pad_mode>4) { fprintf(stderr,"error in avgpool1_c: pad_mode must be an int in {0,1,2,3,4}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;     //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in avgpool1_c: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in avgpool1_c: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);

    const size_t Nb = 2u*N;            //total num elements in B
    const size_t Nk = 2u*N*Lk*N;      //total num elements in K
    size_t w=0u;                        //current window (frame)
    float smr, smi;                     //intermediate sums
    int ss=-pad, es=ss+(int)Lk-1;       //current start-samp, end-samp
    int t = 0, prev_t = 0;              //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        for (size_t o=N; o>0u; --o)
        {
            smr = *B++; smi = *B++;

            //Negative samps
            if (pad_mode==0)        //zeros
            {
                K += (es<0) ? 2u*N*Lk : 2u*N*(size_t)(-ss);
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=ss; s<0 && s<=es; ++s, X-=2u*N)
                {
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                }
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    t = -s - 1;         //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { t = (n<0) ? -n-1 : (n<(int)Li) ? t : 2*(int)Li-1-n; }
                    X += 2 * (n-prev_t) * (int)N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
                X -= 2 * prev_t * (int)N;
                prev_t = 0;
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    t = (int)Li + s;    //this ensures circular extrapolation to any length
                    while (n<0) { t += (int)Li; }
                    X += 2 * (n-prev_t) * (int)N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
                X -= 2 * prev_t * (int)N;
                prev_t = 0;
            }

            //Nn-negative samps
            if (es>=0)
            {
                for (int i=(es+1)*(int)N; i>0; --i, X+=2, K+=2)
                {
                    smr += *X**K - *(X+1)**(K+1);
                    smi += *X**(K+1) + *(X+1)**K;
                }
                X -= 2*(es+1)*(int)N;
            }

            *Y++ = smr; *Y++ = smi;
        }
        K -= Nk;
        ss+=str; es+=str; ++w;
    }
    X += 2*ss*(int)N;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t o=N; o>0u; --o)
        {
            smr = *B++; smi = *B++;
            for (size_t i=N*Lk; i>0u; --i, X+=2, K+=2)
            {
                smr += *X**K - *(X+1)**(K+1);
                smi += *X**(K+1) + *(X+1)**K;
            }
            *Y++ = smr; *Y++ = smi;
            X -= 2u*N*Lk;
        }
        K -= Nk; X += 2u*N*str;
        es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int v = (int)Li - ss;

        for (size_t o=N; o>0u; --o)
        {
            smr = *B++; smi = *B++;

            //Valid samps
            if (v>0)
            {
                for (int i=v*(int)N; i>0; --i, X+=2, K+=2)
                {
                    smr += *X**K - *(X+1)**(K+1);
                    smi += *X**(K+1) + *(X+1)**K;
                }
            }
            else { X += 2 * ((int)Li-ss) * (int)N; }

            //Nn-valid samps
            if (pad_mode==0)        //zeros
            {
                K += (v>0) ? 2u*N*(Lk-(size_t)v) : 2u*N*Lk;
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    X -= 2u*N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                }
            }
            else if (pad_mode==2)   //reflect
            {
                prev_t = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    t = 2*(int)Li-1-s;  //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { t = (n<0) ? -n-1 : (n<(int)Li) ? t : 2*(int)Li-1-n; }
                    X += 2 * (n-prev_t) * (int)N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
                X += 2 * ((int)Li-prev_t) * (int)N;
            }
            else                    //circular
            {
                prev_t = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    t = s - (int)Li;    //this ensures circular extrapolation to any length
                    while (n>=(int)Li) { t -= (int)Li; }
                    X += 2 * (n-prev_t) * (int)N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
                X += 2 * ((int)Li-prev_t) * (int)N;
            }
            X += 2 * (ss-(int)Li) * (int)N;

            *Y++ = smr; *Y++ = smi;
        }
        K -= Nk; X += 2u*str*N;
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
    if (N<1u) { fprintf(stderr,"error in avgpool1_z: N (num output neurons) must be positive\n"); return 1; }
    if (pad_mode<0 || pad_mode>4) { fprintf(stderr,"error in avgpool1_z: pad_mode must be an int in {0,1,2,3,4}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;     //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in avgpool1_z: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in avgpool1_z: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);

    const size_t Nb = 2u*N;            //total num elements in B
    const size_t Nk = 2u*N*Lk*N;      //total num elements in K
    size_t w=0u;                        //current window (frame)
    double smr, smi;                    //intermediate sums
    int ss=-pad, es=ss+(int)Lk-1;       //current start-samp, end-samp
    int t = 0, prev_t = 0;              //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        for (size_t o=N; o>0u; --o)
        {
            smr = *B++; smi = *B++;

            //Negative samps
            if (pad_mode==0)        //zeros
            {
                K += (es<0) ? 2u*N*Lk : 2u*N*(size_t)(-ss);
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=ss; s<0 && s<=es; ++s, X-=2u*N)
                {
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                }
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    t = -s - 1;         //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { t = (n<0) ? -n-1 : (n<(int)Li) ? t : 2*(int)Li-1-n; }
                    X += 2 * (n-prev_t) * (int)N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
                X -= 2 * prev_t * (int)N;
                prev_t = 0;
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    t = (int)Li + s;    //this ensures circular extrapolation to any length
                    while (n<0) { t += (int)Li; }
                    X += 2 * (n-prev_t) * (int)N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
                X -= 2 * prev_t * (int)N;
                prev_t = 0;
            }

            //Nn-negative samps
            if (es>=0)
            {
                for (int i=(es+1)*(int)N; i>0; --i, X+=2, K+=2)
                {
                    smr += *X**K - *(X+1)**(K+1);
                    smi += *X**(K+1) + *(X+1)**K;
                }
                X -= 2*(es+1)*(int)N;
            }

            *Y++ = smr; *Y++ = smi;
        }
        K -= Nk;
        ss+=str; es+=str; ++w;
    }
    X += 2*ss*(int)N;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t o=N; o>0u; --o)
        {
            smr = *B++; smi = *B++;
            for (size_t i=N*Lk; i>0u; --i, X+=2, K+=2)
            {
                smr += *X**K - *(X+1)**(K+1);
                smi += *X**(K+1) + *(X+1)**K;
            }
            *Y++ = smr; *Y++ = smi;
            X -= 2u*N*Lk;
        }
        K -= Nk; X += 2u*N*str;
        es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int v = (int)Li - ss;

        for (size_t o=N; o>0u; --o)
        {
            smr = *B++; smi = *B++;

            //Valid samps
            if (v>0)
            {
                for (int i=v*(int)N; i>0; --i, X+=2, K+=2)
                {
                    smr += *X**K - *(X+1)**(K+1);
                    smi += *X**(K+1) + *(X+1)**K;
                }
            }
            else { X += 2 * ((int)Li-ss) * (int)N; }

            //Nn-valid samps
            if (pad_mode==0)        //zeros
            {
                K += (v>0) ? 2u*N*(Lk-(size_t)v) : 2u*N*Lk;
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    X -= 2u*N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                }
            }
            else if (pad_mode==2)   //reflect
            {
                prev_t = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    t = 2*(int)Li-1-s;  //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { t = (n<0) ? -n-1 : (n<(int)Li) ? t : 2*(int)Li-1-n; }
                    X += 2 * (n-prev_t) * (int)N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
                X += 2 * ((int)Li-prev_t) * (int)N;
            }
            else                    //circular
            {
                prev_t = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    t = s - (int)Li;    //this ensures circular extrapolation to any length
                    while (n>=(int)Li) { t -= (int)Li; }
                    X += 2 * (n-prev_t) * (int)N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
                X += 2 * ((int)Li-prev_t) * (int)N;
            }
            X += 2 * (ss-(int)Li) * (int)N;

            *Y++ = smr; *Y++ = smi;
        }
        K -= Nk; X += 2u*str*N;
        ss+=str; es+=str; ++w;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
