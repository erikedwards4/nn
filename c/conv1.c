//This is just like conv1d, but doesn't allow dilation,
//and doesn't allow non-zero padding greater than Li (signal length).
//This allows shorter and more efficient code.

//X is the input of size Ni x Li,
//where Li==L_in is usually thought of as the number of time points.
//X can be row- or col-major, but Ni==C_in must always be the leading dim, i.e.:
//X has size Ni x Li for col-major.
//X has size Li x Ni for row-major.

//K is the tensor of convolving kernels with size Ni x Lk x No.
//K can be row- or col-major, but Ni==C_in must always be the leading dim, i.e.:
//K has size Ni x Lk x No for col-major.
//K has size No x Lk x Ni for row-major.

//Each vector in Y has length Lo==L_out, set by:
//Lo =  ceil[1 + (Li + 2*pad - Lk)/stride], if ceil_mode is true
//Lo = floor[1 + (Li + 2*pad - Lk)/stride], if ceil_mode is false [default]
//Y has size No x Lo for col-major.
//Y has size Lo x No for row-major.

//The following params/opts are included:
//Ni:           size_t  num input neurons (leading dim of X)
//No:           size_t  num output neurons (leading dim of Y)
//Li:           size_t  length of input time-series (other dim of X)
//Lk:           size_t  length of kernel in time (num cols of K)
//pad:          int     padding length in [1-Li Li] (pad>0 means add extra samps)
//str:          size_t  stride length (step-size) in samps
//ceil_mode:    bool    calculate Lo with ceil (see above)
//pad_mode:     int     padding mode (0=zeros, 1=replicate, 2=reflect, 3=circular)

#include <stdio.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int conv1_s (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int conv1_d (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int conv1_c (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int conv1_z (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);


int conv1_s (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1_s: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1_s: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1_s: Lk (kernel_size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1_s: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1_s: No (num output neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in conv1_s: pad length must be > -Li\n"); return 1; }
    if (pad_mode && pad>(int)Li) { fprintf(stderr,"error in conv1_s: Li (length of input vecs) must be >= pad length\n"); return 1; }
    if (pad_mode<0 || pad_mode>3) { fprintf(stderr,"error in conv1_s: pad_mode must be an int in {0,1,2,3}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;     //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in conv1_s: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in conv1_s: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);
    
    const size_t Nk = Ni*Lk*No;         //total num elements in K
    size_t w=0u;                        //current window (frame)
    float sm;                           //intermediate sum
    int ss=-pad, es=ss+(int)Lk-1;       //current start-samp, end-samp
    int t = 0, prev_t = 0;              //non-negative samps during extrapolation (padding)

    struct timespec tic, toc; clock_gettime(CLOCK_REALTIME,&tic);

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        for (size_t o=No; o>0u; --o, ++B, ++Y)
        {
            sm = *B;

            //Negative samps
            if (pad_mode==0)        //zeros
            {
                K += (es<0) ? Ni*Lk : Ni*(size_t)(-ss);
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=ss; s<0 && s<=es; ++s, X-=Ni)
                {
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                }
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    t = -s;         //PyTorch-style
                    //t = -s - 1;   //Kaldi-style
                    X += (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_t = t + 1;
                }
                X -= prev_t * (int)Ni;
                prev_t = 0;
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    t = (int)Li + s;
                    X += (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_t = t + 1;
                }
                X -= prev_t * (int)Ni;
                prev_t = 0;
            }

            //Non-negative samps
            if (es>=0)
            {
                for (int i=(es+1)*(int)Ni; i>0; --i, ++X, ++K) { sm += *X * *K; }
                X -= (es+1)*(int)Ni;
            }

            *Y = sm;
        }
        B -= No; K -= Nk;
        ss+=str; es+=str; ++w;
    }
    X += ss*(int)Ni;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t o=No; o>0u; --o, ++B, ++Y)
        {
            sm = *B;
            for (size_t i=Ni*Lk; i>0u; --i, ++X, ++K) { sm += *X * *K; }
            *Y = sm;
            X -= Ni*Lk;
        }
        B -= No; K -= Nk; X += Ni*str;
        es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int V = (int)Li - ss;

        for (size_t o=No; o>0u; --o, ++B, ++Y)
        {
            sm = *B;

            //Valid samps
            if (V>0)
            {
                for (int i=V*(int)Ni; i>0; --i, ++X, ++K) { sm += *X * *K; }
            }
            else { X += ((int)Li-ss) * (int)Ni; }

            //Non-valid samps
            if (pad_mode==0)        //zeros
            {
                K += (V>0) ? Ni*(Lk-(size_t)V) : Ni*Lk;
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    X -= Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                }
            }
            else if (pad_mode==2)   //reflect
            {
                prev_t = (int)Li; 
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    t = 2*(int)Li - 2 - s;      //PyTorch-style
                    //t = 2*(int)Li - 1 - s;    //Kaldi-style
                    X += (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_t = t + 1;
                }
                X += ((int)Li-prev_t) * (int)Ni;
            }
            else                    //circular
            {
                prev_t = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    t = s - (int)Li;
                    X += (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_t = t + 1;
                }
                X += ((int)Li-prev_t) * (int)Ni;
            }
            X += (ss-(int)Li) * (int)Ni;

            *Y = sm;
        }
        B -= No; K -= Nk; X += str*Ni;
        ss+=str; es+=str; ++w;
    }

    clock_gettime(CLOCK_REALTIME,&toc);
    fprintf(stderr,"elapsed time = %.6f ms\n",(double)(toc.tv_sec-tic.tv_sec)*1e3+(double)(toc.tv_nsec-tic.tv_nsec)/1e6);

    return 0;
}


int conv1_d (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1_d: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1_d: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1_d: Lk (kernel_size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1_d: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1_d: No (num output neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in conv1_d: pad length must be > -Li\n"); return 1; }
    if (pad_mode && pad>(int)Li) { fprintf(stderr,"error in conv1_d: Li (length of input vecs) must be >= pad length\n"); return 1; }
    if (pad_mode<0 || pad_mode>3) { fprintf(stderr,"error in conv1_d: pad_mode must be an int in {0,1,2,3}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;     //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in conv1_d: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in conv1_d: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);
    
    const size_t Nk = Ni*Lk*No;         //total num elements in K
    size_t w=0u;                        //current window (frame)
    double sm;                          //intermediate sum
    int ss=-pad, es=ss+(int)Lk-1;       //current start-samp, end-samp
    int t = 0, prev_t = 0;              //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        for (size_t o=No; o>0u; --o, ++B, ++Y)
        {
            sm = *B;

            //Negative samps
            if (pad_mode==0)        //zeros
            {
                K += (es<0) ? Ni*Lk : Ni*(size_t)(-ss);
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=ss; s<0 && s<=es; ++s, X-=Ni)
                {
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                }
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    t = -s;         //PyTorch-style
                    //t = -s - 1;   //Kaldi-style
                    X += (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_t = t + 1;
                }
                X -= prev_t * (int)Ni;
                prev_t = 0;
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    t = (int)Li + s;
                    X += (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_t = t + 1;
                }
                X -= prev_t * (int)Ni;
                prev_t = 0;
            }

            //Non-negative samps
            if (es>=0)
            {
                for (int i=(es+1)*(int)Ni; i>0; --i, ++X, ++K) { sm += *X * *K; }
                X -= (es+1)*(int)Ni;
            }

            *Y = sm;
        }
        B -= No; K -= Nk;
        ss+=str; es+=str; ++w;
    }
    X += ss*(int)Ni;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t o=No; o>0u; --o, ++B, ++Y)
        {
            sm = *B;
            for (size_t i=Ni*Lk; i>0u; --i, ++X, ++K) { sm += *X * *K; }
            *Y = sm;
            X -= Ni*Lk;
        }
        B -= No; K -= Nk; X += Ni*str;
        es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int V = (int)Li - ss;

        for (size_t o=No; o>0u; --o, ++B, ++Y)
        {
            sm = *B;

            //Valid samps
            if (V>0)
            {
                for (int i=V*(int)Ni; i>0; --i, ++X, ++K) { sm += *X * *K; }
            }
            else { X += ((int)Li-ss) * (int)Ni; }

            //Non-valid samps
            if (pad_mode==0)        //zeros
            {
                K += (V>0) ? Ni*(Lk-(size_t)V) : Ni*Lk;
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    X -= Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                }
            }
            else if (pad_mode==2)   //reflect
            {
                prev_t = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    t = 2*(int)Li - 2 - s;      //PyTorch-style
                    //t = 2*(int)Li - 1 - s;    //Kaldi-style
                    X += (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_t = t + 1;
                }
                X += ((int)Li-prev_t) * (int)Ni;
            }
            else                    //circular
            {
                prev_t = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    t = s - (int)Li;
                    X += (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_t = t + 1;
                }
                X += ((int)Li-prev_t) * (int)Ni;
            }
            X += (ss-(int)Li) * (int)Ni;

            *Y = sm;
        }
        B -= No; K -= Nk; X += str*Ni;
        ss+=str; es+=str; ++w;
    }

    return 0;
}


int conv1_c (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1_c: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1_c: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1_c: Lk (kernel_size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1_c: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1_c: No (num output neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in conv1_c: pad length must be > -Li\n"); return 1; }
    if (pad_mode && pad>(int)Li) { fprintf(stderr,"error in conv1_c: Li (length of input vecs) must be >= pad length\n"); return 1; }
    if (pad_mode<0 || pad_mode>3) { fprintf(stderr,"error in conv1_c: pad_mode must be an int in {0,1,2,3}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;     //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in conv1_c: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in conv1_c: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);

    const size_t Nb = 2u*No;            //total num elements in B
    const size_t Nk = 2u*Ni*Lk*No;      //total num elements in K
    size_t w=0u;                        //current window (frame)
    float smr, smi;                     //intermediate sums
    int ss=-pad, es=ss+(int)Lk-1;       //current start-samp, end-samp
    int t = 0, prev_t = 0;              //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        for (size_t o=No; o>0u; --o)
        {
            smr = *B++; smi = *B++;

            //Negative samps
            if (pad_mode==0)        //zeros
            {
                K += (es<0) ? 2u*Ni*Lk : 2u*Ni*(size_t)(-ss);
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=ss; s<0 && s<=es; ++s, X-=2u*Ni)
                {
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
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
                    t = -s;         //PyTorch-style
                    //t = -s - 1;   //Kaldi-style
                    X += 2 * (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
                X -= 2 * prev_t * (int)Ni;
                prev_t = 0;
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    t = (int)Li + s;
                    X += 2 * (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
                X -= 2 * prev_t * (int)Ni;
                prev_t = 0;
            }

            //Non-negative samps
            if (es>=0)
            {
                for (int i=(es+1)*(int)Ni; i>0; --i, X+=2, K+=2)
                {
                    smr += *X**K - *(X+1)**(K+1);
                    smi += *X**(K+1) + *(X+1)**K;
                }
                X -= 2*(es+1)*(int)Ni;
            }

            *Y++ = smr; *Y++ = smi;
        }
        B -= Nb; K -= Nk;
        ss+=str; es+=str; ++w;
    }
    X += 2*ss*(int)Ni;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t o=No; o>0u; --o)
        {
            smr = *B++; smi = *B++;
            for (size_t i=Ni*Lk; i>0u; --i, X+=2, K+=2)
            {
                smr += *X**K - *(X+1)**(K+1);
                smi += *X**(K+1) + *(X+1)**K;
            }
            *Y++ = smr; *Y++ = smi;
            X -= 2u*Ni*Lk;
        }
        B -= Nb; K -= Nk; X += 2u*Ni*str;
        es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int V = (int)Li - ss;

        for (size_t o=No; o>0u; --o)
        {
            smr = *B++; smi = *B++;

            //Valid samps
            if (V>0)
            {
                for (int i=V*(int)Ni; i>0; --i, X+=2, K+=2)
                {
                    smr += *X**K - *(X+1)**(K+1);
                    smi += *X**(K+1) + *(X+1)**K;
                }
            }
            else { X += 2 * ((int)Li-ss) * (int)Ni; }

            //Non-valid samps
            if (pad_mode==0)        //zeros
            {
                K += (V>0) ? 2u*Ni*(Lk-(size_t)V) : 2u*Ni*Lk;
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    X -= 2u*Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
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
                    t = 2*(int)Li - 2 - s;      //PyTorch-style
                    //t = 2*(int)Li - 1 - s;    //Kaldi-style
                    X += 2 * (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
                X += 2 * ((int)Li-prev_t) * (int)Ni;
            }
            else                    //circular
            {
                prev_t = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    t = s - (int)Li;
                    X += 2 * (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
                X += 2 * ((int)Li-prev_t) * (int)Ni;
            }
            X += 2 * (ss-(int)Li) * (int)Ni;

            *Y++ = smr; *Y++ = smi;
        }
        B -= Nb; K -= Nk; X += 2u*str*Ni;
        ss+=str; es+=str; ++w;
    }

    return 0;
}


int conv1_z (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1_z: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1_z: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1_z: Lk (kernel_size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1_z: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1_z: No (num output neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in conv1_z: pad length must be > -Li\n"); return 1; }
    if (pad_mode && pad>(int)Li) { fprintf(stderr,"error in conv1_z: Li (length of input vecs) must be >= pad length\n"); return 1; }
    if (pad_mode<0 || pad_mode>3) { fprintf(stderr,"error in conv1_z: pad_mode must be an int in {0,1,2,3}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;     //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in conv1_z: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in conv1_z: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);

    const size_t Nb = 2u*No;            //total num elements in B
    const size_t Nk = 2u*Ni*Lk*No;      //total num elements in K
    size_t w=0u;                        //current window (frame)
    double smr, smi;                    //intermediate sums
    int ss=-pad, es=ss+(int)Lk-1;       //current start-samp, end-samp
    int t = 0, prev_t = 0;              //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        for (size_t o=No; o>0u; --o)
        {
            smr = *B++; smi = *B++;

            //Negative samps
            if (pad_mode==0)        //zeros
            {
                K += (es<0) ? 2u*Ni*Lk : 2u*Ni*(size_t)(-ss);
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=ss; s<0 && s<=es; ++s, X-=2u*Ni)
                {
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
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
                    t = -s;         //PyTorch-style
                    //t = -s - 1;   //Kaldi-style
                    X += 2 * (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
                X -= 2 * prev_t * (int)Ni;
                prev_t = 0;
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    t = (int)Li + s;
                    X += 2 * (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
                X -= 2 * prev_t * (int)Ni;
                prev_t = 0;
            }

            //Non-negative samps
            if (es>=0)
            {
                for (int i=(es+1)*(int)Ni; i>0; --i, X+=2, K+=2)
                {
                    smr += *X**K - *(X+1)**(K+1);
                    smi += *X**(K+1) + *(X+1)**K;
                }
                X -= 2*(es+1)*(int)Ni;
            }

            *Y++ = smr; *Y++ = smi;
        }
        B -= Nb; K -= Nk;
        ss+=str; es+=str; ++w;
    }
    X += 2*ss*(int)Ni;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t o=No; o>0u; --o)
        {
            smr = *B++; smi = *B++;
            for (size_t i=Ni*Lk; i>0u; --i, X+=2, K+=2)
            {
                smr += *X**K - *(X+1)**(K+1);
                smi += *X**(K+1) + *(X+1)**K;
            }
            *Y++ = smr; *Y++ = smi;
            X -= 2u*Ni*Lk;
        }
        B -= Nb; K -= Nk; X += 2u*Ni*str;
        es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int V = (int)Li - ss;

        for (size_t o=No; o>0u; --o)
        {
            smr = *B++; smi = *B++;

            //Valid samps
            if (V>0)
            {
                for (int i=V*(int)Ni; i>0; --i, X+=2, K+=2)
                {
                    smr += *X**K - *(X+1)**(K+1);
                    smi += *X**(K+1) + *(X+1)**K;
                }
            }
            else { X += 2 * ((int)Li-ss) * (int)Ni; }

            //Non-valid samps
            if (pad_mode==0)        //zeros
            {
                K += (V>0) ? 2u*Ni*(Lk-(size_t)V) : 2u*Ni*Lk;
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    X -= 2u*Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
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
                    t = 2*(int)Li - 2 - s;      //PyTorch-style
                    //t = 2*(int)Li - 1 - s;    //Kaldi-style
                    X += 2 * (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
                X += 2 * ((int)Li-prev_t) * (int)Ni;
            }
            else                    //circular
            {
                prev_t = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    t = s - (int)Li;
                    X += 2 * (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
                X += 2 * ((int)Li-prev_t) * (int)Ni;
            }
            X += 2 * (ss-(int)Li) * (int)Ni;

            *Y++ = smr; *Y++ = smi;
        }
        B -= Nb; K -= Nk; X += 2u*str*Ni;
        ss+=str; es+=str; ++w;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
