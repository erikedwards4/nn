//This does average pooling, like PyTorch AvgPool1d,
//but uses the full set of input params/opts available for conv1d.

//Considered as a component of the present framework,
//this is an IN (input) method for N==N neurons.
//In PyTorch, N and N are called C, i.e. num chans.

//The parameters are kernel_size == Lk (i.e., kernel width);
//and stride, dilation and padding for the avgpooling itself.
//Note that no kernel is input; it is implicitly just ones.

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
//N:            size_t  num input and output neurons (leading dim of X and Y)
//Li:           size_t  length of input time-series (other dim of X)
//Lk:           size_t  length of kernel in time (num cols of K)
//pad:          int     padding length (pad>0 means add extra samps)
//str:          size_t  stride length (step-size) in samps
//dil:          size_t  dilation factor in samps
//ceil_mode:    bool    calculate Lo with ceil (see above)
//pad_mode:     int     padding mode (0=zeros, 1=replicate, 2=reflect, 3=circular, 4=no_count_pad)

#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int avgpool1d_s (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int avgpool1d_d (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int avgpool1d_c (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int avgpool1d_z (double *Y, const double *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);


int avgpool1d_s (float *Y, const float *X, const size_t N, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in avgpool1d_s: str (stride) must be positive\n"); return 1; }
    if (dil<1u) { fprintf(stderr,"error in avgpool1d_s: dil (dilation) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in avgpool1d_s: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in avgpool1d_s: Lk (kernel_size) must be positive\n"); return 1; }
    if (N<1u) { fprintf(stderr,"error in avgpool1d_s: N (num input neurons) must be positive\n"); return 1; }
    if (pad_mode<0 || pad_mode>4) { fprintf(stderr,"error in avgpool1d_s: pad_mode must be an int in {0,1,2,3,4}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in avgpool1d_s: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in avgpool1d_s: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-Tk)/str + (size_t)(ceil_mode && (size_t)(Ti-Tk)%str);
    
    const size_t jump = N*(dil-1u);         //fixed jump due to dilation for X below
    size_t w=0u;                            //current window (frame)
    float sm;                               //intermediate sum
    int ss=-pad, es=ss+(int)Tk-1;           //current start-samp, end-samp
    int n = 0, prev_n = 0;                  //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        for (size_t o=N; o>0u; --o, ++Y)
        {
            sm = 0.0f;

            //Negative samps
            if (pad_mode==0)        //zeros
            {
                K += (es<0) ? N*Lk : N*(Lk-1u-(size_t)es/dil);
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=ss; s<0 && s<=es; s+=dil, X-=N)
                {
                    for (size_t i=N; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                }
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss; s<0 && s<=es; s+=dil)
                {
                    n = -s - 1;         //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { n = (n<0) ? -n-1 : (n<(int)Li) ? n : 2*(int)Li-1-n; }
                    X += (n-prev_n) * (int)N;
                    for (size_t i=N; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_n = n + 1;
                }
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; s+=dil)
                {
                    n = (int)Li + s;    //this ensures circular extrapolation to any length
                    while (n<0) { n += (int)Li; }
                    X += (n-prev_n) * (int)N;
                    for (size_t i=N; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_n = n + 1;
                }
            }

            //Nn-negative samps
            if (es>=0)
            {
                for (int s=es%(int)dil; s<=es; s+=dil)
                {
                    X += (s-prev_n) * (int)N;
                    for (size_t i=N; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_n = s + 1;
                }
            }
            X -= prev_n * (int)N;
            prev_n = 0;

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
            sm = 0.0f;
            for (size_t l=Lk; l>0u; --l, X+=jump)
            {
                for (size_t i=N; i>0u; --i, ++X, ++K) { sm += *X * *K; }
            }
            *Y = sm;
            X -= N*Lk*dil;
        }
        K -= Nk; X += N*str;
        es+=str; ++w;
    }
    prev_n = ss = es - Tk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int v = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

        for (size_t o=N; o>0u; --o, ++Y)
        {
            sm = 0.0f;

            //Valid samps
            for (int s=ss; s<(int)Li; s+=dil)
            {
                X += (s-prev_n) * (int)N;
                for (size_t i=N; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                prev_n = s + 1;
            }

            //Nn-valid samps
            if (pad_mode==0)        //zeros
            {
                K += N*(Lk-(size_t)v);
            }
            else if (pad_mode==1)   //repeat
            {
                X += ((int)Li-1-prev_n) * N;
                for (int s=ss+v*(int)dil; s<=es; s+=dil, X-=N)
                {
                    for (size_t i=N; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                }
                prev_n = (int)Li - 1;
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss+v*(int)dil; s<=es; s+=dil)
                {
                    n = 2*(int)Li-1-s;  //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { n = (n<0) ? -n-1 : (n<(int)Li) ? n : 2*(int)Li-1-n; }
                    X += (n-prev_n) * (int)N;
                    for (size_t i=N; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_n = n + 1;
                }
            }
            else                    //circular
            {
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; s+=dil)
                {
                    n = s - (int)Li;    //this ensures circular extrapolation to any length
                    while (n>=(int)Li) { n -= (int)Li; }
                    X += (n-prev_n) * (int)N;
                    for (size_t i=N; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_n = n + 1;
                }
            }

            *Y = sm;
        }
        K -= Nk;
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
    if (N<1u) { fprintf(stderr,"error in avgpool1d_d: N (num output neurons) must be positive\n"); return 1; }
    if (pad_mode<0 || pad_mode>4) { fprintf(stderr,"error in avgpool1d_d: pad_mode must be an int in {0,1,2,3,4}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in avgpool1d_d: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in avgpool1d_d: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-Tk)/str + (size_t)(ceil_mode && (size_t)(Ti-Tk)%str);

    const size_t Nk = N*Lk*N;             //total num elements in K
    const size_t jump = N*(dil-1u);        //fixed jump due to dilation for X below
    size_t w=0u;                            //current window (frame)
    double sm;                              //intermediate sum
    int ss=-pad, es=ss+(int)Tk-1;           //current start-samp, end-samp
    int n = 0, prev_n = 0;                  //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        for (size_t o=N; o>0u; --o, ++Y)
        {
            sm = 0.0;

            //Negative samps
            if (pad_mode==0)        //zeros
            {
                K += (es<0) ? N*Lk : N*(Lk-1u-(size_t)es/dil);
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=ss; s<0 && s<=es; s+=dil, X-=N)
                {
                    for (size_t i=N; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                }
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss; s<0 && s<=es; s+=dil)
                {
                    n = -s - 1;         //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { n = (n<0) ? -n-1 : (n<(int)Li) ? n : 2*(int)Li-1-n; }
                    X += (n-prev_n) * (int)N;
                    for (size_t i=N; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_n = n + 1;
                }
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; s+=dil)
                {
                    n = (int)Li + s;    //this ensures circular extrapolation to any length
                    while (n<0) { n += (int)Li; }
                    X += (n-prev_n) * (int)N;
                    for (size_t i=N; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_n = n + 1;
                }
            }

            //Nn-negative samps
            if (es>=0)
            {
                for (int s=es%(int)dil; s<=es; s+=dil)
                {
                    X += (s-prev_n) * (int)N;
                    for (size_t i=N; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_n = s + 1;
                }
            }
            X -= prev_n * (int)N;
            prev_n = 0;

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
            for (size_t l=Lk; l>0u; --l, X+=jump)
            {
                for (size_t i=N; i>0u; --i, ++X, ++K) { sm += *X * *K; }
            }
            *Y = sm;
            X -= N*Lk*dil;
        }
        K -= Nk; X += N*str;
        es+=str; ++w;
    }
    prev_n = ss = es - Tk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int v = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

        for (size_t o=N; o>0u; --o, ++Y)
        {
            sm = 0.0;

            //Valid samps
            for (int s=ss; s<(int)Li; s+=dil)
            {
                X += (s-prev_n) * (int)N;
                for (size_t i=N; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                prev_n = s + 1;
            }

            //Nn-valid samps
            if (pad_mode==0)        //zeros
            {
                K += N*(Lk-(size_t)v);
            }
            else if (pad_mode==1)   //repeat
            {
                X += ((int)Li-1-prev_n) * N;
                for (int s=ss+v*(int)dil; s<=es; s+=dil, X-=N)
                {
                    for (size_t i=N; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                }
                prev_n = (int)Li - 1;
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss+v*(int)dil; s<=es; s+=dil)
                {
                    n = 2*(int)Li-1-s;  //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { n = (n<0) ? -n-1 : (n<(int)Li) ? n : 2*(int)Li-1-n; }
                    X += (n-prev_n) * (int)N;
                    for (size_t i=N; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_n = n + 1;
                }
            }
            else                    //circular
            {
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; s+=dil)
                {
                    n = s - (int)Li;    //this ensures circular extrapolation to any length
                    while (n>=(int)Li) { n -= (int)Li; }
                    X += (n-prev_n) * (int)N;
                    for (size_t i=N; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_n = n + 1;
                }
            }

            *Y = sm;
        }
        K -= Nk;
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
    if (N<1u) { fprintf(stderr,"error in avgpool1d_c: N (num output neurons) must be positive\n"); return 1; }
    if (pad_mode<0 || pad_mode>4) { fprintf(stderr,"error in avgpool1d_c: pad_mode must be an int in {0,1,2,3,4}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in avgpool1d_c: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in avgpool1d_c: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-Tk)/str + (size_t)(ceil_mode && (size_t)(Ti-Tk)%str);

    const size_t Nb = 2u*N;                //total num elements in B
    const size_t Nk = 2u*N*Lk*N;          //total num elements in K
    const size_t jump = 2u*N*(dil-1u);     //fixed jump due to dilation for X below
    size_t w=0u;                            //current window (frame)
    float smr, smi;                         //intermediate sums
    int ss=-pad, es=ss+(int)Tk-1;           //current start-samp, end-samp
    int n = 0, prev_n = 0;                  //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        for (size_t o=N; o>0u; --o)
        {
            smr = *B++; smi = *B++;

            //Negative samps
            if (pad_mode==0)        //zeros
            {
                K += (es<0) ? 2u*N*Lk : 2u*N*(Lk-1u-(size_t)es/dil);
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=ss; s<0 && s<=es; s+=dil, X-=2u*N)
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
                for (int s=ss; s<0 && s<=es; s+=dil)
                {
                    n = -s - 1;         //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { n = (n<0) ? -n-1 : (n<(int)Li) ? n : 2*(int)Li-1-n; }
                    X += 2*(n-prev_n)*(int)N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_n = n + 1;
                }
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; s+=dil)
                {
                    n = (int)Li + s;    //this ensures circular extrapolation to any length
                    while (n<0) { n += (int)Li; }
                    X += 2*(n-prev_n)*(int)N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_n = n + 1;
                }
            }

            //Nn-negative samps
            if (es>=0)
            {
                for (int s=es%(int)dil; s<=es; s+=dil)
                {
                    X += 2*(s-prev_n)*(int)N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_n = s + 1;
                }
            }
            X -= 2*prev_n*(int)N;
            prev_n = 0;

            *Y++ = smr; *Y++ = smi;
        }
        K -= Nk;
        ss+=str; es+=str; ++w;
    }
    X += ss*(int)N;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t o=N; o>0u; --o)
        {
            smr = *B++; smi = *B++;
            for (size_t l=Lk; l>0u; --l, X+=jump)
            {
                for (size_t i=N; i>0u; --i, X+=2, K+=2)
                {
                    smr += *X**K - *(X+1)**(K+1);
                    smi += *X**(K+1) + *(X+1)**K;
                }
            }
            *Y++ = smr; *Y++ = smi;
            X -= 2u*N*Lk*dil;
        }
        K -= Nk; X += 2u*N*str;
        es+=str; ++w;
    }
    prev_n = ss = es - Tk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int v = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

        for (size_t o=N; o>0u; --o)
        {
            smr = *B++; smi = *B++;

            //Valid samps
            for (int s=ss; s<(int)Li; s+=dil)
            {
                X += 2*(s-prev_n)*(int)N;
                for (size_t i=N; i>0u; --i, X+=2, K+=2)
                {
                    smr += *X**K - *(X+1)**(K+1);
                    smi += *X**(K+1) + *(X+1)**K;
                }
                prev_n = s + 1;
            }

            //Nn-valid samps
            if (pad_mode==0)        //zeros
            {
                K += 2u*N*(Lk-(size_t)v);
            }
            else if (pad_mode==1)   //repeat
            {
                X += 2*((int)Li-1-prev_n)*N;
                for (int s=ss+v*(int)dil; s<=es; s+=dil, X-=2u*N)
                {
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                }
                prev_n = (int)Li - 1;
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss+v*(int)dil; s<=es; s+=dil)
                {
                    n = 2*(int)Li-1-s;  //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { n = (n<0) ? -n-1 : (n<(int)Li) ? n : 2*(int)Li-1-n; }
                    X += 2*(n-prev_n)*(int)N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_n = n + 1;
                }
            }
            else                    //circular
            {
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; s+=dil)
                {
                    n = s - (int)Li;    //this ensures circular extrapolation to any length
                    while (n>=(int)Li) { n -= (int)Li; }
                    X += 2*(n-prev_n)*(int)N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_n = n + 1;
                }
            }

            *Y++ = smr; *Y++ = smi;
        }
        K -= Nk;
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
    if (N<1u) { fprintf(stderr,"error in avgpool1d_z: N (num output neurons) must be positive\n"); return 1; }
    if (pad_mode<0 || pad_mode>4) { fprintf(stderr,"error in avgpool1d_z: pad_mode must be an int in {0,1,2,3,4}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in avgpool1d_z: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in avgpool1d_z: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-Tk)/str + (size_t)(ceil_mode && (size_t)(Ti-Tk)%str);

    const size_t Nb = 2u*N;                //total num elements in B
    const size_t Nk = 2u*N*Lk*N;          //total num elements in K
    const size_t jump = 2u*N*(dil-1u);     //fixed jump due to dilation for X below
    size_t w=0u;                            //current window (frame)
    double smr, smi;                        //intermediate sums
    int ss=-pad, es=ss+(int)Tk-1;           //current start-samp, end-samp
    int n = 0, prev_n = 0;                  //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        for (size_t o=N; o>0u; --o)
        {
            smr = *B++; smi = *B++;

            //Negative samps
            if (pad_mode==0)        //zeros
            {
                K += (es<0) ? 2u*N*Lk : 2u*N*(Lk-1u-(size_t)es/dil);
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=ss; s<0 && s<=es; s+=dil, X-=2u*N)
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
                for (int s=ss; s<0 && s<=es; s+=dil)
                {
                    n = -s - 1;         //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { n = (n<0) ? -n-1 : (n<(int)Li) ? n : 2*(int)Li-1-n; }
                    X += 2*(n-prev_n)*(int)N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_n = n + 1;
                }
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; s+=dil)
                {
                    n = (int)Li + s;    //this ensures circular extrapolation to any length
                    while (n<0) { n += (int)Li; }
                    X += 2*(n-prev_n)*(int)N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_n = n + 1;
                }
            }

            //Nn-negative samps
            if (es>=0)
            {
                for (int s=es%(int)dil; s<=es; s+=dil)
                {
                    X += 2*(s-prev_n)*(int)N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_n = s + 1;
                }
            }
            X -= 2*prev_n*(int)N;
            prev_n = 0;

            *Y++ = smr; *Y++ = smi;
        }
        K -= Nk;
        ss+=str; es+=str; ++w;
    }
    X += ss*(int)N;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t o=N; o>0u; --o)
        {
            smr = *B++; smi = *B++;
            for (size_t l=Lk; l>0u; --l, X+=jump)
            {
                for (size_t i=N; i>0u; --i, X+=2, K+=2)
                {
                    smr += *X**K - *(X+1)**(K+1);
                    smi += *X**(K+1) + *(X+1)**K;
                }
            }
            *Y++ = smr; *Y++ = smi;
            X -= 2u*N*Lk*dil;
        }
        K -= Nk; X += 2u*N*str;
        es+=str; ++w;
    }
    prev_n = ss = es - Tk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int v = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

        for (size_t o=N; o>0u; --o)
        {
            smr = *B++; smi = *B++;

            //Valid samps
            for (int s=ss; s<(int)Li; s+=dil)
            {
                X += 2*(s-prev_n)*(int)N;
                for (size_t i=N; i>0u; --i, X+=2, K+=2)
                {
                    smr += *X**K - *(X+1)**(K+1);
                    smi += *X**(K+1) + *(X+1)**K;
                }
                prev_n = s + 1;
            }

            //Nn-valid samps
            if (pad_mode==0)        //zeros
            {
                K += 2u*N*(Lk-(size_t)v);
            }
            else if (pad_mode==1)   //repeat
            {
                X += 2*((int)Li-1-prev_n)*N;
                for (int s=ss+v*(int)dil; s<=es; s+=dil, X-=2u*N)
                {
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                }
                prev_n = (int)Li - 1;
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss+v*(int)dil; s<=es; s+=dil)
                {
                    n = 2*(int)Li-1-s;  //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { n = (n<0) ? -n-1 : (n<(int)Li) ? n : 2*(int)Li-1-n; }
                    X += 2*(n-prev_n)*(int)N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_n = n + 1;
                }
            }
            else                    //circular
            {
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; s+=dil)
                {
                    n = s - (int)Li;    //this ensures circular extrapolation to any length
                    while (n>=(int)Li) { n -= (int)Li; }
                    X += 2*(n-prev_n)*(int)N;
                    for (size_t i=N; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_n = n + 1;
                }
            }

            *Y++ = smr; *Y++ = smi;
        }
        K -= Nk;
        ss+=str; es+=str; ++w;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
