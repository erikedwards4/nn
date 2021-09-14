//This emulates PyTorch Conv1d, which actually does cross-correlation.

//However, it is considered as a component of the present framework,
//wherein it is an IN (input) method for No neurons,
//where the usual C_out (number of output channels) == No.
//Each output neuron has a bias, so the bias vector is of length No==C_out.

//The usual C_in (number of input channels) == Ni, i.e.
//it is inherited from the preceding layer, just as for any IN method.

//The final parameters are kernel_size == Lk (i.e., kernel width);
//and stride, dilation and padding for the convolution itself.

//This doesn't yet support groups or batch size.
//Batch size (BS) could easily be supported by adding an outer loop,
//but the calling program can easily loop through batch members.
//Also, the Pool functions don't have batch size, but otherwise share the same form.
//Finally, the order in memory here is different convention from PyTorch,
//which uses row-major with size BS x Ni x Li for X, whereas here:

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
//Lo =  ceil[1 + (Li + 2*pad - dil*(Lk-1) - 1)/stride], if ceil_mode is true
//Lo = floor[1 + (Li + 2*pad - dil*(Lk-1) - 1)/stride], if ceil_mode is false [default]
//Y has size No x Lo for col-major.
//Y has size Lo x No for row-major.

//The following params/opts are included:
//Ni:           size_t  num input neurons (leading dim of X)
//No:           size_t  num output neurons (leading dim of Y)
//Li:           size_t  length of input time-series (other dim of X)
//Lk:           size_t  length of kernel in time (num cols of K)
//pad:          int     padding length (pad>0 means add extra samps)
//str:          size_t  stride length (step-size) in samps
//dil:          size_t  dilation factor in samps
//ceil_mode:    bool    calculate Lo with ceil (see above)
//pad_mode:     int     padding mode (0=zeros, 1=replicate, 2=reflect, 3=circular)

#include <stdio.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int conv1d_s (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int conv1d_d (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int conv1d_c (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int conv1d_z (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);


int conv1d_s (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1d_s: str (stride) must be positive\n"); return 1; }
    if (dil<1u) { fprintf(stderr,"error in conv1d_s: dil (dilation) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1d_s: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1d_s: Lk (kernel_size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1d_s: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1d_s: No (num output neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in conv1d_s: pad length must be > -Li\n"); return 1; }
    if (pad_mode<0 || pad_mode>3) { fprintf(stderr,"error in conv1d_s: pad_mode must be an int in {0,1,2,3}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in conv1d_s: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in conv1d_s: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-Tk)/str + (size_t)(ceil_mode && (size_t)(Ti-Tk)%str);
    
    const size_t Nk = Ni*Lk*No;             //total num elements in K
    const size_t jump = Ni*(dil-1u);        //fixed jump due to dilation for X below
    size_t w=0u;                            //current window (frame)
    float sm;                               //intermediate sum
    int ss=-pad, es=ss+Tk-1;                //current start-samp, end-samp
    int t = 0, prev_t = 0;                  //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        for (size_t o=No; o>0u; --o, ++B, ++Y)
        {
            sm = *B;

            //Negative samps
            if (pad_mode==0)        //zeros
            {
                K += (es<0) ? Ni*Lk : Ni*(Lk-1u-(size_t)es/dil);
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=ss; s<0 && s<=es; s+=dil, X-=Ni)
                {
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                }
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss; s<0 && s<=es; s+=dil)
                {
                    t = -s - 1;         //this ensures reflected extrapolation to any length
                    while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                    X += (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_t = t + 1;
                }
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; s+=dil)
                {
                    t = (int)Li + s;    //this ensures circular extrapolation to any length
                    while (t<0) { t += (int)Li; }
                    X += (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_t = t + 1;
                }
            }

            //Non-negative samps
            if (es>=0)
            {
                for (int s=es%(int)dil; s<=es; s+=dil)
                {
                    X += (s-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_t = s + 1;
                }
            }
            X -= prev_t * (int)Ni;
            prev_t = 0;

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
            for (size_t l=Lk; l>0u; --l, X+=jump)
            {
                for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
            }
            *Y = sm;
            X -= Ni*Lk*dil;
        }
        B -= No; K -= Nk; X += Ni*str;
        es+=str; ++w;
    }
    prev_t = ss = es - Tk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int V = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

        for (size_t o=No; o>0u; --o, ++B, ++Y)
        {
            sm = *B;

            //Valid samps
            for (int s=ss; s<(int)Li; s+=dil)
            {
                X += (s-prev_t) * (int)Ni;
                for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                prev_t = s + 1;
            }

            //Non-valid samps
            if (pad_mode==0)        //zeros
            {
                K += Ni*(Lk-(size_t)V);
            }
            else if (pad_mode==1)   //repeat
            {
                X += ((int)Li-1-prev_t) * (int)Ni;
                for (int s=ss+V*(int)dil; s<=es; s+=dil, X-=Ni)
                {
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                }
                prev_t = (int)Li - 1;
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss+V*(int)dil; s<=es; s+=dil)
                {
                    t = 2*(int)Li-1-s;  //this ensures reflected extrapolation to any length
                    while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                    X += (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_t = t + 1;
                }
            }
            else                    //circular
            {
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; s+=dil)
                {
                    t = s - (int)Li;    //this ensures circular extrapolation to any length
                    while (t>=(int)Li) { t -= (int)Li; }
                    X += (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_t = t + 1;
                }
            }

            *Y = sm;
        }
        B -= No; K -= Nk;
        ss+=str; es+=str; ++w;
    }

    return 0;
}


int conv1d_d (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1d_d: str (stride) must be positive\n"); return 1; }
    if (dil<1u) { fprintf(stderr,"error in conv1d_d: dil (dilation) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1d_d: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1d_d: Lk (kernel_size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1d_d: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1d_d: No (num output neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in conv1d_d: pad length must be > -Li\n"); return 1; }
    if (pad_mode<0 || pad_mode>3) { fprintf(stderr,"error in conv1d_d: pad_mode must be an int in {0,1,2,3}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in conv1d_d: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in conv1d_d: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-Tk)/str + (size_t)(ceil_mode && (size_t)(Ti-Tk)%str);

    const size_t Nk = Ni*Lk*No;             //total num elements in K
    const size_t jump = Ni*(dil-1u);        //fixed jump due to dilation for X below
    size_t w=0u;                            //current window (frame)
    double sm;                              //intermediate sum
    int ss=-pad, es=ss+Tk-1;                //current start-samp, end-samp
    int t = 0, prev_t = 0;                  //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        for (size_t o=No; o>0u; --o, ++B, ++Y)
        {
            sm = *B;

            //Negative samps
            if (pad_mode==0)        //zeros
            {
                K += (es<0) ? Ni*Lk : Ni*(Lk-1u-(size_t)es/dil);
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=ss; s<0 && s<=es; s+=dil, X-=Ni)
                {
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                }
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss; s<0 && s<=es; s+=dil)
                {
                    t = -s - 1;         //this ensures reflected extrapolation to any length
                    while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                    X += (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_t = t + 1;
                }
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; s+=dil)
                {
                    t = (int)Li + s;    //this ensures circular extrapolation to any length
                    while (t<0) { t += (int)Li; }
                    X += (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_t = t + 1;
                }
            }

            //Non-negative samps
            if (es>=0)
            {
                for (int s=es%(int)dil; s<=es; s+=dil)
                {
                    X += (s-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_t = s + 1;
                }
            }
            X -= prev_t * (int)Ni;
            prev_t = 0;

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
            for (size_t l=Lk; l>0u; --l, X+=jump)
            {
                for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
            }
            *Y = sm;
            X -= Ni*Lk*dil;
        }
        B -= No; K -= Nk; X += Ni*str;
        es+=str; ++w;
    }
    prev_t = ss = es - Tk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int V = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

        for (size_t o=No; o>0u; --o, ++B, ++Y)
        {
            sm = *B;

            //Valid samps
            for (int s=ss; s<(int)Li; s+=dil)
            {
                X += (s-prev_t) * (int)Ni;
                for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                prev_t = s + 1;
            }

            //Non-valid samps
            if (pad_mode==0)        //zeros
            {
                K += Ni*(Lk-(size_t)V);
            }
            else if (pad_mode==1)   //repeat
            {
                X += ((int)Li-1-prev_t) * (int)Ni;
                for (int s=ss+V*(int)dil; s<=es; s+=dil, X-=Ni)
                {
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                }
                prev_t = (int)Li - 1;
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss+V*(int)dil; s<=es; s+=dil)
                {
                    t = 2*(int)Li-1-s;  //this ensures reflected extrapolation to any length
                    while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                    X += (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_t = t + 1;
                }
            }
            else                    //circular
            {
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; s+=dil)
                {
                    t = s - (int)Li;    //this ensures circular extrapolation to any length
                    while (t>=(int)Li) { t -= (int)Li; }
                    X += (t-prev_t) * (int)Ni;
                    for (size_t i=Ni; i>0u; --i, ++X, ++K) { sm += *X * *K; }
                    prev_t = t + 1;
                }
            }

            *Y = sm;
        }
        B -= No; K -= Nk;
        ss+=str; es+=str; ++w;
    }

    return 0;
}


int conv1d_c (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1d_c: str (stride) must be positive\n"); return 1; }
    if (dil<1u) { fprintf(stderr,"error in conv1d_c: dil (dilation) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1d_c: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1d_c: Lk (kernel_size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1d_c: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1d_c: No (num output neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in conv1d_c: pad length must be > -Li\n"); return 1; }
    if (pad_mode<0 || pad_mode>3) { fprintf(stderr,"error in conv1d_c: pad_mode must be an int in {0,1,2,3}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in conv1d_c: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in conv1d_c: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-Tk)/str + (size_t)(ceil_mode && (size_t)(Ti-Tk)%str);

    const size_t Nb = 2u*No;                //total num elements in B
    const size_t Nk = 2u*Ni*Lk*No;          //total num elements in K
    const size_t jump = 2u*Ni*(dil-1u);     //fixed jump due to dilation for X below
    size_t w=0u;                            //current window (frame)
    float smr, smi;                         //intermediate sums
    int ss=-pad, es=ss+Tk-1;                //current start-samp, end-samp
    int t = 0, prev_t = 0;                  //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        for (size_t o=No; o>0u; --o)
        {
            smr = *B++; smi = *B++;

            //Negative samps
            if (pad_mode==0)        //zeros
            {
                K += (es<0) ? 2u*Ni*Lk : 2u*Ni*(Lk-1u-(size_t)es/dil);
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=ss; s<0 && s<=es; s+=dil, X-=2u*Ni)
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
                for (int s=ss; s<0 && s<=es; s+=dil)
                {
                    t = -s - 1;         //this ensures reflected extrapolation to any length
                    while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                    X += 2*(t-prev_t)*(int)Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; s+=dil)
                {
                    t = (int)Li + s;    //this ensures circular extrapolation to any length
                    while (t<0) { t += (int)Li; }
                    X += 2*(t-prev_t)*(int)Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
            }

            //Non-negative samps
            if (es>=0)
            {
                for (int s=es%(int)dil; s<=es; s+=dil)
                {
                    X += 2*(s-prev_t)*(int)Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = s + 1;
                }
            }
            X -= 2*prev_t*(int)Ni;
            prev_t = 0;

            *Y++ = smr; *Y++ = smi;
        }
        B -= Nb; K -= Nk;
        ss+=str; es+=str; ++w;
    }
    X += ss*(int)Ni;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t o=No; o>0u; --o)
        {
            smr = *B++; smi = *B++;
            for (size_t l=Lk; l>0u; --l, X+=jump)
            {
                for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                {
                    smr += *X**K - *(X+1)**(K+1);
                    smi += *X**(K+1) + *(X+1)**K;
                }
            }
            *Y++ = smr; *Y++ = smi;
            X -= 2u*Ni*Lk*dil;
        }
        B -= Nb; K -= Nk; X += 2u*Ni*str;
        es+=str; ++w;
    }
    prev_t = ss = es - Tk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int V = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

        for (size_t o=No; o>0u; --o)
        {
            smr = *B++; smi = *B++;

            //Valid samps
            for (int s=ss; s<(int)Li; s+=dil)
            {
                X += 2*(s-prev_t)*(int)Ni;
                for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                {
                    smr += *X**K - *(X+1)**(K+1);
                    smi += *X**(K+1) + *(X+1)**K;
                }
                prev_t = s + 1;
            }

            //Non-valid samps
            if (pad_mode==0)        //zeros
            {
                K += 2u*Ni*(Lk-(size_t)V);
            }
            else if (pad_mode==1)   //repeat
            {
                X += 2*((int)Li-1-prev_t)*(int)Ni;
                for (int s=ss+V*(int)dil; s<=es; s+=dil, X-=2u*Ni)
                {
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                }
                prev_t = (int)Li - 1;
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss+V*(int)dil; s<=es; s+=dil)
                {
                    t = 2*(int)Li-1-s;  //this ensures reflected extrapolation to any length
                    while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                    X += 2*(t-prev_t)*(int)Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
            }
            else                    //circular
            {
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; s+=dil)
                {
                    t = s - (int)Li;    //this ensures circular extrapolation to any length
                    while (t>=(int)Li) { t -= (int)Li; }
                    X += 2*(t-prev_t)*(int)Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
            }

            *Y++ = smr; *Y++ = smi;
        }
        B -= Nb; K -= Nk;
        ss+=str; es+=str; ++w;
    }

    return 0;
}


int conv1d_z (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1d_z: str (stride) must be positive\n"); return 1; }
    if (dil<1u) { fprintf(stderr,"error in conv1d_z: dil (dilation) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1d_z: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1d_z: Lk (kernel_size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1d_z: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1d_z: No (num output neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in conv1d_z: pad length must be > -Li\n"); return 1; }
    if (pad_mode<0 || pad_mode>3) { fprintf(stderr,"error in conv1d_z: pad_mode must be an int in {0,1,2,3}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in conv1d_z: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in conv1d_z: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-Tk)/str + (size_t)(ceil_mode && (size_t)(Ti-Tk)%str);

    const size_t Nb = 2u*No;                //total num elements in B
    const size_t Nk = 2u*Ni*Lk*No;          //total num elements in K
    const size_t jump = 2u*Ni*(dil-1u);     //fixed jump due to dilation for X below
    size_t w=0u;                            //current window (frame)
    double smr, smi;                        //intermediate sums
    int ss=-pad, es=ss+Tk-1;                //current start-samp, end-samp
    int t = 0, prev_t = 0;                  //non-negative samps during extrapolation (padding)

    //K before or overlapping first samp of X
    while (ss<0 && w<Lo)
    {
        for (size_t o=No; o>0u; --o)
        {
            smr = *B++; smi = *B++;

            //Negative samps
            if (pad_mode==0)        //zeros
            {
                K += (es<0) ? 2u*Ni*Lk : 2u*Ni*(Lk-1u-(size_t)es/dil);
            }
            else if (pad_mode==1)   //repeat
            {
                for (int s=ss; s<0 && s<=es; s+=dil, X-=2u*Ni)
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
                for (int s=ss; s<0 && s<=es; s+=dil)
                {
                    t = -s - 1;         //this ensures reflected extrapolation to any length
                    while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                    X += 2*(t-prev_t)*(int)Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; s+=dil)
                {
                    t = (int)Li + s;    //this ensures circular extrapolation to any length
                    while (t<0) { t += (int)Li; }
                    X += 2*(t-prev_t)*(int)Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
            }

            //Non-negative samps
            if (es>=0)
            {
                for (int s=es%(int)dil; s<=es; s+=dil)
                {
                    X += 2*(s-prev_t)*(int)Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = s + 1;
                }
            }
            X -= 2*prev_t*(int)Ni;
            prev_t = 0;

            *Y++ = smr; *Y++ = smi;
        }
        B -= Nb; K -= Nk;
        ss+=str; es+=str; ++w;
    }
    X += ss*(int)Ni;

    //K fully within X
    while (es<(int)Li && w<Lo)
    {
        for (size_t o=No; o>0u; --o)
        {
            smr = *B++; smi = *B++;
            for (size_t l=Lk; l>0u; --l, X+=jump)
            {
                for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                {
                    smr += *X**K - *(X+1)**(K+1);
                    smi += *X**(K+1) + *(X+1)**K;
                }
            }
            *Y++ = smr; *Y++ = smi;
            X -= 2u*Ni*Lk*dil;
        }
        B -= Nb; K -= Nk; X += 2u*Ni*str;
        es+=str; ++w;
    }
    prev_t = ss = es - Tk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int V = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

        for (size_t o=No; o>0u; --o)
        {
            smr = *B++; smi = *B++;

            //Valid samps
            for (int s=ss; s<(int)Li; s+=dil)
            {
                X += 2*(s-prev_t)*(int)Ni;
                for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                {
                    smr += *X**K - *(X+1)**(K+1);
                    smi += *X**(K+1) + *(X+1)**K;
                }
                prev_t = s + 1;
            }

            //Non-valid samps
            if (pad_mode==0)        //zeros
            {
                K += 2u*Ni*(Lk-(size_t)V);
            }
            else if (pad_mode==1)   //repeat
            {
                X += 2*((int)Li-1-prev_t)*(int)Ni;
                for (int s=ss+V*(int)dil; s<=es; s+=dil, X-=2u*Ni)
                {
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                }
                prev_t = (int)Li - 1;
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss+V*(int)dil; s<=es; s+=dil)
                {
                    t = 2*(int)Li-1-s;  //this ensures reflected extrapolation to any length
                    while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                    X += 2*(t-prev_t)*(int)Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
            }
            else                    //circular
            {
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; s+=dil)
                {
                    t = s - (int)Li;    //this ensures circular extrapolation to any length
                    while (t>=(int)Li) { t -= (int)Li; }
                    X += 2*(t-prev_t)*(int)Ni;
                    for (size_t i=Ni; i>0u; --i, X+=2, K+=2)
                    {
                        smr += *X**K - *(X+1)**(K+1);
                        smi += *X**(K+1) + *(X+1)**K;
                    }
                    prev_t = t + 1;
                }
            }

            *Y++ = smr; *Y++ = smi;
        }
        B -= Nb; K -= Nk;
        ss+=str; es+=str; ++w;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
