//This emulates PyTorch Conv1d, which actually does cross-correlation.

//However, it is considered as a component of the present framework,
//wherein it is an IN (input) method for No neurons,
//where the usual C_out (number of output channels) == No.
//Each output neuron has a bias, so the bias vector is of length No==C_out.

//The usual C_in (number of input channels) == Ni, i.e.
//it is inherited from the preceding layer, just as for any IN method.

//The final parameters are kernel_size == Lk (i.e., kernel width);
//and stride, dilation and padding for the convolution itself.
//This doesn't yet support groups.

//X can be row- or col-major,
//but Ni==C_in must always be the leading dim.
//and Nb (batch size) must be the trailing dim, i.e.:
//X has size Ni x Li x Nb for col-major.
//X has size Nb x Li x Ni for row-major.
//where: Li==L_in is usually thought of as the number of time points.

//K is the tensor of convolving kernels with size Ni x Lk x No.
//K can be row- or col-major, but Ni==C_in must always be the leading dim, i.e.:
//K has size Ni x Lk x No for col-major.
//K has size No x Lk x Ni for row-major.

//Each vector in Y has length Lo==L_out, set by:
//Lo = floor[1 + (Li + 2*pad - dil*(Lk-1) - 1)/stride].
//Y has size No x Lo x Nb for col-major.
//Y has size Nb x Lo x No for row-major.

//The following params/opts are included:
//Ni:           size_t  num input neurons (leading dim of X)
//No:           size_t  num output neurons (leading dim of Y)
//Nb:           size_t  batch size (trailing dim of X and Y)
//Li:           size_t  length of input time-series (other dim of X)
//Lk:           size_t  length of kernel in time (num cols of K)
//pad:          int     padding length (pad>0 means add extra samps)
//str:          size_t  stride length (step-size) in samps
//dil:          size_t  dilation factor in samps
//ceil_mode:    bool    calculate Lo with ceil (see above)
//pad_mode:     int     padding mode (0=zeros, 1=replicate, 2=reflect, 3=circular)

#include <stdio.h>
#include <cblas.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int conv1d_cblas_s (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1d_cblas_s: str (stride) must be positive\n"); return 1; }
    if (dil<1u) { fprintf(stderr,"error in conv1d_cblas_s: dil (dilation) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1d_cblas_s: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1d_cblas_s: Lk (kernel_size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1d_cblas_s: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1d_cblas_s: No (num output neurons) must be positive\n"); return 1; }
    if (Nb<1u) { fprintf(stderr,"error in conv1d_cblas_s: Nb (batch size) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in conv1d_cblas_s: pad length must be > -Li\n"); return 1; }
    if (pad_mode<0 || pad_mode>3) { fprintf(stderr,"error in conv1d_cblas_s: pad_mode must be an int in {0,1,2,3}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in conv1d_cblas_s: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in conv1d_cblas_s: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-Tk)/str + (size_t)(ceil_mode && (size_t)(Ti-Tk)%str);

    const size_t Nk = Ni*Lk*No;             //total num elements in K
    const size_t jump = Ni*dil;             //fixed jump due to dilation for X below
    size_t l;                               //current samp within Y in [0 Lo-1]
    float sm;                               //intermediate sum
    int ss, es;                             //current start-samp, end-samp
    int t, prev_t;                          //non-negative samps during extrapolation (padding)

    //Loop over batch members
    for (size_t b=Nb; b>0u; --b)
    {
        //Reset samp counts
        l = 0u; t = prev_t = 0;
        ss = -pad; es = ss + Tk - 1;

        //K before or overlapping first samp of X
        while (ss<0 && l<Lo)
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
                    for (int s=ss; s<0 && s<=es; s+=dil, K+=Ni)
                    {
                        sm += cblas_sdot((int)Ni,X,1,K,1);
                    }
                }
                else if (pad_mode==2)   //reflect
                {
                    for (int s=ss; s<0 && s<=es; s+=dil)
                    {
                        t = -s;         //PyTorch-style
                        //t = -s - 1;   //Kaldi-style
                        //this ensures reflected extrapolation to any length
                        while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                        X += (t-prev_t) * (int)Ni;
                        sm += cblas_sdot((int)Ni,X,1,K,1);
                        K += Ni; prev_t = t;
                    }
                    X -= prev_t * (int)Ni;
                    prev_t = 0;
                }
                else                    //circular
                {
                    for (int s=ss; s<0 && s<=es; s+=dil)
                    {
                        t = (int)Li + s;    //this ensures circular extrapolation to any length
                        while (t<0) { t += (int)Li; }
                        X += (t-prev_t) * (int)Ni;
                        sm += cblas_sdot((int)Ni,X,1,K,1);
                        K += Ni; prev_t = t;
                    }
                    X -= prev_t * (int)Ni;
                    prev_t = 0;
                }

                //Non-negative samps
                if (es>=0)
                {
                    for (int s=es%(int)dil; s<=es; s+=dil)
                    {
                        X += (s-prev_t) * (int)Ni;
                        sm += cblas_sdot((int)Ni,X,1,K,1);
                        K += Ni; prev_t = s;
                    }
                }
                X -= prev_t * (int)Ni;
                prev_t = 0;

                *Y = sm;
            }
            B -= No; K -= Nk;
            ss += str; es += str; ++l;
        }
        X += ss*(int)Ni;

        //K fully within X
        while (es<(int)Li && l<Lo)
        {
            for (size_t o=No; o>0u; --o, ++B, ++Y)
            {
                sm = *B;
                for (size_t l=Lk; l>0u; --l, X+=jump, K+=Ni)
                {
                    sm += cblas_sdot((int)Ni,X,1,K,1);
                }
                *Y = sm;
                X -= Lk*jump;
            }
            B -= No; K -= Nk; X += Ni*str;
            es += str; ++l;
        }
        prev_t = ss = es - Tk + 1;

        //K past or overlapping last samp of X
        while (l<Lo)
        {
            //Get num valid samps
            const int V = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

            for (size_t o=No; o>0u; --o, ++B, ++Y)
            {
                sm = *B;

                //Valid samps
                for (int s=ss; s<(int)Li; s+=dil)
                {
                    X += (s-prev_t) * (int)Ni;
                    sm += cblas_sdot((int)Ni,X,1,K,1);
                    K += Ni; prev_t = s;
                }

                //Non-valid samps
                if (pad_mode==0)        //zeros
                {
                    K += Ni*(Lk-(size_t)V);
                }
                else if (pad_mode==1)   //repeat
                {
                    X += ((int)Li-1-prev_t) * (int)Ni;
                    for (int s=ss+V*(int)dil; s<=es; s+=dil, K+=Ni)
                    {
                        sm += cblas_sdot((int)Ni,X,1,K,1);
                    }
                    prev_t = (int)Li - 1;
                }
                else if (pad_mode==2)   //reflect
                {
                    for (int s=ss+V*(int)dil; s<=es; s+=dil)
                    {
                        t = 2*(int)Li - 2 - s;      //PyTorch-style
                        //t = 2*(int)Li - 1 - s;    //Kaldi-style
                        //this ensures reflected extrapolation to any length
                        while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                        X += (t-prev_t) * (int)Ni;
                        sm += cblas_sdot((int)Ni,X,1,K,1);
                        K += Ni; prev_t = t;
                    }
                }
                else                    //circular
                {
                    for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; s+=dil)
                    {
                        t = s - (int)Li;    //this ensures circular extrapolation to any length
                        while (t>=(int)Li) { t -= (int)Li; }
                        X += (t-prev_t) * (int)Ni;
                        sm += cblas_sdot((int)Ni,X,1,K,1);
                        K += Ni; prev_t = t;
                    }
                }
                X += (ss-prev_t) * (int)Ni;
                prev_t = ss;

                *Y = sm;
            }
            B -= No; K -= Nk;
            X += str*Ni; prev_t += str;
            ss += str; es += str; ++l;
        }

        //Put X to start of next batch member
        X += ((int)Li-ss) * (int)Ni;
    }

    return 0;
}


int conv1d_cblas_d (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1d_cblas_d: str (stride) must be positive\n"); return 1; }
    if (dil<1u) { fprintf(stderr,"error in conv1d_cblas_d: dil (dilation) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1d_cblas_d: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1d_cblas_d: Lk (kernel_size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1d_cblas_d: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1d_cblas_d: No (num output neurons) must be positive\n"); return 1; }
    if (Nb<1u) { fprintf(stderr,"error in conv1d_cblas_d: Nb (batch size) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in conv1d_cblas_d: pad length must be > -Li\n"); return 1; }
    if (pad_mode<0 || pad_mode>3) { fprintf(stderr,"error in conv1d_cblas_d: pad_mode must be an int in {0,1,2,3}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in conv1d_cblas_d: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in conv1d_cblas_d: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-Tk)/str + (size_t)(ceil_mode && (size_t)(Ti-Tk)%str);

    const size_t Nk = Ni*Lk*No;             //total num elements in K
    const size_t jump = Ni*dil;             //fixed jump due to dilation for X below
    size_t l;                               //current samp within Y in [0 Lo-1]
    double sm;                              //intermediate sum
    int ss, es;                             //current start-samp, end-samp
    int t, prev_t;                          //non-negative samps during extrapolation (padding)

    //Loop over batch members
    for (size_t b=Nb; b>0u; --b)
    {
        //Reset samp counts
        l = 0u; t = prev_t = 0;
        ss = -pad; es = ss + Tk - 1;

        //K before or overlapping first samp of X
        while (ss<0 && l<Lo)
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
                    for (int s=ss; s<0 && s<=es; s+=dil, K+=Ni)
                    {
                        sm += cblas_ddot((int)Ni,X,1,K,1);
                    }
                }
                else if (pad_mode==2)   //reflect
                {
                    for (int s=ss; s<0 && s<=es; s+=dil)
                    {
                        t = -s;         //PyTorch-style
                        //this ensures reflected extrapolation to any length
                        while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                        X += (t-prev_t) * (int)Ni;
                        sm += cblas_ddot((int)Ni,X,1,K,1);
                        K += Ni; prev_t = t;
                    }
                    X -= prev_t * (int)Ni;
                    prev_t = 0;
                }
                else                    //circular
                {
                    for (int s=ss; s<0 && s<=es; s+=dil)
                    {
                        t = (int)Li + s;    //this ensures circular extrapolation to any length
                        while (t<0) { t += (int)Li; }
                        X += (t-prev_t) * (int)Ni;
                        sm += cblas_ddot((int)Ni,X,1,K,1);
                        K += Ni; prev_t = t;
                    }
                    X -= prev_t * (int)Ni;
                    prev_t = 0;
                }

                //Non-negative samps
                if (es>=0)
                {
                    for (int s=es%(int)dil; s<=es; s+=dil)
                    {
                        X += (s-prev_t) * (int)Ni;
                        sm += cblas_ddot((int)Ni,X,1,K,1);
                        K += Ni; prev_t = s;
                    }
                }
                X -= prev_t * (int)Ni;
                prev_t = 0;

                *Y = sm;
            }
            B -= No; K -= Nk;
            ss += str; es += str; ++l;
        }
        X += ss*(int)Ni;

        //K fully within X
        while (es<(int)Li && l<Lo)
        {
            for (size_t o=No; o>0u; --o, ++B, ++Y)
            {
                sm = *B;
                for (size_t l=Lk; l>0u; --l, X+=jump, K+=Ni)
                {
                    sm += cblas_ddot((int)Ni,X,1,K,1);
                }
                *Y = sm;
                X -= Lk*jump;
            }
            B -= No; K -= Nk; X += Ni*str;
            es += str; ++l;
        }
        prev_t = ss = es - Tk + 1;

        //K past or overlapping last samp of X
        while (l<Lo)
        {
            //Get num valid samps
            const int V = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

            for (size_t o=No; o>0u; --o, ++B, ++Y)
            {
                sm = *B;

                //Valid samps
                for (int s=ss; s<(int)Li; s+=dil)
                {
                    X += (s-prev_t) * (int)Ni;
                    sm += cblas_ddot((int)Ni,X,1,K,1);
                    K += Ni; prev_t = s;
                }

                //Non-valid samps
                if (pad_mode==0)        //zeros
                {
                    K += Ni*(Lk-(size_t)V);
                }
                else if (pad_mode==1)   //repeat
                {
                    X += ((int)Li-1-prev_t) * (int)Ni;
                    for (int s=ss+V*(int)dil; s<=es; s+=dil, K+=Ni)
                    {
                        sm += cblas_ddot((int)Ni,X,1,K,1);
                    }
                    prev_t = (int)Li - 1;
                }
                else if (pad_mode==2)   //reflect
                {
                    for (int s=ss+V*(int)dil; s<=es; s+=dil)
                    {
                        t = 2*(int)Li - 2 - s;      //PyTorch-style
                        //this ensures reflected extrapolation to any length
                        while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                        X += (t-prev_t) * (int)Ni;
                        sm += cblas_ddot((int)Ni,X,1,K,1);
                        K += Ni; prev_t = t;
                    }
                }
                else                    //circular
                {
                    for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; s+=dil)
                    {
                        t = s - (int)Li;    //this ensures circular extrapolation to any length
                        while (t>=(int)Li) { t -= (int)Li; }
                        X += (t-prev_t) * (int)Ni;
                        sm += cblas_ddot((int)Ni,X,1,K,1);
                        K += Ni; prev_t = t;
                    }
                }
                X += (ss-prev_t) * (int)Ni;
                prev_t = ss;

                *Y = sm;
            }
            B -= No; K -= Nk;
            X += str*Ni; prev_t += str;
            ss += str; es += str; ++l;
        }

        //Put X to start of next batch member
        X += ((int)Li-ss) * (int)Ni;
    }

    return 0;
}


int conv1d_cblas_c (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1d_cblas_c: str (stride) must be positive\n"); return 1; }
    if (dil<1u) { fprintf(stderr,"error in conv1d_cblas_c: dil (dilation) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1d_cblas_c: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1d_cblas_c: Lk (kernel_size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1d_cblas_c: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1d_cblas_c: No (num output neurons) must be positive\n"); return 1; }
    if (Nb<1u) { fprintf(stderr,"error in conv1d_cblas_c: Nb (batch size) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in conv1d_cblas_c: pad length must be > -Li\n"); return 1; }
    if (pad_mode<0 || pad_mode>3) { fprintf(stderr,"error in conv1d_cblas_c: pad_mode must be an int in {0,1,2,3}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in conv1d_cblas_c: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in conv1d_cblas_c: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-Tk)/str + (size_t)(ceil_mode && (size_t)(Ti-Tk)%str);

    const size_t Nk = 2u*Ni*Lk*No;          //total num elements in K
    const size_t jump = 2u*Ni*dil;          //fixed jump due to dilation for X below
    size_t l;                               //current samp within Y in [0 Lo-1]
    float sm[2], smr, smi;                  //intermediate sums
    int ss, es;                             //current start-samp, end-samp
    int t, prev_t;                          //non-negative samps during extrapolation (padding)

    //Loop over batch members
    for (size_t b=Nb; b>0u; --b)
    {
        //Reset samp counts
        l = 0u; t = prev_t = 0;
        ss = -pad; es = ss + Tk - 1;

        //K before or overlapping first samp of X
        while (ss<0 && l<Lo)
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
                    for (int s=ss; s<0 && s<=es; s+=dil, K+=2u*Ni)
                    {
                        cblas_cdotu_sub((int)Ni,X,1,K,1,sm);
                        smr += sm[0]; smi += sm[1];
                    }
                }
                else if (pad_mode==2)   //reflect
                {
                    for (int s=ss; s<0 && s<=es; s+=dil)
                    {
                        t = -s;         //PyTorch-style
                        //this ensures reflected extrapolation to any length
                        while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                        X += 2*(t-prev_t)*(int)Ni;
                        cblas_cdotu_sub((int)Ni,X,1,K,1,sm);
                        smr += sm[0]; smi += sm[1];
                        K += 2u*Ni; prev_t = t;
                    }
                    X -= 2 * prev_t * (int)Ni;
                    prev_t = 0;
                }
                else                    //circular
                {
                    for (int s=ss; s<0 && s<=es; s+=dil)
                    {
                        t = (int)Li + s;    //this ensures circular extrapolation to any length
                        while (t<0) { t += (int)Li; }
                        X += 2*(t-prev_t)*(int)Ni;
                        cblas_cdotu_sub((int)Ni,X,1,K,1,sm);
                        smr += sm[0]; smi += sm[1];
                        K += 2u*Ni; prev_t = t;
                    }
                    X -= 2 * prev_t * (int)Ni;
                    prev_t = 0;
                }

                //Non-negative samps
                if (es>=0)
                {
                    for (int s=es%(int)dil; s<=es; s+=dil)
                    {
                        X += 2*(s-prev_t)*(int)Ni;
                        cblas_cdotu_sub((int)Ni,X,1,K,1,sm);
                        smr += sm[0]; smi += sm[1];
                        K += 2u*Ni; prev_t = s;
                    }
                }
                X -= 2*prev_t*(int)Ni;
                prev_t = 0;

                *Y++ = smr; *Y++ = smi;
            }
            B -= 2u*No; K -= Nk;
            ss += str; es += str; ++l;
        }
        X += 2*ss*(int)Ni;

        //K fully within X
        while (es<(int)Li && l<Lo)
        {
            for (size_t o=No; o>0u; --o)
            {
                smr = *B++; smi = *B++;
                for (size_t l=Lk; l>0u; --l, X+=jump, K+=2u*Ni)
                {
                    cblas_cdotu_sub((int)Ni,X,1,K,1,sm);
                    smr += sm[0]; smi += sm[1];
                }
                *Y++ = smr; *Y++ = smi;
                X -= 2u*Ni*Lk*dil;
            }
            B -= 2u*No; K -= Nk; X += 2u*Ni*str;
            es += str; ++l;
        }
        prev_t = ss = es - Tk + 1;

        //K past or overlapping last samp of X
        while (l<Lo)
        {
            //Get num valid samps
            const int V = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

            for (size_t o=No; o>0u; --o)
            {
                smr = *B++; smi = *B++;

                //Valid samps
                for (int s=ss; s<(int)Li; s+=dil)
                {
                    X += 2*(s-prev_t)*(int)Ni;
                    cblas_cdotu_sub((int)Ni,X,1,K,1,sm);
                    smr += sm[0]; smi += sm[1];
                    K += 2u*Ni; prev_t = s;
                }

                //Non-valid samps
                if (pad_mode==0)        //zeros
                {
                    K += 2u*Ni*(Lk-(size_t)V);
                }
                else if (pad_mode==1)   //repeat
                {
                    X += 2*((int)Li-1-prev_t)*(int)Ni;
                    for (int s=ss+V*(int)dil; s<=es; s+=dil, K+=2u*Ni)
                    {
                        cblas_cdotu_sub((int)Ni,X,1,K,1,sm);
                        smr += sm[0]; smi += sm[1];
                    }
                    prev_t = (int)Li - 1;
                }
                else if (pad_mode==2)   //reflect
                {
                    for (int s=ss+V*(int)dil; s<=es; s+=dil)
                    {
                        t = 2*(int)Li - 2 - s;      //PyTorch-style
                        //this ensures reflected extrapolation to any length
                        while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                        X += 2*(t-prev_t)*(int)Ni;
                        cblas_cdotu_sub((int)Ni,X,1,K,1,sm);
                        smr += sm[0]; smi += sm[1];
                        K += 2u*Ni; prev_t = t;
                    }
                }
                else                    //circular
                {
                    for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; s+=dil)
                    {
                        t = s - (int)Li;    //this ensures circular extrapolation to any length
                        while (t>=(int)Li) { t -= (int)Li; }
                        X += 2*(t-prev_t)*(int)Ni;
                        cblas_cdotu_sub((int)Ni,X,1,K,1,sm);
                        smr += sm[0]; smi += sm[1];
                        K += 2u*Ni; prev_t = t;
                    }
                }
                X += 2 * (ss-prev_t) * (int)Ni;
                prev_t = ss;

                *Y++ = smr; *Y++ = smi;
            }
            B -= 2u*No; K -= Nk;
            X += 2u*str*Ni; prev_t += str;
            ss += str; es += str; ++l;
        }

        //Put X to start of next batch member
        X += 2 * ((int)Li-ss) * (int)Ni;
    }

    return 0;
}


int conv1d_cblas_z (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1d_cblas_z: str (stride) must be positive\n"); return 1; }
    if (dil<1u) { fprintf(stderr,"error in conv1d_cblas_z: dil (dilation) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1d_cblas_z: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1d_cblas_z: Lk (kernel_size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1d_cblas_z: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1d_cblas_z: No (num output neurons) must be positive\n"); return 1; }
    if (Nb<1u) { fprintf(stderr,"error in conv1d_cblas_z: Nb (batch size) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in conv1d_cblas_z: pad length must be > -Li\n"); return 1; }
    if (pad_mode<0 || pad_mode>3) { fprintf(stderr,"error in conv1d_cblas_z: pad_mode must be an int in {0,1,2,3}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in conv1d_cblas_z: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in conv1d_cblas_z: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-Tk)/str + (size_t)(ceil_mode && (size_t)(Ti-Tk)%str);

    const size_t Nk = 2u*Ni*Lk*No;          //total num elements in K
    const size_t jump = 2u*Ni*dil;          //fixed jump due to dilation for X below
    size_t l;                               //current samp within Y in [0 Lo-1]
    double sm[2], smr, smi;                 //intermediate sums
    int ss, es;                             //current start-samp, end-samp
    int t, prev_t;                          //non-negative samps during extrapolation (padding)

    //Loop over batch members
    for (size_t b=Nb; b>0u; --b)
    {
        //Reset samp counts
        l = 0u; t = prev_t = 0;
        ss = -pad; es = ss + Tk - 1;

        //K before or overlapping first samp of X
        while (ss<0 && l<Lo)
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
                    for (int s=ss; s<0 && s<=es; s+=dil, K+=2u*Ni)
                    {
                        cblas_zdotu_sub((int)Ni,X,1,K,1,sm);
                        smr += sm[0]; smi += sm[1];
                    }
                }
                else if (pad_mode==2)   //reflect
                {
                    for (int s=ss; s<0 && s<=es; s+=dil)
                    {
                        t = -s;         //PyTorch-style
                        //this ensures reflected extrapolation to any length
                        while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                        X += 2*(t-prev_t)*(int)Ni;
                        cblas_zdotu_sub((int)Ni,X,1,K,1,sm);
                        smr += sm[0]; smi += sm[1];
                        K += 2u*Ni; prev_t = t;
                    }
                    X -= 2 * prev_t * (int)Ni;
                    prev_t = 0;
                }
                else                    //circular
                {
                    for (int s=ss; s<0 && s<=es; s+=dil)
                    {
                        t = (int)Li + s;    //this ensures circular extrapolation to any length
                        while (t<0) { t += (int)Li; }
                        X += 2*(t-prev_t)*(int)Ni;
                        cblas_zdotu_sub((int)Ni,X,1,K,1,sm);
                        smr += sm[0]; smi += sm[1];
                        K += 2u*Ni; prev_t = t;
                    }
                    X -= 2 * prev_t * (int)Ni;
                    prev_t = 0;
                }

                //Non-negative samps
                if (es>=0)
                {
                    for (int s=es%(int)dil; s<=es; s+=dil)
                    {
                        X += 2*(s-prev_t)*(int)Ni;
                        cblas_zdotu_sub((int)Ni,X,1,K,1,sm);
                        smr += sm[0]; smi += sm[1];
                        K += 2u*Ni; prev_t = s;
                    }
                }
                X -= 2*prev_t*(int)Ni;
                prev_t = 0;

                *Y++ = smr; *Y++ = smi;
            }
            B -= 2u*No; K -= Nk;
            ss += str; es += str; ++l;
        }
        X += 2*ss*(int)Ni;

        //K fully within X
        while (es<(int)Li && l<Lo)
        {
            for (size_t o=No; o>0u; --o)
            {
                smr = *B++; smi = *B++;
                for (size_t l=Lk; l>0u; --l, X+=jump, K+=2u*Ni)
                {
                    cblas_zdotu_sub((int)Ni,X,1,K,1,sm);
                    smr += sm[0]; smi += sm[1];
                }
                *Y++ = smr; *Y++ = smi;
                X -= 2u*Ni*Lk*dil;
            }
            B -= 2u*No; K -= Nk; X += 2u*Ni*str;
            es += str; ++l;
        }
        prev_t = ss = es - Tk + 1;

        //K past or overlapping last samp of X
        while (l<Lo)
        {
            //Get num valid samps
            const int V = (ss<(int)Li) ? 1 + ((int)Li-1-ss)/dil : 0;

            for (size_t o=No; o>0u; --o)
            {
                smr = *B++; smi = *B++;

                //Valid samps
                for (int s=ss; s<(int)Li; s+=dil)
                {
                    X += 2*(s-prev_t)*(int)Ni;
                    cblas_zdotu_sub((int)Ni,X,1,K,1,sm);
                    smr += sm[0]; smi += sm[1];
                    K += 2u*Ni; prev_t = s;
                }

                //Non-valid samps
                if (pad_mode==0)        //zeros
                {
                    K += 2u*Ni*(Lk-(size_t)V);
                }
                else if (pad_mode==1)   //repeat
                {
                    X += 2*((int)Li-1-prev_t)*(int)Ni;
                    for (int s=ss+V*(int)dil; s<=es; s+=dil, K+=2u*Ni)
                    {
                        cblas_zdotu_sub((int)Ni,X,1,K,1,sm);
                        smr += sm[0]; smi += sm[1];
                    }
                    prev_t = (int)Li - 1;
                }
                else if (pad_mode==2)   //reflect
                {
                    for (int s=ss+V*(int)dil; s<=es; s+=dil)
                    {
                        t = 2*(int)Li - 2 - s;      //PyTorch-style
                        //this ensures reflected extrapolation to any length
                        while (t<0 || t>=(int)Li) { t = (t<0) ? -t-1 : (t<(int)Li) ? t : 2*(int)Li-1-t; }
                        X += 2*(t-prev_t)*(int)Ni;
                        cblas_zdotu_sub((int)Ni,X,1,K,1,sm);
                        smr += sm[0]; smi += sm[1];
                        K += 2u*Ni; prev_t = t;
                    }
                }
                else                    //circular
                {
                    for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; s+=dil)
                    {
                        t = s - (int)Li;    //this ensures circular extrapolation to any length
                        while (t>=(int)Li) { t -= (int)Li; }
                        X += 2*(t-prev_t)*(int)Ni;
                        cblas_zdotu_sub((int)Ni,X,1,K,1,sm);
                        smr += sm[0]; smi += sm[1];
                        K += 2u*Ni; prev_t = t;
                    }
                }
                X += 2 * (ss-prev_t) * (int)Ni;
                prev_t = ss;

                *Y++ = smr; *Y++ = smi;
            }
            B -= 2u*No; K -= Nk;
            X += 2u*str*Ni; prev_t += str;
            ss += str; es += str; ++l;
        }

        //Put X to start of next batch member
        X += 2 * ((int)Li-ss) * (int)Ni;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
