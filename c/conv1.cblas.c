//This is just like conv1d, but doesn't allow dilation.
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

//Profile notes:
//Compared to this, conv1 ran at 0.45 time for small-to-medium input.
//Compared to this, conv1 ran at 0.65 time for a large input.

//The following params/opts are included:
//Ni:           size_t  num input neurons (leading dim of X)
//No:           size_t  num output neurons (leading dim of Y)
//Li:           size_t  length of input time-series (other dim of X)
//Lk:           size_t  length of kernel in time (num cols of K)
//pad:          int     padding length (pad>0 means add extra samps)
//str:          size_t  stride length (step-size) in samps
//ceil_mode:    bool    calculate Lo with ceil (see above)
//pad_mode:     int     padding mode (0=zeros, 1=replicate, 2=reflect, 3=circular)

#include <stdio.h>
#include <string.h>
#include <cblas.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int conv1_cblas_s (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int conv1_cblas_d (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int conv1_cblas_c (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);
int conv1_cblas_z (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode);


int conv1_cblas_s (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1_blas_s: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1_blas_s: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1_blas_s: Lk (kernel_size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1_blas_s: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1_blas_s: No (num output neurons) must be positive\n"); return 1; }
    if (pad_mode<0 || pad_mode>3) { fprintf(stderr,"error in conv1_blas_s: pad_mode must be an int in {0,1,2,3}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;     //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in conv1_blas_s: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in conv1_blas_s: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);
    
    const size_t Nk = Ni*Lk*No;         //total num elements in K
    size_t w=0u;                        //current window (frame)
    float sm;                           //intermediate sum
    int ss=-pad, es=ss+(int)Lk-1;       //current start-samp, end-samp
    int n = 0, prev_n = 0;              //non-negative samps during extrapolation (padding)

    //struct timespec tic, toc; clock_gettime(CLOCK_REALTIME,&tic);

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
                for (int s=ss; s<0 && s<=es; ++s, K+=Ni)
                {
                    sm += cblas_sdot((int)Ni,X,1,K,1);
                }
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    n = -s - 1;         //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { n = (n<0) ? -n-1 : (n<(int)Li) ? n : 2*(int)Li-1-n; }
                    X += (n-prev_n) * (int)Ni;
                    sm += cblas_sdot((int)Ni,X,1,K,1);
                    K += Ni; prev_n = n;
                }
                X -= prev_n * (int)Ni;
                prev_n = 0;
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    n = (int)Li + s;    //this ensures circular extrapolation to any length
                    while (n<0) { n += (int)Li; }
                    X += (n-prev_n) * (int)Ni;
                    sm += cblas_sdot((int)Ni,X,1,K,1);
                    K += Ni; prev_n = n;
                }
                X -= prev_n * (int)Ni;
                prev_n = 0;
            }

            //Non-negative samps
            if (es>=0)
            {
                sm += cblas_sdot((es+1)*(int)Ni,X,1,K,1);
                K += (es+1)*(int)Ni;
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
            *Y = *B + cblas_sdot((int)(Ni*Lk),X,1,K,1);
            K += Ni*Lk;
        }
        B -= No; K -= Nk; X += Ni*str;
        es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int v = (int)Li - ss;

        for (size_t o=No; o>0u; --o, ++B, ++Y)
        {
            sm = *B;

            //Valid samps
            if (v>0)
            {
                sm += cblas_sdot(v*(int)Ni,X,1,K,1);
                X += v*(int)Ni; K += v*(int)Ni;
            }
            else { X += ((int)Li-ss) * (int)Ni; }

            //Non-valid samps
            if (pad_mode==0)        //zeros
            {
                K += (v>0) ? Ni*(Lk-(size_t)v) : Ni*Lk;
            }
            else if (pad_mode==1)   //repeat
            {
                X -= Ni;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s, K+=Ni)
                {
                    sm += cblas_sdot((int)Ni,X,1,K,1);
                }
                X += Ni;
            }
            else if (pad_mode==2)   //reflect
            {
                prev_n = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    n = 2*(int)Li-1-s;  //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { n = (n<0) ? -n-1 : (n<(int)Li) ? n : 2*(int)Li-1-n; }
                    X += (n-prev_n) * (int)Ni;
                    sm += cblas_sdot((int)Ni,X,1,K,1);
                    K += Ni; prev_n = n;
                }
                X += ((int)Li-prev_n) * (int)Ni;
            }
            else                    //circular
            {
                prev_n = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    n = s - (int)Li;    //this ensures circular extrapolation to any length
                    while (n>=(int)Li) { n -= (int)Li; }
                    X += (n-prev_n) * (int)Ni;
                    sm += cblas_sdot((int)Ni,X,1,K,1);
                    K += Ni; prev_n = n;
                }
                X += ((int)Li-prev_n) * (int)Ni;
            }
            X += (ss-(int)Li) * (int)Ni;

            *Y = sm;
        }
        B -= No; K -= Nk; X += str*Ni;
        ss+=str; es+=str; ++w;
    }

    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(double)(toc.tv_sec-tic.tv_sec)*1e3+(double)(toc.tv_nsec-tic.tv_nsec)/1e6);

    return 0;
}


int conv1_cblas_d (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1_cblas_d: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1_cblas_d: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1_cblas_d: Lk (kernel_size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1_cblas_d: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1_cblas_d: No (num output neurons) must be positive\n"); return 1; }

    const int Ti = (int)Li + 2*pad;     //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in conv1_cblas_d: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in conv1_cblas_d: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);
    
    const size_t Nk = Ni*Lk*No;         //total num elements in K
    size_t w=0u;                        //current window (frame)
    double sm;                          //intermediate sum
    int ss=-pad, es=ss+(int)Lk-1;       //current start-samp, end-samp
    int n = 0, prev_n = 0;              //non-negative samps during extrapolation (padding)

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
                for (int s=ss; s<0 && s<=es; ++s, K+=Ni)
                {
                    sm += cblas_ddot((int)Ni,X,1,K,1);
                }
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    n = -s - 1;         //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { n = (n<0) ? -n-1 : (n<(int)Li) ? n : 2*(int)Li-1-n; }
                    X += (n-prev_n) * (int)Ni;
                    sm += cblas_ddot((int)Ni,X,1,K,1);
                    K += Ni; prev_n = n;
                }
                X -= prev_n * (int)Ni;
                prev_n = 0;
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    n = (int)Li + s;    //this ensures circular extrapolation to any length
                    while (n<0) { n += (int)Li; }
                    X += (n-prev_n) * (int)Ni;
                    sm += cblas_ddot((int)Ni,X,1,K,1);
                    K += Ni; prev_n = n;
                }
                X -= prev_n * (int)Ni;
                prev_n = 0;
            }

            //Non-negative samps
            if (es>=0)
            {
                sm += cblas_ddot((es+1)*(int)Ni,X,1,K,1);
                K += (es+1)*(int)Ni;
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
            *Y = *B + cblas_ddot((int)(Ni*Lk),X,1,K,1);
            K += Ni*Lk;
        }
        B -= No; K -= Nk; X += Ni*str;
        es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int v = (int)Li - ss;

        for (size_t o=No; o>0u; --o, ++B, ++Y)
        {
            sm = *B;

            //Valid samps
            if (v>0)
            {
                sm += cblas_ddot(v*(int)Ni,X,1,K,1);
                X += v*(int)Ni; K += v*(int)Ni;
            }
            else { X += ((int)Li-ss) * (int)Ni; }

            //Non-valid samps
            if (pad_mode==0)        //zeros
            {
                K += (v>0) ? Ni*(Lk-(size_t)v) : Ni*Lk;
            }
            else if (pad_mode==1)   //repeat
            {
                X -= Ni;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s, K+=Ni)
                {
                    sm += cblas_ddot((int)Ni,X,1,K,1);
                }
                X += Ni;
            }
            else if (pad_mode==2)   //reflect
            {
                prev_n = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    n = 2*(int)Li-1-s;  //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { n = (n<0) ? -n-1 : (n<(int)Li) ? n : 2*(int)Li-1-n; }
                    X += (n-prev_n) * (int)Ni;
                    sm += cblas_ddot((int)Ni,X,1,K,1);
                    K += Ni; prev_n = n;
                }
                X += ((int)Li-prev_n) * (int)Ni;
            }
            else                    //circular
            {
                prev_n = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    n = s - (int)Li;    //this ensures circular extrapolation to any length
                    while (n>=(int)Li) { n -= (int)Li; }
                    X += (n-prev_n) * (int)Ni;
                    sm += cblas_ddot((int)Ni,X,1,K,1);
                    K += Ni; prev_n = n;
                }
                X += ((int)Li-prev_n) * (int)Ni;
            }
            X += (ss-(int)Li) * (int)Ni;

            *Y = sm;
        }
        B -= No; K -= Nk; X += str*Ni;
        ss+=str; es+=str; ++w;
    }

    return 0;
}


int conv1_cblas_c (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1_cblas_c: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1_cblas_c: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1_cblas_c: Lk (kernel_size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1_cblas_c: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1_cblas_c: No (num output neurons) must be positive\n"); return 1; }

    const int Ti = (int)Li + 2*pad;     //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in conv1_cblas_c: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in conv1_cblas_c: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);
    
    const size_t Nb = 2u*No;            //total num elements in B
    const size_t Nk = 2u*Ni*Lk*No;      //total num elements in K
    size_t w=0u;                        //current window (frame)
    float sm[2], smr, smi;              //intermediate sums
    int ss=-pad, es=ss+(int)Lk-1;       //current start-samp, end-samp
    int n = 0, prev_n = 0;              //non-negative samps during extrapolation (padding)

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
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    cblas_cdotu_sub((int)Ni,X,1,K,1,sm);
                    smr += sm[0]; smi += sm[1];
                    K += 2u*Ni;
                }
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    n = -s - 1;         //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { n = (n<0) ? -n-1 : (n<(int)Li) ? n : 2*(int)Li-1-n; }
                    X += 2 * (n-prev_n) * (int)Ni;
                    cblas_cdotu_sub((int)Ni,X,1,K,1,sm);
                    smr += sm[0]; smi += sm[1];
                    K += 2u*Ni;
                    prev_n = n;
                }
                X -= 2 * prev_n * (int)Ni;
                prev_n = 0;
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    n = (int)Li + s;    //this ensures circular extrapolation to any length
                    while (n<0) { n += (int)Li; }
                    X += 2 * (n-prev_n) * (int)Ni;
                    cblas_cdotu_sub((int)Ni,X,1,K,1,sm);
                    smr += sm[0]; smi += sm[1];
                    K += 2u*Ni;
                    prev_n = n;
                }
                X -= 2 * prev_n * (int)Ni;
                prev_n = 0;
            }

            //Non-negative samps
            if (es>=0)
            {
                cblas_cdotu_sub((es+1)*(int)Ni,X,1,K,1,sm);
                smr += sm[0]; smi += sm[1];
                K += 2*(es+1)*(int)Ni;
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
            cblas_cdotu_sub((int)(Ni*Lk),X,1,K,1,Y);
            *Y++ += *B++; *Y++ += *B++;
            K += 2u*Ni*Lk;
        }
        B -= Nb; K -= Nk; X += 2u*Ni*str;
        es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int v = (int)Li - ss;

        for (size_t o=No; o>0u; --o)
        {
            smr = *B++; smi = *B++;

            //Valid samps
            if (v>0)
            {
                cblas_cdotu_sub(v*(int)Ni,X,1,K,1,sm);
                smr += sm[0]; smi += sm[1];
                X += 2*v*(int)Ni; K += 2*v*(int)Ni;
            }
            else { X += 2*((int)Li-ss)*(int)Ni; }

            //Non-valid samps
            if (pad_mode==0)        //zeros
            {
                K += (v>0) ? 2u*Ni*(Lk-(size_t)v) : 2u*Ni*Lk;
            }
            else if (pad_mode==1)   //repeat
            {
                X -= 2u*Ni;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    cblas_cdotu_sub((int)Ni,X,1,K,1,sm);
                    smr += sm[0]; smi += sm[1];
                    K += 2u*Ni;
                    prev_n = n;
                }
                X += 2u*Ni;
            }
            else if (pad_mode==2)   //reflect
            {
                prev_n = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    n = 2*(int)Li-1-s;  //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { n = (n<0) ? -n-1 : (n<(int)Li) ? n : 2*(int)Li-1-n; }
                    X += 2 * (n-prev_n) * (int)Ni;
                    cblas_cdotu_sub((int)Ni,X,1,K,1,sm);
                    smr += sm[0]; smi += sm[1];
                    K += 2u*Ni;
                    prev_n = n;
                }
                X += 2 * ((int)Li-prev_n) * (int)Ni;
            }
            else                    //circular
            {
                prev_n = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    n = s - (int)Li;    //this ensures circular extrapolation to any length
                    while (n>=(int)Li) { n -= (int)Li; }
                    X += 2 * (n-prev_n) * (int)Ni;
                    cblas_cdotu_sub((int)Ni,X,1,K,1,sm);
                    smr += sm[0]; smi += sm[1];
                    K += 2u*Ni;
                    prev_n = n;
                }
                X += 2 * ((int)Li-prev_n) * (int)Ni;
            }
            X += 2 * (ss-(int)Li) * (int)Ni;

            *Y++ = smr; *Y++ = smi;
        }
        B -= Nb; K -= Nk; X += 2u*str*Ni;
        ss+=str; es+=str; ++w;
    }

    return 0;
}


int conv1_cblas_z (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1_cblas_z: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1_cblas_z: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1_cblas_z: Lk (kernel_size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1_cblas_z: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1_cblas_z: No (num output neurons) must be positive\n"); return 1; }

    const int Ti = (int)Li + 2*pad;     //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in conv1_cblas_z: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in conv1_cblas_z: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);
    
    const size_t Nb = 2u*No;            //total num elements in B
    const size_t Nk = 2u*Ni*Lk*No;      //total num elements in K
    size_t w=0u;                        //current window (frame)
    double sm[2], smr, smi;             //intermediate sums
    int ss=-pad, es=ss+(int)Lk-1;       //current start-samp, end-samp
    int n = 0, prev_n = 0;              //non-negative samps during extrapolation (padding)

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
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    cblas_zdotu_sub((int)Ni,X,1,K,1,sm);
                    smr += sm[0]; smi += sm[1];
                    K += 2u*Ni;
                }
            }
            else if (pad_mode==2)   //reflect
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    n = -s - 1;         //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { n = (n<0) ? -n-1 : (n<(int)Li) ? n : 2*(int)Li-1-n; }
                    X += 2 * (n-prev_n) * (int)Ni;
                    cblas_zdotu_sub((int)Ni,X,1,K,1,sm);
                    smr += sm[0]; smi += sm[1];
                    K += 2u*Ni;
                    prev_n = n;
                }
                X -= 2 * prev_n * (int)Ni;
                prev_n = 0;
            }
            else                    //circular
            {
                for (int s=ss; s<0 && s<=es; ++s)
                {
                    n = (int)Li + s;    //this ensures circular extrapolation to any length
                    while (n<0) { n += (int)Li; }
                    X += 2 * (n-prev_n) * (int)Ni;
                    cblas_zdotu_sub((int)Ni,X,1,K,1,sm);
                    smr += sm[0]; smi += sm[1];
                    K += 2u*Ni;
                    prev_n = n;
                }
                X -= 2 * prev_n * (int)Ni;
                prev_n = 0;
            }

            //Non-negative samps
            if (es>=0)
            {
                cblas_zdotu_sub((es+1)*(int)Ni,X,1,K,1,sm);
                smr += sm[0]; smi += sm[1];
                K += 2*(es+1)*(int)Ni;
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
            cblas_zdotu_sub((int)(Ni*Lk),X,1,K,1,Y);
            *Y++ += *B++; *Y++ += *B++;
            K += 2u*Ni*Lk;
        }
        B -= Nb; K -= Nk; X += 2u*Ni*str;
        es+=str; ++w;
    }
    ss = es - (int)Lk + 1;

    //K past or overlapping last samp of X
    while (w<Lo)
    {
        //Get num valid samps
        int v = (int)Li - ss;

        for (size_t o=No; o>0u; --o)
        {
            smr = *B++; smi = *B++;

            //Valid samps
            if (v>0)
            {
                cblas_zdotu_sub(v*(int)Ni,X,1,K,1,sm);
                smr += sm[0]; smi += sm[1];
                X += 2*v*(int)Ni; K += 2*v*(int)Ni;
            }
            else { X += 2*((int)Li-ss)*(int)Ni; }

            //Non-valid samps
            if (pad_mode==0)        //zeros
            {
                K += (v>0) ? 2u*Ni*(Lk-(size_t)v) : 2u*Ni*Lk;
            }
            else if (pad_mode==1)   //repeat
            {
                X -= 2u*Ni;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    cblas_zdotu_sub((int)Ni,X,1,K,1,sm);
                    smr += sm[0]; smi += sm[1];
                    K += 2u*Ni;
                    prev_n = n;
                }
                X += 2u*Ni;
            }
            else if (pad_mode==2)   //reflect
            {
                prev_n = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    n = 2*(int)Li-1-s;  //this ensures reflected extrapolation to any length
                    while (n<0 || n>=(int)Li) { n = (n<0) ? -n-1 : (n<(int)Li) ? n : 2*(int)Li-1-n; }
                    X += 2 * (n-prev_n) * (int)Ni;
                    cblas_zdotu_sub((int)Ni,X,1,K,1,sm);
                    smr += sm[0]; smi += sm[1];
                    K += 2u*Ni;
                    prev_n = n;
                }
                X += 2 * ((int)Li-prev_n) * (int)Ni;
            }
            else                    //circular
            {
                prev_n = (int)Li;
                for (int s=(ss>(int)Li)?ss:(int)Li; s<=es; ++s)
                {
                    n = s - (int)Li;    //this ensures circular extrapolation to any length
                    while (n>=(int)Li) { n -= (int)Li; }
                    X += 2 * (n-prev_n) * (int)Ni;
                    cblas_zdotu_sub((int)Ni,X,1,K,1,sm);
                    smr += sm[0]; smi += sm[1];
                    K += 2u*Ni;
                    prev_n = n;
                }
                X += 2 * ((int)Li-prev_n) * (int)Ni;
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
