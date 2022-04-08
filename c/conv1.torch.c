//1D convolution using PyTorch shape conventions for X, K, Y.
//Shapes are described below for row-major case.

//X is the input of size Nb x Ni x Li.
//K is the tensor of convolving kernels with size No x Ni x Lk.
//Y is the output of size Nb x No x Lo,
//where Lo==L_out is set by:
//Lo =  ceil[1 + (Li + 2*pad - Lk)/stride], if ceil_mode is true
//Lo = floor[1 + (Li + 2*pad - Lk)/stride], if ceil_mode is false [default]

//The following params/opts are included:
//Ni:           size_t  num input channels (leading dim of X)
//No:           size_t  num output channels (leading dim of Y)
//Nb:           size_t  batch size (trailing dim of X and Y)
//Li:           size_t  length of input time-series (other dim of X)
//Lk:           size_t  length of kernel in time (num cols of K)
//pad:          int     padding length in [1-Li Li] (pad>0 means add extra samps)
//str:          size_t  stride length (step-size) in samps
//ceil_mode:    bool    calculate Lo with ceil (see above)
//pad_mode:     int     padding mode (0=zeros, 1=replicate, 2=reflect, 3=circular)

//Profile notes:
//It is about 5% faster to loop over output chans first, and then output samps,
//but then there is no real-time interpretation, so leaving it over output samps first.

//cblas_sdot was way (3x) slower!
//Probably because it is only over Lk, which is usually short.

#include <stdio.h>
#include "codee_nn.h"

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif


int conv1_torch_s (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1_torch_s: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1_torch_s: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1_torch_s: Lk (kernel_size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1_torch_s: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1_torch_s: No (num output neurons) must be positive\n"); return 1; }
    if (Nb<1u) { fprintf(stderr,"error in conv1_torch_s: Nb (batch size) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in conv1_torch_s: pad length must be > -Li\n"); return 1; }
    if (pad_mode && pad>(int)Li) { fprintf(stderr,"error in conv1_torch_s: Li (length of input vecs) must be >= pad length\n"); return 1; }
    if (pad_mode<0 || pad_mode>3) { fprintf(stderr,"error in conv1_torch_s: pad_mode must be an int in {0,1,2,3}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;     //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in conv1_torch_s: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in conv1_torch_s: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);
    
    const size_t Nk = Ni*Lk*No;         //total num elements in K
    float sm;                           //intermediate accumulation
    int ss, es;                         //current start-samp, end-samp
    size_t l;                           //current samp within Y in [0 Lo-1]
    
    //struct timespec tic, toc; clock_gettime(CLOCK_REALTIME,&tic);

    //Loop over batch members
    for (size_t b=Nb; b>0u; --b)
    {
        //Reset samp counts
        l = 0u; ss = -pad; es = ss + (int)Lk - 1;

        //Kernel overlaps first samp
        while (ss<0 && l<Lo)
        {
            //Loop over output chans
            for (size_t o=No; o>0u; --o, ++B, X-=Ni*Li, Y+=Lo)
            {
                sm = *B;
                for (size_t i=Ni; i>0u; --i, X+=(int)(Li-Lk)-ss)
                {
                    //Negative samps
                    if (pad_mode==0)        //zero pad
                    {
                        K += -ss;
                    }
                    else if (pad_mode==1)   //repeat pad
                    {
                        for (int k=ss; k<0; ++k, ++K) { sm += *X * *K; }
                    }
                    else if (pad_mode==2)   //reflect pad
                    {
                        X += -ss;
                        for (int k=ss; k<0; ++k, --X, ++K) { sm += *X * *K; }
                    }
                    else                    //circular pad
                    {
                        X += (int)Li + ss;
                        for (int k=ss; k<0; ++k, ++X, ++K) { sm += *X * *K; }
                        X -= Li;
                    }

                    //Nonnegative samps
                    for (int k=(int)Lk+ss; k>0; --k, ++X, ++K)
                    {
                        sm += *X * *K;
                    }
                }
                *Y = sm;
            }
            ss += str; ++l;
            K -= Nk; B -= No; Y -= No*Lo-1u;
        }
        es = ss + (int)Lk - 1;

        //Kernel fully within X
        while (es<(int)Li && l<Lo)
        {
            //Loop over output chans
            for (size_t o=No; o>0u; --o, ++B, X-=Ni*Li, Y+=Lo)
            {
                sm = *B;
                for (size_t i=Ni; i>0u; --i, X+=Li-Lk)
                {
                    for (size_t k=Lk; k>0u; --k, ++X, ++K) { sm += *X * *K; }
                }
                *Y = sm;
            }
            es += str; ++l;
            X += str; K -= Nk; B -= No; Y -= No*Lo-1u;
        }
        ss = es - (int)Lk + 1;

        //Kernel overlaps end samp
        while (l<Lo)
        {
            //Loop over output chans
            for (size_t o=No; o>0u; --o, ++B, X-=Ni*Li, Y+=Lo)
            {
                sm = *B;
                for (size_t i=Ni; i>0u; --i, X+=Li-Lk)
                {
                    //Valid samps
                    for (size_t k=Li-(size_t)ss; k>0u; --k, ++X, ++K)
                    {
                        sm += *X * *K;
                    }

                    //Nonvalid samps
                    if (pad_mode==0)        //zero pad
                    {
                        X += es - (int)Li + 1;
                        K += es - (int)Li + 1;
                    }
                    else if (pad_mode==1)   //repeat pad
                    {
                        --X;
                        for (size_t k=(size_t)es-Li+1u; k>0u; --k, ++K) { sm += *X * *K; }
                        X += es - (int)Li + 2;
                    }
                    else if (pad_mode==2)   //reflect pad
                    {
                        X -= 2;
                        for (size_t k=(size_t)es-Li+1u; k>0u; --k, --X, ++K) { sm += *X * *K; }
                        X += 2*(es-(int)Li+2);
                    }
                    else                    //circular pad
                    {
                        X -= Li;
                        for (size_t k=(size_t)es-Li+1u; k>0u; --k, ++X, ++K) { sm += *X * *K; }
                        X += (int)(Li+Lk) + ss - es - 1;
                    }
                }
                *Y = sm;
            }
            ss += str; es += str; ++l;
            X += str; K -= Nk; B -= No; Y -= No*Lo-1u; 
        }
        X += (int)(Ni*Li) - ss; Y += (No-1u)*Lo;
    }

    // //Loop over output chans
    // for (size_t o=No; o>0u; --o, ++B, X-=Li-Lk+str, K+=Ni*Lk)
    // {
    //     //Loop over output samps
    //     for (size_t l=Lo; l>0u; --l, ++Y, X-=Ni*Li-str, K-=Ni*Lk)
    //     {
    //         //Initialize sum
    //         sm = *B;

    //         //Loop over input chans
    //         for (size_t i=Ni; i>0u; --i, X+=Li-Lk)
    //         {
    //             //Dot product over kernel width
    //             for (size_t k=Lk; k>0u; --k, ++X, ++K) { sm += *X * *K; }
    //         }

    //         //Set Y[o,l]
    //         *Y = sm;
    //     }
    // }

    //clock_gettime(CLOCK_REALTIME,&toc);
    //fprintf(stderr,"elapsed time = %.6f ms\n",(double)(toc.tv_sec-tic.tv_sec)*1e3+(double)(toc.tv_nsec-tic.tv_nsec)/1e6);

    return 0;
}


int conv1_torch_d (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Li, const size_t Lk, const int pad, const size_t str, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1_torch_d: str (stride) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1_torch_d: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1_torch_d: Lk (kernel_size) must be positive\n"); return 1; }
    if (Nb<1u) { fprintf(stderr,"error in conv1_torch_d: Nb (batch size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1_torch_d: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1_torch_d: No (num output neurons) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in conv1_torch_d: pad length must be > -Li\n"); return 1; }
    if (pad_mode && pad>(int)Li) { fprintf(stderr,"error in conv1_torch_d: Li (length of input vecs) must be >= pad length\n"); return 1; }
    if (pad_mode<0 || pad_mode>3) { fprintf(stderr,"error in conv1_torch_d: pad_mode must be an int in {0,1,2,3}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;     //total length of vecs in X including padding
    if (Lk>Li) { fprintf(stderr,"error in conv1_torch_d: Li (length of input vecs) must be >= Lk\n"); return 1; }
    if (Ti<(int)Lk) { fprintf(stderr,"error in conv1_torch_d: Li+2*pad must be >= Lk\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-(int)Lk)/str + (size_t)(ceil_mode && (size_t)(Ti-(int)Lk)%str);
    
    const size_t Nk = Ni*Lk*No;         //total num elements in K
    double sm;                          //intermediate sum
    int ss=-pad, es=ss+(int)Lk-1;       //current start-samp, end-samp
    size_t l = 0u;                      //current samp within Y in [0 Lo-1]

    //Loop over batch members
    for (size_t b=Nb; b>0u; --b)
    {
        //Kernel overlaps first samp
        while (ss<0 && l<Lo)
        {
            //Loop over output chans
            for (size_t o=No; o>0u; --o, ++B, X-=Ni*Li, Y+=Lo)
            {
                sm = *B;
                for (size_t i=Ni; i>0u; --i, X+=(int)(Li-Lk)-ss)
                {
                    //Negative samps
                    if (pad_mode==0)        //zero pad
                    {
                        K += -ss;
                    }
                    else if (pad_mode==1)   //repeat pad
                    {
                        for (int k=ss; k<0; ++k, ++K) { sm += *X * *K; }
                    }
                    else if (pad_mode==2)   //reflect pad
                    {
                        X += -ss;
                        for (int k=ss; k<0; ++k, --X, ++K) { sm += *X * *K; }
                    }
                    else                    //circular pad
                    {
                        X += (int)Li + ss;
                        for (int k=ss; k<0; ++k, ++X, ++K) { sm += *X * *K; }
                        X -= Li;
                    }

                    //Nonnegative samps
                    for (int k=(int)Lk+ss; k>0; --k, ++X, ++K)
                    {
                        sm += *X * *K;
                    }
                }
                *Y = sm;
            }
            ss += str; ++l;
            K -= Nk; B -= No; Y -= No*Lo-1u;
        }
        es = ss + (int)Lk;

        //Kernel fully within X
        while (es<(int)Li && l<Lo)
        {
            //Loop over output chans
            for (size_t o=No; o>0u; --o, ++B, X-=Ni*Li, Y+=Lo)
            {
                sm = *B;
                for (size_t i=Ni; i>0u; --i, X+=Li-Lk)
                {
                    for (size_t k=Lk; k>0u; --k, ++X, ++K) { sm += *X * *K; }
                }
                *Y = sm;
            }
            es += str; ++l;
            X += str; K -= Nk; B -= No; Y -= No*Lo-1u;
        }
        ss = es - (int)Lk;

        //Kernel overlaps end samp
        while (l<Lo)
        {
            //Loop over output chans
            for (size_t o=No; o>0u; --o, ++B, X-=Ni*Li, Y+=Lo)
            {
                sm = *B;
                for (size_t i=Ni; i>0u; --i, X+=Li-Lk)
                {
                    //Valid samps
                    for (size_t k=Li-(size_t)ss; k>0u; --k, ++X, ++K)
                    {
                        sm += *X * *K;
                    }

                    //Nonvalid samps
                    if (pad_mode==0)        //zero pad
                    {
                        X += es - (int)Li + 1;
                        K += es - (int)Li + 1;
                    }
                    else if (pad_mode==1)   //repeat pad
                    {
                        --X;
                        for (size_t k=(size_t)es-Li+1u; k>0u; --k, ++K) { sm += *X * *K; }
                        X += es - (int)Li + 2;
                    }
                    else if (pad_mode==2)   //reflect pad
                    {
                        X -= 2;
                        for (size_t k=(size_t)es-Li+1u; k>0u; --k, --X, ++K) { sm += *X * *K; }
                        X += 2*(es-(int)Li+2);
                    }
                    else                    //circular pad
                    {
                        X -= Li;
                        for (size_t k=(size_t)es-Li+1u; k>0u; --k, ++X, ++K) { sm += *X * *K; }
                        X += (int)(Li+Lk) + ss - es - 1;
                    }
                }
                *Y = sm;
            }
            ss += str; es += str; ++l;
            X += str; K -= Nk; B -= No; Y -= No*Lo-1u;
        }
        X -= ss;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
