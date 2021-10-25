//1D convolution using PyTorch shape conventions for X, K, Y.
//Shapes are described below for row-major case.

//X is the input of size Nb x Ni x Li.
//K is the tensor of convolving kernels with size No x Ni x Lk.
//Y is the output of size Nb x No x Lo,
//where Lo==L_out is set by:
//Lo =  ceil[1 + (Li + 2*pad - dil*(Lk-1) - 1)/stride], if ceil_mode is true
//Lo = floor[1 + (Li + 2*pad - dil*(Lk-1) - 1)/stride], if ceil_mode is false [default]

//The following params/opts are included:
//Ni:           size_t  num input channels (leading dim of X)
//No:           size_t  num output channels (leading dim of Y)
//Nb:           size_t  batch size (trailing dim of X and Y)
//Li:           size_t  length of input time-series (other dim of X)
//Lk:           size_t  length of kernel in time (num cols of K)
//pad:          int     padding length in [1-Li Li] (pad>0 means add extra samps)
//str:          size_t  stride length (step-size) in samps
//dil:          size_t  dilation factor in samps
//ceil_mode:    bool    calculate Lo with ceil (see above)
//pad_mode:     int     padding mode (0=zeros, 1=replicate, 2=reflect, 3=circular)

#include <stdio.h>

#ifdef __cplusplus
namespace codee {
extern "C" {
#endif

int conv1d_torch_s (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Ng, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);
int conv1d_torch_d (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Ng, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode);


int conv1d_torch_s (float *Y, const float *X, const float *K, const float *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Ng, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1d_torch_s: str (stride) must be positive\n"); return 1; }
    if (dil<1u) { fprintf(stderr,"error in conv1d_torch_s: dil (dilation) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1d_torch_s: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1d_torch_s: Lk (kernel_size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1d_torch_s: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1d_torch_s: No (num output neurons) must be positive\n"); return 1; }
    if (Nb<1u) { fprintf(stderr,"error in conv1d_torch_s: Nb (batch size) must be positive\n"); return 1; }
    if (Ng<1u) { fprintf(stderr,"error in conv1d_torch_s: Ng (num groups) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in conv1d_torch_s: pad length must be > -Li\n"); return 1; }
    if (pad_mode && pad>(int)Li) { fprintf(stderr,"error in conv1d_torch_s: Li (length of input vecs) must be >= pad length\n"); return 1; }
    if (pad_mode<0 || pad_mode>3) { fprintf(stderr,"error in conv1d_torch_s: pad_mode must be an int in {0,1,2,3}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in conv1d_torch_s: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in conv1d_torch_s: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-Tk)/str + (size_t)(ceil_mode && (size_t)(Ti-Tk)%str);
    
    const size_t Gi = Ni/Ng;                //num input chans per group
    const size_t Go = No/Ng;                //num output chans per group
    const size_t Nk = Gi*Lk*No;             //total num elements in K
    float sm;                               //intermediate accumulation
    int ss, es;                             //current start-samp, end-samp
    size_t l;                               //current samp within Y in [0 Lo-1]

    //Loop over batch members
    for (size_t b=Nb; b>0u; --b)
    {
        //Reset samp counts
        l = 0u; ss = -pad; es = ss + Tk - 1;

        //K before or overlapping first samp of X
        while (ss<0 && l<Lo)
        {
            //Get num valid samps
            const size_t V = (es<0) ? 0u : 1u + (size_t)(es-es%(int)dil)/dil;

            //Loop over groups
            for (size_t g=Ng; g>0u; --g, X+=Gi*Li)
            {
                //Loop over output chans
                for (size_t o=Go; o>0u; --o, ++B, X-=Gi*Li, Y+=Lo)
                {
                    sm = *B;
                    for (size_t i=Gi; i>0u; --i, X+=Li)
                    {
                        //Negative samps
                        if (pad_mode==0)        //zero pad
                        {
                            K += Lk - V;
                        }
                        else if (pad_mode==1)   //repeat pad
                        {
                            for (size_t k=Lk-V; k>0u; --k, ++K) { sm += *X * *K; }
                        }
                        else if (pad_mode==2)   //reflect pad
                        {
                            X -= ss;
                            for (size_t k=Lk-V; k>0u; --k, X-=dil, ++K) { sm += *X * *K; }
                            X += ss + (int)((Lk-V)*dil);
                        }
                        else                    //circular pad
                        {
                            X += (int)Li + ss;
                            for (size_t k=Lk-V; k>0u; --k, X+=dil, ++K) { sm += *X * *K; }
                            X -= (int)Li + ss + (int)((Lk-V)*dil);
                        }

                        //Valid samps
                        if (es>=0)
                        {
                            X += (size_t)es%dil;
                            for (int k=V; k>0; --k, X+=dil, ++K)
                            {
                                sm += *X * *K;
                            }
                            X -= (size_t)es%dil + V*dil;
                        }
                    }
                    *Y = sm;
                }
            }
            K -= Nk; B -= No;
            X -= Ni*Li; Y -= No*Lo-1u;
            ss += str; es += str; ++l;
        }
        X += ss;

        //Kernel fully within X
        while (es<(int)Li && l<Lo)
        {
            //Loop over groups
            for (size_t g=Ng; g>0u; --g, X+=Gi*Li)
            {
                //Loop over output chans
                for (size_t o=Go; o>0u; --o, ++B, X-=Gi*Li, Y+=Lo)
                {
                    sm = *B;
                    for (size_t i=Gi; i>0u; --i, X+=Li-Lk*dil)
                    {
                        for (size_t k=Lk; k>0u; --k, X+=dil, ++K) { sm += *X * *K; }
                    }
                    *Y = sm;
                }
            }
            K -= Nk; B -= No;
            X -= Ni*Li-str; Y -= No*Lo-1u;
            es += str; ++l;
        }
        ss = es - Tk + 1;

        //Kernel overlaps end samp
        while (l<Lo)
        {
            //Get num valid samps
            const size_t V = (ss<(int)Li) ? 1u + (Li-1u-(size_t)ss)/dil : 0u;

            //Loop over groups
            for (size_t g=Ng; g>0u; --g, X+=Gi*Li)
            {
                //Loop over output chans
                for (size_t o=Go; o>0u; --o, ++B, X-=Gi*Li, Y+=Lo)
                {
                    sm = *B;
                    for (size_t i=Gi; i>0u; --i, X+=Li-Lk*dil)
                    {
                        //Valid samps
                        for (size_t k=V; k>0u; --k, X+=dil, ++K)
                        {
                            sm += *X * *K;
                        }

                        //Nonvalid samps
                        if (pad_mode==0)        //zero pad
                        {
                            X += (Lk-V) * dil;
                            K += Lk - V;
                        }
                        else if (pad_mode==1)   //repeat pad
                        {
                            X += (int)Li - 1 - ss - (int)(V*dil);
                            for (size_t k=Lk-V; k>0u; --k, ++K) { sm += *X * *K; }
                            X -= (int)Li - 1 - ss - (int)(Lk*dil);
                        }
                        else if (pad_mode==2)   //reflect pad
                        {
                            X -= 2 * (ss+(int)(V*dil)-(int)Li+1);
                            for (size_t k=Lk-V; k>0u; --k, X-=dil, ++K) { sm += *X * *K; }
                            X += 2 * (ss-(int)Li+1+(int)(Lk*dil));
                        }
                        else                    //circular pad
                        {
                            X -= Li;
                            for (size_t k=Lk-V; k>0u; --k, X+=dil, ++K) { sm += *X * *K; }
                            X += Li;
                        }
                    }
                    *Y = sm;
                }
            }
            K -= Nk; B -= No; 
            X -= Ni*Li-str; Y -= No*Lo-1u;
            ss += str; es += str; ++l;
        }

        //Go to start of next batch member
        X += (int)(Ni*Li) - ss;
        Y += (No-1u)*Lo;
    }

    return 0;
}


int conv1d_torch_d (double *Y, const double *X, const double *K, const double *B, const size_t Ni, const size_t No, const size_t Nb, const size_t Ng, const size_t Li, const size_t Lk, const int pad, const size_t str, const size_t dil, const int ceil_mode, const int pad_mode)
{
    if (str<1u) { fprintf(stderr,"error in conv1d_torch_d: str (stride) must be positive\n"); return 1; }
    if (dil<1u) { fprintf(stderr,"error in conv1d_torch_d: dil (dilation) must be positive\n"); return 1; }
    if (Li<1u) { fprintf(stderr,"error in conv1d_torch_d: Li (length of input vecs) must be positive\n"); return 1; }
    if (Lk<1u) { fprintf(stderr,"error in conv1d_torch_d: Lk (kernel_size) must be positive\n"); return 1; }
    if (Ni<1u) { fprintf(stderr,"error in conv1d_torch_d: Ni (num input neurons) must be positive\n"); return 1; }
    if (No<1u) { fprintf(stderr,"error in conv1d_torch_d: No (num output neurons) must be positive\n"); return 1; }
    if (Nb<1u) { fprintf(stderr,"error in conv1d_torch_d: Nb (batch size) must be positive\n"); return 1; }
    if (Ng<1u) { fprintf(stderr,"error in conv1d_torch_d: Ng (num groups) must be positive\n"); return 1; }
    if (pad<=-(int)Li) { fprintf(stderr,"error in conv1d_torch_d: pad length must be > -Li\n"); return 1; }
    if (pad_mode && pad>(int)Li) { fprintf(stderr,"error in conv1d_torch_d: Li (length of input vecs) must be >= pad length\n"); return 1; }
    if (pad_mode<0 || pad_mode>3) { fprintf(stderr,"error in conv1d_torch_d: pad_mode must be an int in {0,1,2,3}\n"); return 1; }

    const int Ti = (int)Li + 2*pad;         //total length of vecs in X including padding
    const int Tk = (int)(dil*(Lk-1u)) + 1;  //total length of vecs in K including dilation
    if (Tk>(int)Li) { fprintf(stderr,"error in conv1d_torch_d: Li (length of input vecs) must be >= dil*(Lk-1)\n"); return 1; }
    if (Ti<Tk) { fprintf(stderr,"error in conv1d_torch_d: Li+2*pad must be >= dil*(Lk-1)\n"); return 1; }

    //Set Lo (L_out, output length) according to ceil mode
    const size_t Lo = 1u + (size_t)(Ti-Tk)/str + (size_t)(ceil_mode && (size_t)(Ti-Tk)%str);
    
    const size_t Gi = Ni/Ng;                //num input chans per group
    const size_t Go = No/Ng;                //num output chans per group
    const size_t Nk = Gi*Lk*No;             //total num elements in K
    double sm;                              //intermediate accumulation
    int ss, es;                             //current start-samp, end-samp
    size_t l;                               //current samp within Y in [0 Lo-1]

    //Loop over batch members
    for (size_t b=Nb; b>0u; --b)
    {
        //Reset samp counts
        l = 0u; ss = -pad; es = ss + Tk - 1;

        //K before or overlapping first samp of X
        while (ss<0 && l<Lo)
        {
            //Get num valid samps
            const size_t V = (es<0) ? 0u : 1u + (size_t)(es-es%(int)dil)/dil;

            //Loop over groups
            for (size_t g=Ng; g>0u; --g, X+=Gi*Li)
            {
                //Loop over output chans
                for (size_t o=Go; o>0u; --o, ++B, X-=Gi*Li, Y+=Lo)
                {
                    sm = *B;
                    for (size_t i=Gi; i>0u; --i, X+=Li)
                    {
                        //Negative samps
                        if (pad_mode==0)        //zero pad
                        {
                            K += Lk - V;
                        }
                        else if (pad_mode==1)   //repeat pad
                        {
                            for (size_t k=Lk-V; k>0u; --k, ++K) { sm += *X * *K; }
                        }
                        else if (pad_mode==2)   //reflect pad
                        {
                            X -= ss;
                            for (size_t k=Lk-V; k>0u; --k, X-=dil, ++K) { sm += *X * *K; }
                            X += ss + (int)((Lk-V)*dil);
                        }
                        else                    //circular pad
                        {
                            X += (int)Li + ss;
                            for (size_t k=Lk-V; k>0u; --k, X+=dil, ++K) { sm += *X * *K; }
                            X -= (int)Li + ss + (int)((Lk-V)*dil);
                        }

                        //Valid samps
                        if (es>=0)
                        {
                            X += (size_t)es%dil;
                            for (int k=V; k>0; --k, X+=dil, ++K)
                            {
                                sm += *X * *K;
                            }
                            X -= (size_t)es%dil + V*dil;
                        }
                    }
                    *Y = sm;
                }
            }
            K -= Nk; B -= No;
            X -= Ni*Li; Y -= No*Lo-1u;
            ss += str; es += str; ++l;
        }
        X += ss;

        //Kernel fully within X
        while (es<(int)Li && l<Lo)
        {
            //Loop over groups
            for (size_t g=Ng; g>0u; --g, X+=Gi*Li)
            {
                //Loop over output chans
                for (size_t o=Go; o>0u; --o, ++B, X-=Gi*Li, Y+=Lo)
                {
                    sm = *B;
                    for (size_t i=Gi; i>0u; --i, X+=Li-Lk*dil)
                    {
                        for (size_t k=Lk; k>0u; --k, X+=dil, ++K) { sm += *X * *K; }
                    }
                    *Y = sm;
                }
            }
            K -= Nk; B -= No;
            X -= Ni*Li-str; Y -= No*Lo-1u;
            es += str; ++l;
        }
        ss = es - Tk + 1;

        //Kernel overlaps end samp
        while (l<Lo)
        {
            //Get num valid samps
            const size_t V = (ss<(int)Li) ? 1u + (Li-1u-(size_t)ss)/dil : 0u;

            //Loop over groups
            for (size_t g=Ng; g>0u; --g, X+=Gi*Li)
            {
                //Loop over output chans
                for (size_t o=Go; o>0u; --o, ++B, X-=Gi*Li, Y+=Lo)
                {
                    sm = *B;
                    for (size_t i=Gi; i>0u; --i, X+=Li-Lk*dil)
                    {
                        //Valid samps
                        for (size_t k=V; k>0u; --k, X+=dil, ++K)
                        {
                            sm += *X * *K;
                        }

                        //Nonvalid samps
                        if (pad_mode==0)        //zero pad
                        {
                            X += (Lk-V) * dil;
                            K += Lk - V;
                        }
                        else if (pad_mode==1)   //repeat pad
                        {
                            X += (int)Li - 1 - ss - (int)(V*dil);
                            for (size_t k=Lk-V; k>0u; --k, ++K) { sm += *X * *K; }
                            X -= (int)Li - 1 - ss - (int)(Lk*dil);
                        }
                        else if (pad_mode==2)   //reflect pad
                        {
                            X -= 2 * (ss+(int)(V*dil)-(int)Li+1);
                            for (size_t k=Lk-V; k>0u; --k, X-=dil, ++K) { sm += *X * *K; }
                            X += 2 * (ss-(int)Li+1+(int)(Lk*dil));
                        }
                        else                    //circular pad
                        {
                            X -= Li;
                            for (size_t k=Lk-V; k>0u; --k, X+=dil, ++K) { sm += *X * *K; }
                            X += Li;
                        }
                    }
                    *Y = sm;
                }
            }
            K -= Nk; B -= No; 
            X -= Ni*Li-str; Y -= No*Lo-1u;
            ss += str; es += str; ++l;
        }

        //Go to start of next batch member
        X += (int)(Ni*Li) - ss;
        Y += (No-1u)*Lo;
    }

    return 0;
}


#ifdef __cplusplus
}
}
#endif
