//Includes
//#include <chrono>
#include <cblas.h>
#include "fukushima.c"

//Declarations
const valarray<uint8_t> oktypes = {1,2};
const size_t I = 1, O = 1;
int dim, N, T;

//Description
string descr;
descr += "Neural soma stage.\n";
descr += "Does Fukushima model for driving inputs X.\n";
descr += "\n";
descr += "X has size 2NxT or Tx2N, where N is the number of neurons\n";
descr += "and T is the number of observations (e.g. time points).\n";
descr += "\n";
descr += "Use -d (--dim) to specify the dimension (axis) of length 2N [default=0].\n";
descr += "\n";
descr += "This model divides the excitatory (E) part of X by the inhibitory (I) part.\n";
descr += "as a quick model of shunting inhibition at the soma level.\n";
descr += "Thus, the input X has 2 time-series per neuron, one for E and one for I.\n";
descr += "By the original model, the input would come from wx, i.e. apply weights W\n";
descr += "by matrix multiplication. In this case, W should also have E and I parts.\n";
descr += "\n";
descr += "For dim=0, Y[n,t] = (1+X[n,t])/(1+X[n+N,t]) - 1. \n";
descr += "with sizes X: 2N x T \n";
descr += "           Y:  N x T \n";
descr += "\n";
descr += "For dim=1, Y[t,n] = (1+X[t,n])/(1+X[t,n+N]) - 1. \n";
descr += "with sizes X:  T x 2N \n";
descr += "           Y:  T x N  \n";
descr += "\n";
descr += "X has 2N time-series because it stacks E and I parts into one matrix:\n";
descr += "X = [Xe; Xi] for dim=0, and X = [Xe Xi] for dim=1. \n";
descr += "\n";
descr += "The output Y should be passed through an activation function with range [0 1].\n";
descr += "The original model used a ReLU activation function, but that is not \n";
descr += "included here so that other activation functions can be tried. \n";
descr += "\n";
descr += "Examples:\n";
descr += "$ fukushima X -o Y \n";
descr += "$ fukushima X > Y \n";
descr += "$ cat X | fukushima > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input file (X)");
struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension (0 or 1) [default=0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get dim
if (a_d->count==0) { dim = 0; }
else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
else { dim = a_d->ival[0]; }
if (dim>1) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1}" << endl; return 1; }

//Checks
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) found to be empty" << endl; return 1; }
if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) must be a matrix" << endl; return 1; }
if (dim==0 && i1.R%2u) { cerr << progstr+": " << __LINE__ << errstr << "num rows X must be even for dim=0" << endl; return 1; }
if (dim==1 && i1.C%2u) { cerr << progstr+": " << __LINE__ << errstr << "num cols X must be even for dim=1" << endl; return 1; }

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = (dim==0) ? i1.R/2u : i1.R;
o1.C = (dim==1) ? i1.C/2u : i1.C;
o1.S = i1.S; o1.H = i1.H;

//Other prep
N = (dim==0) ? int(o1.R) : int(o1.C);
T = (dim==0) ? int(o1.C) : int(o1.R);

//Process
if (i1.T==1)
{
    float *X; //*Y;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
    //try { Y = new float[o1.N()]; }
    //catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
    //auto tic = chrono::high_resolution_clock::now();
    //if (openn::fukushima_s(Y,X,N,T,dim,i1.iscolmajor()))
    if (openn::fukushima_inplace_s(X,N,T,dim,i1.iscolmajor()))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; } 
    if (wo1)
    {
        //try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        //catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        if ((dim==0 && o1.isrowmajor()) || (dim==1 && o1.iscolmajor()))
        {
            try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        else
        {
            float *Y;
            try { Y = new float[o1.N()]; }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
            for (int t=0; t<T; t++)
            {
                try { cblas_scopy(N,&X[2*t*N],1,&Y[t*N],1); }
                catch(...) { cerr << progstr+": " << __LINE__ << errstr << "problem copying to output file (Y)" << endl; return 1; }
            }
            try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
            delete[] Y;
        }
    }
    delete[] X; //delete[] Y;
    //auto toc = chrono::high_resolution_clock::now();
    //auto dur = chrono::duration_cast<chrono::microseconds>(toc-tic);
    //cerr << dur.count()/1000.0 << " ms" << endl; 
}

//Finish

