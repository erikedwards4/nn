//Includes
//#include <chrono>
#include "fukushima2.c"

//Declarations
const valarray<size_t> oktypes = {1u,2u};
const size_t I = 2u, O = 1u;
size_t dim, N, T;

//Description
string descr;
descr += "Neural soma stage.\n";
descr += "Does Fukushima model for driving inputs Xe and Xi.\n";
descr += "\n";
descr += "Xe is the excitatory input and Xi is the inhibitory input.\n";
descr += "Both have size NxT or TxN, where N is the number of neurons,\n";
descr += "and T is the number of observations (e.g. time points).\n";
descr += "\n";
descr += "Use -d (--dim) to specify the dimension (axis) of length N [default=0].\n";
descr += "\n";
descr += "This model divides the excitatory (Xe) by the inhibitory (Xi) input.\n";
descr += "This is a quick model of shunting inhibition at the soma level.\n";
descr += "By the original model, the input would come from linear0,\n";
descr += "where there should be separate We and Wi weight matrices.\n";
descr += "\n";
descr += "For dim=0, Y[n,t] = (1+Xe[n,t])/(1+Xi[n,t]) - 1. \n";
descr += "with sizes Xe: N x T \n";
descr += "           Xi: N x T \n";
descr += "           Y:  N x T \n";
descr += "\n";
descr += "For dim=1, Y[t,n] = (1+Xe[t,n])/(1+Xi[t,n]) - 1. \n";
descr += "with sizes Xe: T x N \n";
descr += "           Xi: T x N \n";
descr += "           Y:  T x N \n";
descr += "\n";
descr += "The output Y should be passed through an activation function with range [0 1].\n";
descr += "The original model used a ReLU activation function, but that is not \n";
descr += "included here so that other activation functions can be tried. \n";
descr += "\n";
descr += "Examples:\n";
descr += "$ fukushima2 Xe Xi -o Y \n";
descr += "$ fukushima2 Xe Xi > Y \n";
descr += "$ cat Xi | fukushima2 Xe - > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (Xe,Xi)");
struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension (0 or 1) [default=0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get dim
if (a_d->count==0) { dim = 0u; }
else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
else { dim = size_t(a_d->ival[0]); }
if (dim>1u) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1}" << endl; return 1; }

//Checks
if (i1.T!=i2.T) { cerr << progstr+": " << __LINE__ << errstr << "inputs must have the same data type" << endl; return 1; }
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (Xe) found to be empty" << endl; return 1; }
if (i2.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (Xi) found to be empty" << endl; return 1; }
if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (Xe) must be a matrix" << endl; return 1; }
if (!i2.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (Xi) must be a matrix" << endl; return 1; }
if (i1.R!=i2.R || i1.C!=i2.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs (Xe, Xi) must have the same size" << endl; return 1; }

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = i1.R; o1.C = i1.C; o1.S = i1.S; o1.H = i1.H;

//Other prep
N = (dim==0u) ? o1.R : o1.C;
T = (dim==0u) ? o1.C : o1.R;

//Process
if (i1.T==1u)
{
    float *Xe, *Xi; //*Y;
    try { Xe = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (Xe)" << endl; return 1; }
    try { Xi = new float[i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (Xi)" << endl; return 1; }
    //try { Y = new float[o1.N()]; }
    //catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(Xe),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (Xe)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(Xi),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (Xi)" << endl; return 1; }
    //auto tic = chrono::high_resolution_clock::now();
    //if (codee::fukushima_s(Y,X,N,T,i1.iscolmajor(),dim))
    if (codee::fukushima2_inplace_s(Xe,Xi,N,T,i1.iscolmajor(),dim))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; } 
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Xe),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] Xe; delete[] Xi;
    //auto toc = chrono::high_resolution_clock::now();
    //auto dur = chrono::duration_cast<chrono::microseconds>(toc-tic);
    //cerr << dur.count()/1000.0 << " ms" << endl; 
}

//Finish
