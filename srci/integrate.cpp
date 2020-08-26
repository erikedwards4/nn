//Includes
#include "integrate.c"

//Declarations
const valarray<size_t> oktypes = {1u,2u,101u,102u};
const size_t I = 2u, O = 1u;
size_t dim, N, T;
double fs;

//Description
string descr;
descr += "Neural soma stage.\n";
descr += "Does simple temporal integration (1st-order IIR filter),\n";
descr += "with time-constant tau, for each row or col of X.\n";
descr += "\n";
descr += "X has size NxT or TxN, where N is the number of neurons\n";
descr += "and T is the number of observations (e.g. time points).\n";
descr += "\n";
descr += "Use -d (--dim) to specify the dimension (axis) of length N [default=0].\n";
descr += "\n";
descr += "For dim=0, Y[n,t] = a[n]*Y[n,t-1] + b[n]*X[n,t]. \n";
descr += "with sizes X:  N x T \n";
descr += "           Y:  N x T \n";
descr += "\n";
descr += "For dim=1, Y[t,n] = a[n]*Y[t-1,n] + b[n]*X[t,n]. \n";
descr += "with sizes X:  T x N \n";
descr += "           Y:  T x N \n";
descr += "\n";
descr += "where a[n] = exp(-1/(fs*taus[n])) and b[n] = 1 - a[n].\n";
descr += "\n";
descr += "Use -s (--fs) to give the sample rate of X in Hz [default=10000].\n";
descr += "\n";
descr += "Enter a vector of N taus as the 2nd input.\n";
descr += "Or enter a single tau as the 2nd input, to be used by all N neurons.\n";
descr += "For an RC circuit: tau = R*C.\n";
descr += "The cutoff frequeny in Hz is: f_c = 1/(2*pi*tau).\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ integrate X taus -o Y \n";
descr += "$ integrate X taus > Y \n";
descr += "$ cat X | integrate - taus > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X,taus)");
struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension (0 or 1) [default=0]");
struct arg_dbl   *a_fs = arg_dbln("s","fs","<dbl>",0,1,"sample rate of X [default=10000]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get dim
if (a_d->count==0) { dim = (i1.C==1u) ? 1 : 0; }
else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
else { dim = size_t(a_d->ival[0]); }
if (dim>1u) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1}" << endl; return 1; }

//Get fs
fs = (a_fs->count>0) ? a_fs->dval[0] : 10000.0;
if (fs<=0.0) { cerr << progstr+": " << __LINE__ << errstr << "fs (sample rate) must be positive" << endl; return 1; }

//Checks
if (i1.T!=i2.T && i1.T-100u!=i2.T) { cerr << progstr+": " << __LINE__ << errstr << "inputs must have compatible data types" << endl; return 1; }
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) found to be empty" << endl; return 1; }
if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) must be a matrix" << endl; return 1; }
if (i2.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (taus) found to be empty" << endl; return 1; }
if (!i2.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (taus) must be a vector or scalar" << endl; return 1; }
if (i2.N()!=1u)
{
    if ((dim==0u && i2.N()!=i1.R) || (dim==1u && i2.N()!=i1.C))
    { cerr << progstr+": " << __LINE__ << errstr << "length taus must equal N (num neurons)" << endl; return 1; }
}

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = i1.R; o1.C = i1.C; o1.S = i1.S; o1.H = i1.H;

//Other prep
N = (dim==0u) ? o1.R : o1.C;
T = (dim==0u) ? o1.C : o1.R;
if (T<2u) { cerr << progstr+": " << __LINE__ << errstr << "num time points must be > 1" << endl; return 1; }

//Process
if (i1.T==1u)
{
    float *X, *taus; // *Y;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
    try { taus = new float[i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (taus)" << endl; return 1; }
    //try { Y = new float[o1.N()]; }
    //catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(taus),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (taus)" << endl; return 1; }
    if (i2.N()==1u) { for (size_t n=1u; n<N; ++n) { taus[n] = taus[0]; } }
    //if (codee::integrate_s(Y,X,taus,N,T,i1.iscolmajor(),dim,float(fs)))
    if (codee::integrate_inplace_s(X,taus,N,T,i1.iscolmajor(),dim,float(fs)))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; } 
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X; delete[] taus; //delete[] Y;
}
else if (i1.T==101u)
{
    float *X, *taus;
    try { X = new float[2u*i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
    try { taus = new float[i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (taus)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(taus),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (taus)" << endl; return 1; }
    if (i2.N()==1u) { for (size_t n=1u; n<N; ++n) { taus[n] = taus[0]; } }
    if (codee::integrate_inplace_c(X,taus,N,T,i1.iscolmajor(),dim,float(fs)))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; } 
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X; delete[] taus;
}

//Finish
