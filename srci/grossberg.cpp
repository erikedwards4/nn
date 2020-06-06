//Includes
#include "grossberg.c"

//Declarations
const valarray<uint8_t> oktypes = {1,2,101,102};
const size_t I = 5, O = 1;
int dim, N, T;
double fs;

//Description
string descr;
descr += "Neural soma stage.\n";
descr += "Does middle stage of Grossberg model for each row or col of X.\n";
descr += "This temporal integration (1st-order IIR filter), with time-constant tau,.\n";
descr += "along with a subtractive feedback with gain alpha,\n";
descr += "along with adaptive gain control with parameters beta and gamma.\n";
descr += "\n";
descr += "X has size NxT or TxN, where N is the number of neurons\n";
descr += "and T is the number of observations (e.g. time points).\n";
descr += "\n";
descr += "Use -d (--dim) to specify the dimension (axis) of length N [default=0].\n";
descr += "\n";
descr += "For dim=0, Y[n,t] = a[n]*Y[n,t-1] + b[n]*(g[n,t]*X[n,t]-alpha[n]*Y[n,t-1]). \n";
descr += "with sizes X:  N x T \n";
descr += "           Y:  N x T \n";
descr += "\n";
descr += "For dim=1, Y[t,n] = a[n]*Y[t-1,n] + b[n]*(g[t,n]*X[t,n]-alpha[n]*Y[t-1,n]). \n";
descr += "with sizes X:  T x N \n";
descr += "           Y:  T x N \n";
descr += "\n";
descr += "where a[n] = exp(-1/(fs*taus[n])) and b[n] = 1 - a[n].\n";
descr += "and g[n,t] = gamma[n] - beta[n]*Y[n,t-1].\n";
descr += "\n";
descr += "Use -s (--fs) to give the sample rate of X in Hz [default=10000].\n";
descr += "\n";
descr += "Enter a vector of N taus as the 2nd input.\n";
descr += "Or enter a single tau as the 2nd input, to be used by all N neurons.\n";
descr += "\n";
descr += "Enter a vector of N alphas as the 3rd input.\n";
descr += "Or enter a single alpha as the 3rd input, to be used by all N neurons.\n";
descr += "\n";
descr += "Enter a vector of N betas as the 4th input.\n";
descr += "Or enter a single beta as the 4th input, to be used by all N neurons.\n";
descr += "\n";
descr += "Enter a vector of N gammas as the 5th input.\n";
descr += "Or enter a single gamma as the 5th input, to be used by all N neurons.\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ grossberg X taus alphas betas gammas -o Y \n";
descr += "$ grossberg X taus alphas betas gammas > Y \n";
descr += "$ cat X | grossberg - taus alphas betas gammas > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X,taus,alphas,betas,gammas)");
struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension (0 or 1) [default=0]");
struct arg_dbl   *a_fs = arg_dbln("s","fs","<dbl>",0,1,"sample rate of X [default=10000]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get dim
if (a_d->count==0) { dim = (i1.C==1u) ? 1 : 0; }
else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
else { dim = a_d->ival[0]; }
if (dim>1) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1}" << endl; return 1; }

//Get fs
fs = (a_fs->count>0) ? a_fs->dval[0] : 10000.0;
if (fs<=0.0) { cerr << progstr+": " << __LINE__ << errstr << "fs (sample rate) must be positive" << endl; return 1; }

//Checks
if (i1.T!=i2.T && i1.T-100!=i2.T) { cerr << progstr+": " << __LINE__ << errstr << "inputs must have compatible data types" << endl; return 1; }
if (i1.T!=i3.T && i1.T-100!=i3.T) { cerr << progstr+": " << __LINE__ << errstr << "inputs must have compatible data types" << endl; return 1; }
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) found to be empty" << endl; return 1; }
if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) must be a matrix" << endl; return 1; }
if (i2.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (taus) found to be empty" << endl; return 1; }
if (!i2.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (taus) must be a vector or scalar" << endl; return 1; }
if (i3.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (alphas) found to be empty" << endl; return 1; }
if (!i3.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (alphas) must be a vector or scalar" << endl; return 1; }
if (i4.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (betas) found to be empty" << endl; return 1; }
if (!i4.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (betas) must be a vector or scalar" << endl; return 1; }
if (i5.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 5 (gammas) found to be empty" << endl; return 1; }
if (!i5.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 5 (gammas) must be a vector or scalar" << endl; return 1; }
if (i2.N()!=1u)
{
    if ((dim==0 && i2.N()!=i1.R) || (dim==1 && i2.N()!=i1.C))
    { cerr << progstr+": " << __LINE__ << errstr << "length taus must equal N (num neurons)" << endl; return 1; }
}
if (i3.N()!=1u)
{
    if ((dim==0 && i3.N()!=i1.R) || (dim==1 && i3.N()!=i1.C))
    { cerr << progstr+": " << __LINE__ << errstr << "length alphas must equal N (num neurons)" << endl; return 1; }
}
if (i4.N()!=1u)
{
    if ((dim==0 && i4.N()!=i1.R) || (dim==1 && i4.N()!=i1.C))
    { cerr << progstr+": " << __LINE__ << errstr << "length betas must equal N (num neurons)" << endl; return 1; }
}
if (i5.N()!=1u)
{
    if ((dim==0 && i5.N()!=i1.R) || (dim==1 && i5.N()!=i1.C))
    { cerr << progstr+": " << __LINE__ << errstr << "length gammas must equal N (num neurons)" << endl; return 1; }
}

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = i1.R; o1.C = i1.C; o1.S = i1.S; o1.H = i1.H;

//Other prep
N = (dim==0) ? int(o1.R) : int(o1.C);
T = (dim==0) ? int(o1.C) : int(o1.R);
if (T<2) { cerr << progstr+": " << __LINE__ << errstr << "num time points must be > 1" << endl; return 1; }

//Process
if (i1.T==1)
{
    float *X, *taus, *alphas, *betas, *gammas;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
    try { taus = new float[i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (taus)" << endl; return 1; }
    try { alphas = new float[i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (alphas)" << endl; return 1; }
    try { betas = new float[i4.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 4 (betas)" << endl; return 1; }
    try { gammas = new float[i5.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 5 (gammas)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(taus),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (taus)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(alphas),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (alphas)" << endl; return 1; }
    try { ifs4.read(reinterpret_cast<char*>(betas),i4.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 4 (betas)" << endl; return 1; }
    try { ifs5.read(reinterpret_cast<char*>(gammas),i5.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 5 (gammas)" << endl; return 1; }
    if (i2.N()==1u) { for (int n=1; n<N; n++) { taus[n] = taus[0]; } }
    if (i3.N()==1u) { for (int n=1; n<N; n++) { alphas[n] = alphas[0]; } }
    if (i4.N()==1u) { for (int n=1; n<N; n++) { betas[n] = betas[0]; } }
    if (i5.N()==1u) { for (int n=1; n<N; n++) { gammas[n] = gammas[0]; } }
    if (openn::grossberg_inplace_s(X,taus,alphas,betas,gammas,N,T,dim,i1.iscolmajor(),float(fs)))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; } 
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X; delete[] taus; delete[] alphas; delete[] betas; delete[] gammas;
}
else if (i1.T==101)
{
    float *X, *taus, *alphas, *betas, *gammas;
    try { X = new float[2u*i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
    try { taus = new float[i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (taus)" << endl; return 1; }
    try { alphas = new float[i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (alphas)" << endl; return 1; }
    try { betas = new float[i4.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 4 (betas)" << endl; return 1; }
    try { gammas = new float[i5.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 5 (gammas)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(taus),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (taus)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(alphas),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (alphas)" << endl; return 1; }
    try { ifs4.read(reinterpret_cast<char*>(betas),i4.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 4 (betas)" << endl; return 1; }
    try { ifs5.read(reinterpret_cast<char*>(gammas),i5.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 5 (gammas)" << endl; return 1; }
    if (i2.N()==1u) { for (int n=1; n<N; n++) { taus[n] = taus[0]; } }
    if (i3.N()==1u) { for (int n=1; n<N; n++) { alphas[n] = alphas[0]; } }
    if (i4.N()==1u) { for (int n=1; n<N; n++) { betas[n] = betas[0]; } }
    if (i5.N()==1u) { for (int n=1; n<N; n++) { gammas[n] = gammas[0]; } }
    if (openn::grossberg_inplace_c(X,taus,alphas,betas,gammas,N,T,dim,i1.iscolmajor(),float(fs)))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; } 
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X; delete[] taus; delete[] alphas; delete[] betas; delete[] gammas;
}

//Finish

